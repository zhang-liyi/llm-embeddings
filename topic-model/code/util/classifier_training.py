import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributions as D
import datetime
import os
import json
from util.metrics import *



def loss_klqp(lda, lda_corpus, doc_id_begin, doc_id_end, q_params, device):
        # Get p
        params = lda.inference(lda_corpus[doc_id_begin: doc_id_end])[0]
        p = D.dirichlet.Dirichlet(torch.tensor(params))
        # Get q
        q = D.dirichlet.Dirichlet(q_params)
        # Compute KLD
        topic_samps = q.sample([1])
        logp = p.log_prob(topic_samps)
        logq = q.log_prob(topic_samps)

        return torch.mean(logq - logp)

def loss_mle(lda, lda_corpus, doc_id_begin, doc_id_end, q_params, device):
    # Get p
    params = lda.inference(lda_corpus[doc_id_begin: doc_id_end])[0]
    p = D.dirichlet.Dirichlet(torch.tensor(params))
    # Samples from target
    samps = p.sample([1])
    # Get q
    q = D.dirichlet.Dirichlet(q_params)
    logq = q.log_prob(samps)

    return torch.mean(-logq)#

def get_doc_embeddings(config, model_outputs, embd_dim, token, train_val='train'):

    if token != 2.5:
        embeddings = torch.zeros((len(model_outputs), embd_dim))
    else:
        embeddings = torch.zeros((len(model_outputs), embd_dim*config['max_len']))

    if token <= 0 or token >= 1 and token != 1.5 and token != 2.5:
        token = int(token)
        for i in range(len(model_outputs)):
            embeddings[i,:] = model_outputs[i][0,token,:]
    elif token == 1.5: # Average
        for i in range(len(model_outputs)):
            embeddings[i,:] = torch.mean(model_outputs[i][0,:,:], dim=0)
    elif token == 2.5: # Concatenate
        for i in range(len(model_outputs)):
            vec = model_outputs[i][0,:,:]
            vec = torch.squeeze(vec.reshape((1, embd_dim*config['max_len'])))
            embeddings[i,:] = vec
    else:
        for i in range(len(model_outputs)):
            if i == 0:
                print(model_outputs[i].shape)
            idx = int(token*len(model_outputs[i][0]))
            embeddings[i,:] = model_outputs[i][0,idx,:]

    return embeddings

def train(config, lda, target_mixtures_t, target_mixtures_v, model_outputs_t, model_outputs_v, lda_corpus_t, lda_corpus_v, token, 
          device, save_appendix=''):

    if 'gpt2' == config['model_type'] or 'gpt2-random' == config['model_type']:
        embd_dim = 768
    elif 'gpt2-medium' == config['model_type']:
        embd_dim = 1024
    elif 'gpt2-large' == config['model_type']:
        embd_dim = 1280
    elif 'llama2' in config['model_type']:
        embd_dim = 4096
    elif 'bert' == config['model_type']:
        embd_dim = 768
    elif 'bert-large' == config['model_type']:
        embd_dim = 1024

    # Define the embeddings (which token) that will train the classifier
    if not config['load_embeddings_all']:
        embeddings_t = get_doc_embeddings(config, model_outputs_t, embd_dim, token, 'train').to(torch.device(device))
        embeddings_v = get_doc_embeddings(config, model_outputs_v, embd_dim, token, 'val').to(torch.device(device))
    else:
        embeddings_t = torch.stack(model_outputs_t, 0)
        embeddings_v = torch.stack(model_outputs_v, 0)
    target_mixtures_t = torch.tensor(target_mixtures_t).to(torch.device(device))
    target_mixtures_v = torch.tensor(target_mixtures_v).to(torch.device(device))
    print('nan', torch.sum(torch.isnan(embeddings_t)))

    ##### Training

    tm = str(datetime.datetime.now())
    TMSTR = tm[:10]+'-'+tm[11:13]+tm[14:16]+tm[17:]
    path = f'results/{TMSTR}/'
    os.makedirs(f'results/{TMSTR}/')
    json_object = json.dumps(config, indent=4)
    with open(path+"parameters.json", "w") as outfile:
        outfile.write(json_object)

    seeds = [1000,2000,3000]
    for seed_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed(seed)
        # Get classifier
        cls_dim = embd_dim if config['token'] != 0.5 else embd_dim*config['max_len']
        if config["use_linear_classifier"]:
            if config["train_mode"] == 'bayesian':
                classifier = torch.nn.Sequential(
                    torch.nn.Linear(cls_dim, 20),
                    torch.nn.Softplus(),
                ).to(torch.device(device))

            elif config["train_mode"] == 'classification':
                classifier = torch.nn.Sequential(
                    torch.nn.Linear(cls_dim, 20)
                ).to(torch.device(device))
        else:
            if config["train_mode"] == 'bayesian':
                classifier = torch.nn.Sequential(
                    torch.nn.Linear(cls_dim, config['cls_hidden_dim']),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config['dropout']),
                    torch.nn.Linear(config['cls_hidden_dim'], 20),
                    torch.nn.Softplus(),
                ).to(torch.device(device))

            elif config["train_mode"] == 'classification':
                classifier = torch.nn.Sequential(
                    torch.nn.Linear(cls_dim, config['cls_hidden_dim']),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(config['dropout']),
                    torch.nn.Linear(config['cls_hidden_dim'], 20)
                ).to(torch.device(device))

        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
        lr = config['learning_rate']  # learning rate
        weight_decay = config['weight_decay']
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
        batch_size = config['batch_size']
        N_t = len(embeddings_t)
        N_v = len(embeddings_v)

        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        train_losses_l2 = []
        train_losses_tv = []
        val_losses_l2 = []
        val_losses_tv = []

        min_val_loss = 1e6

        for epoch in range(300):

            classifier.train()
            epoch_loss_t = 0.
            epoch_loss_t_l2 = 0.
            epoch_loss_t_tv = 0.
            count = 0
            pred_results = []

            for i in range(0, N_t, batch_size):
                end_idx = min(i+batch_size, N_t)

                if config["train_mode"] == 'classification':

                    embd = embeddings_t[i:end_idx, :]
                    target = target_mixtures_t[seed_idx, i:end_idx, :]
                    output = classifier(embd)
                    loss = criterion(output, target)

                elif config["train_mode"] == 'bayesian':

                    embd = embeddings_t[i:end_idx, :]
                    target = target_mixtures_t[seed_idx, i:end_idx, :]
                    output = classifier(embd) + 1e-15
                    loss = loss_mle(lda, lda_corpus_t, i, end_idx, output, device)

                pred_class = torch.argmax(output.cpu(), 1).to(torch.device(device))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if count % 10 == 0:
                #     print(loss)
                #     for param in classifier.parameters():
                #         print(torch.mean(param))

                epoch_loss_t += loss.item()
                epoch_loss_t_l2 += get_loss_l2(output.detach(), target.detach(), True, 'sum').item()
                epoch_loss_t_tv += get_loss_tv(output.detach(), target.detach(), True, 'sum').item()
                count += 1

                true_class = torch.argmax(target.cpu(), 1).to(torch.device(device))
                pred_result = true_class == pred_class
                pred_result = list(pred_result.cpu().numpy())
                pred_results.extend(pred_result)
                
            pred_results = np.array(pred_results)
            acc_t = np.mean(pred_results)
            train_losses.append(epoch_loss_t/N_t)
            train_accs.append(float(acc_t))
            train_losses_l2.append(epoch_loss_t_l2/N_t)
            train_losses_tv.append(epoch_loss_t_tv/N_t)

            print('Total train loss', epoch_loss_t/N_t)
            print('Training accuracy', acc_t)

            classifier.eval()
            epoch_loss_v = 0.
            epoch_loss_v_l2 = 0.
            epoch_loss_v_tv = 0.
            count = 0
            pred_results = []

            val_l2_all = []
            val_tv_all = []

            for i in range(0, N_v, batch_size):
                end_idx = min(i+batch_size, N_v)

                if config["train_mode"] == 'classification':

                    embd = embeddings_v[i:end_idx, :]
                    target = target_mixtures_v[seed_idx, i:end_idx, :]

                    output = classifier(embd)

                    loss = criterion(output, target)

                elif config["train_mode"] == 'bayesian':

                    embd = embeddings_v[i:end_idx, :]
                    target = target_mixtures_v[seed_idx, i:end_idx, :]
                    output = classifier(embd) + 1e-15
                    loss = loss_mle(lda, lda_corpus_v, i, end_idx, output, device)
                
                epoch_loss_v += loss.item()
                epoch_loss_v_l2 += get_loss_l2(output.detach(), target.detach(), True, 'sum').item()
                val_l2_all.append(get_loss_l2(output.detach(), target.detach(), True, 'none'))
                epoch_loss_v_tv += get_loss_tv(output.detach(), target.detach(), True, 'sum').item()
                val_tv_all.append(get_loss_tv(output.detach(), target.detach(), True, 'none'))
                pred_class = torch.argmax(output.cpu(), 1).to(torch.device(device))
                true_class = torch.argmax(target.cpu(), 1).to(torch.device(device))
                pred_result = true_class == pred_class
                pred_result = list(pred_result.cpu().numpy())
                pred_results.extend(pred_result)

                # if count % 10 == 0:
                #     print(loss)
                count += 1

            pred_results = np.array(pred_results)
            acc_v = np.mean(pred_results)
            val_losses.append(epoch_loss_v/N_v)
            val_accs.append(float(acc_v))
            val_losses_l2.append(epoch_loss_v_l2/N_v)
            val_losses_tv.append(epoch_loss_v_tv/N_v)

            if epoch_loss_v/N_v < min_val_loss:
                min_val_loss = epoch_loss_v/N_v
                val_l2_all = torch.concat(val_l2_all, 0).detach().cpu().numpy()
                val_tv_all = torch.concat(val_tv_all, 0).detach().cpu().numpy()
                np.savetxt(path+f'val_l2_all_{seed}{save_appendix}.csv', val_l2_all)
                np.savetxt(path+f'val_tv_all_{seed}{save_appendix}.csv', val_tv_all)
                np.savetxt(path+f'val_acc_all_{seed}{save_appendix}.csv', pred_results)

            print('Total val loss', epoch_loss_v/N_v)
            print('Validation accuracy', acc_v)

        np.savetxt(path+f'train_losses_{seed}{save_appendix}.csv', np.array(train_losses))
        np.savetxt(path+f'train_losses_l2_{seed}{save_appendix}.csv', np.array(train_losses_l2))
        np.savetxt(path+f'train_losses_tv_{seed}{save_appendix}.csv', np.array(train_losses_tv))
        np.savetxt(path+f'train_accs_{seed}{save_appendix}.csv', np.array(train_accs))
        np.savetxt(path+f'val_losses_{seed}{save_appendix}.csv', np.array(val_losses))
        np.savetxt(path+f'val_accs_{seed}{save_appendix}.csv', np.array(val_accs))
        np.savetxt(path+f'val_losses_l2_{seed}{save_appendix}.csv', np.array(val_losses_l2))
        np.savetxt(path+f'val_losses_tv_{seed}{save_appendix}.csv', np.array(val_losses_tv))

