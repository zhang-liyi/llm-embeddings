import torch
import numpy as np
from transformer_model import get_mask




def get_distribution_moments(conj_name, data, hyperparam, factored=False):

    hyperparam = torch.tensor(hyperparam)

    n = data.shape[1]
    running_mean = torch.mean(data.float(), 1)
    running_ss = torch.var(data.float(), 1, correction=0) * n

    if conj_name == 'gaussian-gamma':
        alpha = hyperparam[2] + n/2 # scalar
        beta = hyperparam[3] + running_ss/2 + n * hyperparam[1] * (running_mean - hyperparam[0])**2 / 2 / (hyperparam[1] + n) # (batch-size, 1)
        lambd = hyperparam[1] + n # scalar
        mu = (hyperparam[1] * hyperparam[0] + n * running_mean) / (n + hyperparam[1]) # (batch-size, 1)

        ElnT = torch.special.digamma(alpha) - torch.log(beta)
        ET = alpha / beta
        ETX = alpha / beta * mu
        ETX2 = alpha / beta * mu * mu + 1 / lambd

        return torch.concat([ElnT, ET, ETX, ETX2], dim=1)

    elif conj_name == 'exponential':
        # we are actually doing the posterior, gamma distribution here

        alpha = int(hyperparam[0].numpy()) + n # scalar
        beta = hyperparam[1] + running_mean * n # (batch-size, 1)

        moment1 = (1/beta) ** 1 * torch.prod(torch.tensor(range(alpha, alpha + 1)))
        moment2 = (1/beta) ** 2 * torch.prod(torch.tensor(range(alpha, alpha + 2)))
        moment3 = (1/beta) ** 3 * torch.prod(torch.tensor(range(alpha, alpha + 3)))
        moment4 = (1/beta) ** 4 * torch.prod(torch.tensor(range(alpha, alpha + 4)))

        if factored:
            moment2 = moment2/10
            moment3 = moment3/100
            moment4 = moment4/1000

        return torch.concat([moment1, moment2, moment3, moment4], dim=1)

    elif conj_name == 'bernoulli':
        # we are actually doing the posterior, beta distribution here

        running_mean = torch.unsqueeze(running_mean, 1)

        alpha = hyperparam[0] + running_mean * n # (batch_size,)
        beta = hyperparam[1] + n - running_mean * n # (batch_size,)

        moment1 = alpha / (alpha + beta)
        moment2 = moment1 * ((alpha+1) / (alpha+beta+1))
        moment3 = moment2 * ((alpha+2) / (alpha+beta+2))
        moment4 = moment3 * ((alpha+3) / (alpha+beta+3))

        return torch.concat([moment1, moment2, moment3, moment4], dim=1)
    





def run_hmm(model, mask, val_dataset, test_dataset, val_target, test_target, epochs, lr, probe_type='linear', token_num=-1, token_unseen=0, 
            device='cpu', file_path=None):

    out_dim = 4
    if probe_type == 'linear':
        probe = torch.nn.Sequential(
                torch.nn.Linear(model.d_model, out_dim)
            ).to(torch.device(device))

    else:
        probe = torch.nn.Sequential(
                torch.nn.Linear(model.d_model, model.d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(model.d_model, out_dim)
            ).to(torch.device(device))
        
    if len(mask) == 1:
        mlm = False
        mask = mask[0]
    else:
        mlm = True
        mask_idx = mask[1]
        mask_value = mask[2]
        mask = mask[0]
        unmasked_idx = []
        for i in range(500):
            if i not in mask_idx:
                unmasked_idx.append(i)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    lr = lr  # learning rate
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    batch_size = 64

    ### Get model embeddings first
    embds_train = []
    embds_val = []

    T = val_dataset[0].shape[1]
    mask = get_mask(T-token_unseen, device)

    num_batches = len(val_dataset[0]) // batch_size
    for i in range(0, len(val_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(val_dataset[0]))
        if token_unseen == 0:
            data = val_dataset[0][i:end_idx, :]
        else:
            data = val_dataset[0][i:end_idx, :-token_unseen]
        if mlm:
            data[:, mask_idx] = mask_value
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            elif token_num == -1:
                embd = model.transformer_encoder(src, mask)[:,unmasked_idx[0],:]
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num,:]
        embds_train.extend(embd)

    num_batches = len(test_dataset[0]) // batch_size
    for i in range(0, len(test_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(test_dataset[0]))
        if token_unseen == 0:
            data = test_dataset[0][i:end_idx, :]
        else:
            data = test_dataset[0][i:end_idx, :-token_unseen]
        if mlm:
            data[:, mask_idx] = mask_value
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            elif token_num == -1:
                embd = model.transformer_encoder(src, mask)[:,unmasked_idx[0],:]
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num,:]
        embds_val.extend(embd)

    ### Train probe
    min_val_loss = 1e9
    for epoch in range(epochs):

        probe.train()
        total_loss = 0.
        count = 0

        num_batches = len(val_dataset[0]) // batch_size
        for i in range(0, len(val_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(val_dataset[0]))
            embd = torch.stack(embds_train[i:end_idx], 0)

            target = val_target[i:end_idx]

            output = probe(embd)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            count += 1

        print('Total train loss', total_loss/len(val_dataset[0]))

        probe.eval()
        total_loss = 0.
        total_l2_loss = 0.
        true_targets = []
        pred_results = []

        num_batches = len(test_dataset[0]) // batch_size
        for i in range(0, len(test_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(test_dataset[0]))
            embd = torch.stack(embds_val[i:end_idx], 0)

            target = test_target[i:end_idx]

            output = probe(embd)

            loss = criterion(output, target)

            total_loss += loss.item()
            pred_results.extend(list(torch.squeeze(torch.argmax(output, 1)).detach().cpu().numpy()))
            if len(target.shape) == 2:
                true_targets.extend(list(torch.squeeze(torch.argmax(target, 1)).detach().cpu().numpy()))
                total_l2_loss += torch.sum((torch.nn.functional.softmax(output, 1)-target)**2).detach().cpu().numpy()
            else:
                true_targets.extend(list(torch.squeeze(target).detach().cpu().numpy()))

        pred_results = np.array(pred_results)
        true_targets = np.array(true_targets)
        acc = np.mean(pred_results == true_targets)

        print('Val loss', total_loss/len(test_dataset[0]),
              'L2 loss', total_l2_loss/len(test_dataset[0]),
              'Acc', acc)

        if total_loss/len(test_dataset[0]) < min_val_loss:
            min_val_loss = total_loss/len(test_dataset[0])
            pred_results_final = pred_results
            np.savetxt(file_path, np.array([total_loss/len(test_dataset[0]),
                                           total_l2_loss/len(test_dataset[0]),
                                           acc]))

    return pred_results_final, true_targets


def run_hypo(model, val_dataset, test_dataset, val_target, test_target, epochs, lr, probe_type='linear', target_name='default', token_num=-1, lag=0, device='cpu', file_path=''):

    d_model = model.d_model
    out_dim = val_target.shape[1] if target_name == 'default' else token_num + 1
    if probe_type == 'linear':
        probe = torch.nn.Sequential(
                torch.nn.Linear(d_model, out_dim)
            ).to(torch.device(device))

    else:
        probe = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model*3),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model*3, out_dim)
            ).to(torch.device(device))

    model.eval()
    mask = get_mask(len(val_dataset[0][0]), device)
    for param in model.parameters():
        param.requires_grad = False

    if target_name == 'default':
        criterion = torch.nn.CrossEntropyLoss(reduction='sum')
    elif target_name == 'memorization':
        criterion = torch.nn.MSELoss(reduction='sum')
    lr = lr  # learning rate
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    batch_size = 64

    ### Get model embeddings first
    embds_train = []
    embds_val = []

    num_batches = len(val_dataset[0]) // batch_size
    for i in range(0, len(val_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(val_dataset[0]))
        data = val_dataset[0][i:end_idx, :]
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num-lag,:]
        embds_train.extend(embd)

    num_batches = len(test_dataset[0]) // batch_size
    for i in range(0, len(test_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(test_dataset[0]))
        data = test_dataset[0][i:end_idx, :]
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num-lag,:]
        embds_val.extend(embd)

    print('Embedding shape', embds_val[0].shape)

    ### Train probe
    min_val_loss = 1e9
    for epoch in range(epochs):

        probe.train()
        total_loss = 0.

        num_batches = len(val_dataset[0]) // batch_size
        for i in range(0, len(val_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(val_dataset[0]))
            embd = torch.stack(embds_train[i:end_idx], 0)

            if target_name == 'default':
                target = val_target[i:end_idx, :]
            elif target_name == 'memorization':
                target = torch.squeeze(val_dataset[0][i:end_idx, :(token_num+1), 0])

            output = probe(embd)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Train loss', total_loss/len(val_dataset[0]))

        probe.eval()
        total_loss = 0.
        total_l2_loss = 0.
        true_targets = []
        pred_results = []
        all_outputs = []

        num_batches = len(test_dataset[0]) // batch_size
        for i in range(0, len(test_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(test_dataset[0]))
            embd = torch.stack(embds_val[i:end_idx], 0)

            if target_name == 'default':
                target = test_target[i:end_idx, :]
            elif target_name == 'memorization':
                target = torch.squeeze(test_dataset[0][i:end_idx, :(token_num+1), 0])

            output = probe(embd)

            loss = criterion(output, target)

            total_loss += loss.item()
            pred_results.extend(list(torch.squeeze(torch.argmax(output, 1)).detach().cpu().numpy()))
            all_outputs.extend(list(torch.squeeze(output).detach().cpu().numpy()))
            if len(target.shape) == 2:
                true_targets.extend(list(torch.squeeze(torch.argmax(target, 1)).detach().cpu().numpy()))
                total_l2_loss += torch.sum((torch.nn.functional.softmax(output, 1)-target)**2).detach().cpu().numpy()
            else:
                true_targets.extend(list(torch.squeeze(target).detach().cpu().numpy()))

        pred_results = np.array(pred_results)
        true_targets = np.array(true_targets)
        acc = np.mean(pred_results == true_targets)

        print('Val CE loss', total_loss/len(test_dataset[0]),
              'L2 loss', total_l2_loss/len(test_dataset[0]),
              'Acc', acc, '\n')

        if total_loss/len(test_dataset[0]) < min_val_loss:
            min_val_loss = total_loss/len(test_dataset[0])
            pred_results_final = pred_results
            outputs_final = all_outputs
            np.savetxt(file_path, np.array([total_loss/len(test_dataset[0]),
                                           total_l2_loss/len(test_dataset[0]),
                                           acc]))

    return pred_results_final, outputs_final, true_targets, min_val_loss




def run(target_name, model_name, model, mask, val_dataset, test_dataset, epochs, lr, probe_type='linear', token_num=498, hyperparam=None, factored=False, device='cpu',
        dist_idx=[]):
    # dist_idx is only used for model_name == 'mixed', to keep track of which distribution does the datapoint come from

    d_model = model.d_model
    if target_name == 'moments':
        out_dim = 4
    elif target_name == 'sufficient_stat':
        out_dim = 2 if model_name in ['gaussian-gamma', 'mixed'] else 1
    elif target_name == 'memorization':
        out_dim = token_num + 1

    if probe_type == 'linear':
        probe = torch.nn.Sequential(
                torch.nn.Linear(d_model, out_dim)
            ).to(torch.device(device))

    else:
        probe = torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model),
                torch.nn.ReLU(),
                torch.nn.Linear(d_model, out_dim)
            ).to(torch.device(device))
        
    if len(mask) == 1:
        mlm = False
        mask = mask[0]
    else:
        mlm = True
        mask_idx = mask[1]
        mask_value = mask[2]
        mask = mask[0]
        unmasked_idx = []
        for i in range(500):
            if i not in mask_idx:
                unmasked_idx.append(i)
        print(len(mask_idx))

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    criterion = torch.nn.MSELoss(reduction='sum')
    lr = lr  # learning rate
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    batch_size = 64

    ### Get model embeddings first
    embds_train = []
    embds_val = []

    num_batches = len(val_dataset[0]) // batch_size
    for i in range(0, len(val_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(val_dataset[0]))
        if mlm:
            data = val_dataset[0][i:end_idx, :]
            data[:, mask_idx] = mask_value
        else:
            data = val_dataset[0][i:end_idx, :-1]
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            elif token_num == -1:
                embd = model.transformer_encoder(src, mask)[:,unmasked_idx[0],:]
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num,:]
        embds_train.extend(embd)

    num_batches = len(test_dataset[0]) // batch_size
    for i in range(0, len(test_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(test_dataset[0]))
        if mlm:
            data = test_dataset[0][i:end_idx, :]
            data[:, mask_idx] = mask_value
        else:
            data = test_dataset[0][i:end_idx, :-1]
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            if token_num == 0.5:
                outs = model.transformer_encoder(src, mask)
                embd = torch.mean(outs[:,:,:], dim=1)
            else:
                embd = model.transformer_encoder(src, mask)[:,token_num,:]
        embds_val.extend(embd)
    
    ### Train probe
    min_val_loss = 1e9
    for epoch in range(epochs):

        probe.train()
        total_loss = 0.

        num_batches = len(val_dataset[0]) // batch_size
        for i in range(0, len(val_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(val_dataset[0]))
            embd = torch.stack(embds_train[i:end_idx], 0)

            if model_name in ['gaussian-gamma', 'mixed']:
                if target_name == 'moments':
                    target = get_distribution_moments(model_name, val_dataset[0][i:end_idx, :(token_num+1)], hyperparam)
                elif target_name == 'sufficient_stat':
                    if not mlm:
                        target = torch.concat([torch.mean(val_dataset[0][i:end_idx, :(token_num+1)], 1),
                                        torch.std(val_dataset[0][i:end_idx, :(token_num+1)], 1)], 1)
                    else:
                        target = torch.concat([torch.mean(val_dataset[0][i:end_idx, unmasked_idx], 1),
                                        torch.std(val_dataset[0][i:end_idx, unmasked_idx], 1)], 1)
                elif target_name == 'memorization':
                    target = torch.squeeze(val_dataset[0][i:end_idx, :(token_num+1)])
            else:
                if target_name == 'moments':
                    target = get_distribution_moments(model_name, val_dataset[0][i:end_idx, :(token_num+1)], hyperparam, factored)
                elif target_name == 'sufficient_stat':
                    if not mlm:
                        target = torch.mean(val_dataset[0][i:end_idx, :(token_num+1)].float(), 1)
                    else:
                        target = torch.mean(val_dataset[0][i:end_idx, unmasked_idx].float(), 1)
                    if len(target.shape) == 1:
                        target = torch.unsqueeze(target, 1)
                elif target_name == 'memorization':
                    target = torch.squeeze(val_dataset[0][i:end_idx, :(token_num+1)])

            output = probe(embd)

            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print('Train loss', total_loss/len(val_dataset[0])/out_dim)

        probe.eval()
        total_loss = 0.
        true_targets = []
        pred_results = []

        num_batches = len(test_dataset[0]) // batch_size
        for i in range(0, len(test_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(test_dataset[0]))
            embd = torch.stack(embds_val[i:end_idx], 0)

            if model_name in ['gaussian-gamma', 'mixed']:
                if target_name == 'moments':
                    target = get_distribution_moments(model_name, test_dataset[0][i:end_idx, :(token_num+1)], hyperparam)
                elif target_name == 'sufficient_stat':
                    if not mlm:
                        target = torch.concat([torch.mean(test_dataset[0][i:end_idx, :(token_num+1)], 1),
                                        torch.std(test_dataset[0][i:end_idx, :(token_num+1)], 1)], 1)
                    else:
                        target = torch.concat([torch.mean(test_dataset[0][i:end_idx, unmasked_idx], 1),
                                        torch.std(test_dataset[0][i:end_idx, unmasked_idx], 1)], 1)
                elif target_name == 'memorization':
                    target = torch.squeeze(test_dataset[0][i:end_idx, :(token_num+1)])
            else:
                if target_name == 'moments':
                    target = get_distribution_moments(model_name, test_dataset[0][i:end_idx, :(token_num+1)], hyperparam, factored)
                elif target_name == 'sufficient_stat':
                    if not mlm:
                        target = torch.mean(test_dataset[0][i:end_idx, :(token_num+1)].float(), 1)
                    else:
                        target = torch.mean(test_dataset[0][i:end_idx, unmasked_idx].float(), 1)
                    if len(target.shape) == 1:
                        target = torch.unsqueeze(target, 1)
                elif target_name == 'memorization':
                    target = torch.squeeze(test_dataset[0][i:end_idx, :(token_num+1)])

            output = probe(embd)

            loss = criterion(output, target)

            total_loss += loss.item()
            pred_results.extend(list(torch.squeeze(output).detach().cpu().numpy()))
            true_targets.extend(list(torch.squeeze(target).detach().cpu().numpy()))

        pred_results = np.array(pred_results)
        true_targets = np.array(true_targets)
        acc = np.mean(pred_results)

        print('Val loss', total_loss/len(test_dataset[0])/out_dim)

        if total_loss/len(test_dataset[0])/out_dim < min_val_loss:
            min_val_loss = total_loss/len(test_dataset[0])/out_dim
            pred_results_final = pred_results

    return pred_results_final, true_targets, min_val_loss