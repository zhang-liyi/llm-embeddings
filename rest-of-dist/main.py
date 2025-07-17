import torch
import torch.distributions as D
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os

from data_sampler import *
from transformer_model import *
from inference.hmm_inference import *
from inference.hypothesis_inference import *
from parse_args import *
from probe import *
from plot import *



args = parse_args()
config = vars(args)
seed = args.seed
model_name = args.model_name
target_name = args.target_name
hypo_mode = args.hypothesis_mode
device = args.device
token_num = args.token_num
N = args.N # num training data for transformer
T = args.T # sequence length
V = args.V # vocab size
train_val_idx = args.train_val_idx
lr = args.learning_rate
delta = args.delta
print(config)

np.random.seed(seed)
torch.manual_seed(seed)



if model_name == 'gaussian-gamma':
    hyperparam = [1., 1., 5., 1.]
    continuous_input = True
    V = 1
elif model_name == 'bernoulli':
    hyperparam = [2., 8.]
    continuous_input = False
    V = 2
elif model_name == 'exponential':
    hyperparam = [2., 4.]
    continuous_input = True
    V = 1
elif model_name == 'hmm':
    continuous_input = False
elif model_name == 'hypothesis':
    continuous_input = True

# Get data
if model_name != 'hypothesis':

    if model_name != 'hmm':

        datasampler = DataSampler(model_name, hyperparam, T)

        if args.generate_data:
            dataset = datasampler.get_data(N+train_val_idx).to(torch.device(device))
            print(dataset.shape, datasampler.prior_param.shape)
            with open(f'results/{datasampler.model_name}_dataset_{seed}.pickle', 'wb') as handle:
                pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
            train_dataset = [dataset[:-train_val_idx], datasampler.prior_param[:-train_val_idx]]
            val_dataset = [dataset[-train_val_idx:-1000], datasampler.prior_param[-train_val_idx:-1000]]
            test_dataset = [dataset[-1000:], datasampler.prior_param[-1000:]]
            dataset = [train_dataset, val_dataset, test_dataset, datasampler.hyperparam]
        else:
            dataset = pd.read_pickle(f'results/{datasampler.model_name}_dataset_3007.pickle')
            train_dataset = [dataset[:-train_val_idx], '']
            val_dataset = [dataset[-train_val_idx:-1000], '']
            test_dataset = [dataset[-1000:], '']
            dataset = [train_dataset, val_dataset, test_dataset, '']

    else:

        datasampler = DataSampler(model_name, [4, 0.5, V, delta], T)
        dataset = datasampler.get_data(N+train_val_idx).to(torch.device(device))
        print(dataset.shape)
        train_dataset = [dataset[:-train_val_idx], datasampler.class_assignments[:-train_val_idx]]
        val_dataset = [dataset[-train_val_idx:-1000], datasampler.class_assignments[-train_val_idx:-1000]]
        test_dataset = [dataset[-1000:], datasampler.class_assignments[-1000:]]
        dataset = [train_dataset, val_dataset, test_dataset, datasampler.hyperparam, datasampler.trans_mat, datasampler.emit_mat]

        with open(f'results/{datasampler.model_name}_dataset_{delta}_{seed}.pickle', 'wb') as handle:
            pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

        train_dataset, val_dataset, test_dataset, hyperparam, trans_mat, emit_mat = pd.read_pickle(f'results/{datasampler.model_name}_dataset_{delta}_{seed}.pickle')
        train_dataset[0] = train_dataset[0].to(torch.device(device))
        val_dataset[0] = val_dataset[0].to(torch.device(device))
        test_dataset[0] = test_dataset[0].to(torch.device(device))
        train_dataset[1] = train_dataset[1].to(torch.device(device))
        val_dataset[1] = val_dataset[1].to(torch.device(device))
        test_dataset[1] = test_dataset[1].to(torch.device(device))

        trans_mat = trans_mat.numpy()
        emit_mat = emit_mat.numpy()
        
else:

    datasampler = DataSamplerHypothesis(T, 8, hypo_mode)
    dataset = torch.tensor(datasampler.get_data(N+train_val_idx)).float().to(torch.device(device))
    print(dataset.shape)
    train_dataset = [dataset[:-train_val_idx], None]
    val_dataset = [dataset[-train_val_idx:-1000], None]
    test_dataset = [dataset[-1000:], None]
    dataset = [train_dataset, val_dataset, test_dataset]



# Transformer training

d_model = args.d_model
att_heads = args.att_heads
t_num_layers = args.t_num_layers

if args.mlm:
    mask, mask_idx = get_mask(T, device, mlm=args.mlm)
    if continuous_input:
        mask_value = torch.mean(train_dataset[0])
        model = TransformerModel(V, d_model, att_heads, d_model, t_num_layers, 0.1, use_pos=True, continuous_input=continuous_input).to(torch.device(device))
    else:
        V += 1
        mask_value = V - 1
        model = TransformerModel(V, d_model, att_heads, d_model, t_num_layers, 0.1, use_pos=True, continuous_input=continuous_input).to(torch.device(device))
    np.savetxt(f'results/maskidx_{datasampler.model_name}_{datasampler.mode}_{T}_{token_num}_{seed}_.csv', np.array(mask_idx))
else:
    mask, mask_idx = get_mask(T-1, device, mlm=args.mlm)
    model = TransformerModel(V, d_model, att_heads, d_model, t_num_layers, 0.1, use_pos=True, continuous_input=continuous_input).to(torch.device(device))

if model_name in ['bernoulli', 'hmm']:
    criterion = torch.nn.CrossEntropyLoss()
elif model_name in ['gaussian','gaussian-gamma', 'exponential', 'hypothesis']:
    criterion = torch.nn.MSELoss()
tf_lr = 0.001  if model_name != 'exponential' else 0.0003
optimizer = torch.optim.Adam(model.parameters(), lr=tf_lr)
batch_size = 64
min_val_loss = 1e6

epochs_no_increase = 0

if not args.random_init:
    for epoch in range(300):

        model.train()
        total_loss = 0.
        count = 0

        num_batches = len(train_dataset[0]) // batch_size
        for i in range(0, len(train_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(train_dataset[0]))

            if not args.mlm:
                data = train_dataset[0][i:end_idx, :-1]
                target = train_dataset[0][i:end_idx, 1:]
            else:
                data = train_dataset[0][i:end_idx, :]
                target = train_dataset[0][i:end_idx, :]
                data[:, mask_idx] = mask_value

            output = model(data, torch.zeros(data.shape[0], 1, d_model).to(torch.device(device)), mask)

            if args.mlm:
                output = output[:, mask_idx]
                target = target[:, mask_idx]

            if model_name in ['bernoulli', 'hmm']:
                output_flat = torch.reshape(output, (-1,V))
                target_flat = torch.reshape(target, (-1,))
            elif model_name in ['gaussian','gaussian-gamma','exponential', 'hypothesis']:
                output_flat = torch.reshape(output, (-1,V))
                target_flat = torch.reshape(target, (-1,V))
            loss = criterion(output_flat, target_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.shape[0]

            count += 1

        print('Total train loss', total_loss / len(train_dataset[0]))

        model.eval()
        total_loss = 0.
        count = 0

        num_batches = len(val_dataset[0]) // batch_size
        for i in range(0, len(val_dataset[0]), batch_size):
            end_idx = min(i+batch_size, len(val_dataset[0]))

            if not args.mlm:
                data = train_dataset[0][i:end_idx, :-1]
                target = train_dataset[0][i:end_idx, 1:]
            else:
                data = train_dataset[0][i:end_idx, :]
                target = train_dataset[0][i:end_idx, :]
                data[:, mask_idx] = mask_value

            output = model(data, torch.zeros(data.shape[0], 1, 128).to(torch.device(device)), mask)

            if args.mlm:
                output = output[:, mask_idx]
                target = target[:, mask_idx]

            if model_name in ['bernoulli', 'hmm']:
                output_flat = torch.reshape(output,(-1,V))
                target_flat = torch.reshape(target, (-1,))
            elif model_name in ['gaussian','gaussian-gamma','exponential', 'hypothesis']:
                output_flat = torch.reshape(output,(-1,V))
                target_flat = torch.reshape(target, (-1,V))
            loss = criterion(output_flat, target_flat)

            total_loss += loss.item() * data.shape[0]

            count += 1

        print('Total val loss', total_loss / len(val_dataset[0]))

        if total_loss < min_val_loss:
            epochs_no_increase = 0
            min_val_loss = total_loss
            torch.save(model.state_dict(), 
                    f'results/transformer_model_weights_{args.mlm}_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{token_num}_{seed}.pth')

        else:
            epochs_no_increase += 1

        if epochs_no_increase > 20:
            break

    model.load_state_dict(torch.load(
        f'results/transformer_model_weights_{args.mlm}_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{token_num}_{seed}.pth'))



# Inference and probing

if model_name == 'hmm':
    for target_name in ['viterbi', 'posterior', 'posterior_np1']:
        for token_unseen in [0,1]:
            for lr in [0.003]:
                file_path = f'results/{datasampler.model_name}_{datasampler.mode}_{target_name}_{token_unseen}_{delta}_{lr}_{T}_{seed}.csv'
                if target_name == 'posterior': # p(z_n+1 | x_n)
                    posteriors_val = get_posterior_hmm(val_dataset, trans_mat, emit_mat, stop=-1)
                    posteriors_test = get_posterior_hmm(test_dataset, trans_mat, emit_mat, stop=-1)
                    target_val = get_posterior_next_class(posteriors_val, trans_mat, emit_mat)
                    target_test = get_posterior_next_class(posteriors_test, trans_mat, emit_mat)
                    target_val = torch.tensor(target_val, dtype=torch.float32).to(torch.device(device))
                    target_test = torch.tensor(target_test, dtype=torch.float32).to(torch.device(device))
                elif target_name == 'posterior_np1': # p(z_n+1 | x_n+1)
                    posteriors_val = get_posterior_hmm(val_dataset, trans_mat, emit_mat, stop=0)
                    posteriors_test = get_posterior_hmm(test_dataset, trans_mat, emit_mat, stop=0)
                    target_val = posteriors_val[:,-1,:]
                    target_test = posteriors_test[:,-1,:]
                    target_val = torch.tensor(target_val, dtype=torch.float32).to(torch.device(device))
                    target_test = torch.tensor(target_test, dtype=torch.float32).to(torch.device(device))
                elif target_name == 'viterbi':
                    target_val = run_viterbi(val_dataset, trans_mat, emit_mat)[:,-1]
                    target_test = run_viterbi(test_dataset, trans_mat, emit_mat)[:,-1]
                    target_val = torch.tensor(target_val, dtype=torch.int64).to(torch.device(device))
                    target_test = torch.tensor(target_test, dtype=torch.int64).to(torch.device(device))

                if not args.mlm:
                    pred_results, true_targets = run_hmm(model, mask, val_dataset, test_dataset,
                                                        target_val, target_test, 1000, lr, 'linear', 
                                                        token_num=token_num, token_unseen=token_unseen, 
                                                        device=device, file_path=file_path)
                else:
                    pred_results, true_targets = run_hmm(model, [mask, mask_idx, mask_value], val_dataset, test_dataset,
                                                        target_val, target_test, 1000, lr, 'linear', 
                                                        token_num=token_num, token_unseen=token_unseen, 
                                                        device=device, file_path=file_path)
                
                np.savetxt(f'results/pred_results_hmm_{target_name}_{token_unseen}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{token_num}_{seed}.csv', 
                           pred_results)
                np.savetxt(f'results/true_targets_hmm_{target_name}_{token_unseen}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{token_num}_{seed}.csv', 
                           true_targets)

elif model_name == 'hypothesis':
    posteriors_val = get_posterior_hypothesis(val_dataset[0].cpu().numpy(), datasampler.spaces)
    posteriors_test = get_posterior_hypothesis(test_dataset[0].cpu().numpy(), datasampler.spaces)
    val_target = torch.tensor(posteriors_val)
    test_target = torch.tensor(posteriors_test)
    vec = torch.max(val_target, 1)[0]
    print('Spread of the true posterior', torch.mean(vec), torch.std(vec))

    for lr in [0.003,0.01,0.001]:
        file_path = f'results/{datasampler.model_name}_{datasampler.mode}_{lr}_{T}_{seed}.csv'
        pred_results_final, outputs_final, true_targets, min_val_loss = run_hypo(model, val_dataset, test_dataset,
                                                val_target.to(torch.device(device)), test_target.to(torch.device(device)),
                                                1500, lr, probe_type='linear', token_num=token_num, lag=0, target_name='default', device=device, file_path=file_path)
        
    if args.add_plot:
        plot_hypo_2d(model, datasampler, val_dataset, val_target.cpu().numpy(), device=args.device)

else:
    # OOD Evaluation
    if model_name == 'gaussian-gamma':
        ood_hp = [5.,1.,2.,1.]
        datasampler = DataSampler(model_name, ood_hp, T)
    elif model_name == 'bernoulli':
        ood_hp = [8., 2.]
        datasampler = DataSampler(model_name, ood_hp, T)
    elif model_name == 'exponential':
        ood_hp = [2., 1.]
        datasampler = DataSampler(model_name, ood_hp, T)

    dataset_ood = datasampler.get_data(train_val_idx).to(torch.device(device))
    print(dataset_ood.shape, datasampler.prior_param.shape)
    val_dataset_ood = [dataset_ood[-train_val_idx:-1000], datasampler.prior_param[-train_val_idx:-1000]]
    test_dataset_ood = [dataset_ood[-1000:], datasampler.prior_param[-1000:]]

    # Probe
    probe_type = 'linear' if args.use_linear_classifier else 'nonlinear'
    if not args.mlm:
        pred_results, true_targets, min_val_loss = run(args.target_name, model_name, model, [mask], val_dataset, test_dataset, 
                                        2000, 0.001, probe_type, token_num, hyperparam, False, device)
        
        pred_results_ood, true_targets_ood, min_val_loss_ood = run(args.target_name, model_name, model, [mask], val_dataset_ood, test_dataset_ood, 
                                                2000, 0.001, probe_type, token_num, ood_hp, False, device)
    else:
        pred_results, true_targets, min_val_loss = run(args.target_name, model_name, model, [mask, mask_idx, mask_value], val_dataset, test_dataset, 
                                        2000, 0.001, probe_type, token_num, hyperparam, False, device)
        
        pred_results_ood, true_targets_ood, min_val_loss_ood = run(args.target_name, model_name, model, [mask, mask_idx, mask_value], val_dataset_ood, test_dataset_ood, 
                                                2000, 0.001, probe_type, token_num, ood_hp, False, device)
    
    np.savetxt(f'results/loss_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               np.array([min_val_loss]))
    np.savetxt(f'results/loss_ood_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               np.array([min_val_loss_ood]))
    np.savetxt(f'results/pred_results_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               pred_results)
    np.savetxt(f'results/pred_results_ood_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               pred_results_ood)
    np.savetxt(f'results/true_targets_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               true_targets)
    np.savetxt(f'results/true_targets_ood_{datasampler.model_name}_{datasampler.mode}_{d_model}_{att_heads}_{t_num_layers}_{delta}_{T}_{args.mlm}_{args.random_init}_{token_num}_{seed}.csv', 
               true_targets_ood)
    
    if args.add_plot:
        if target_name == 'moments':
            plot_moments(model_name, pred_results, true_targets, pred_results_ood, true_targets_ood)
        elif target_name == 'sufficient_stat':
            plot_sufficient_stat(model_name, pred_results, true_targets, pred_results_ood, true_targets_ood)
        elif target_name == 'memorization':
            plot_memorization(model_name, pred_results, true_targets)
