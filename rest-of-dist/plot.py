import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from transformer_model import get_mask

from sklearn.decomposition import PCA


def plot_hypo_2d(model, datasampler, val_dataset, val_target, device='cpu'):
    model.eval()
    mask = get_mask(len(val_dataset[0][0]), device)
    batch_size = 64

    embds = []
    num_batches = len(val_dataset[0]) // batch_size
    for i in range(0, len(val_dataset[0]), batch_size):
        end_idx = min(i+batch_size, len(val_dataset[0]))
        data = val_dataset[0][i:end_idx, :]
        with torch.no_grad():
            src = model.embedding(data) * np.sqrt(model.d_model)
            if model.use_pos:
                src = model.pos_encoder(src)
            embd = model.transformer_encoder(src, mask)[:,-1,:]
        embds.extend(embd)


    pca = PCA(n_components=2)
    x = pca.fit_transform(torch.stack(embds, 0).cpu().numpy())


    fig = plt.figure(figsize=(7,7))
    colors = []
    for i in range(len(x)):
        max_class = np.argmax(val_target[i,:])
        max_space = datasampler.spaces[max_class]
        colors.append(np.mean(max_space))
    plt.scatter(x[:,0], x[:,1], s=7, c=colors, alpha=0.6)
    plt.axis('off')
    plt.savefig("pca_mean.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(7,7))
    colors = []
    for i in range(len(x)):
        max_class = np.argmax(val_target[i,:])
        max_space = datasampler.spaces[max_class]
        colors.append(np.mean(max_space[:, 0]))
    plt.scatter(x[:,0], x[:,1], s=7, c=colors, alpha=0.6)
    plt.axis('off')
    plt.savefig("pca_mean_y.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(7.73,7))
    colors = []
    for i in range(len(x)):
        max_class = np.argmax(val_target[i,:])
        max_space = datasampler.spaces[max_class]
        colors.append(max_space[1,0] - max_space[0,0])
    plt.scatter(x[:,0], x[:,1], s=7, c=colors, alpha=0.6)
    plt.colorbar(fraction=0.037)
    plt.axis('off')
    plt.savefig("pca_spread_y.pdf", format="pdf", bbox_inches="tight")
    plt.show()
    plt.close()




def plot_sufficient_stat(model_name, pred_results, true_targets, pred_results_ood, true_targets_ood):
    matplotlib.rcParams['axes.linewidth'] = 2

    if model_name in ['gaussian-gamma', 'mixed']:
        fig = plt.figure(figsize=(8, 8))
        plt.subplot(2, 2, 1)
        indices = np.argsort(np.array(true_targets)[:,0])
        plt.plot(np.array(pred_results)[:,0][indices], color='blue')
        plt.plot(np.array(true_targets)[:,0][indices], color='red',alpha=0.75,linewidth=2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 2, 2)
        indices = np.argsort(np.array(true_targets)[:,1])
        plt.plot(np.array(pred_results)[:,1][indices], color='blue')
        plt.plot(np.array(true_targets)[:,1][indices], color='red',alpha=0.75,linewidth=2)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 2, 3)
        indices = np.argsort(np.array(true_targets_ood)[:,0])
        plt.plot(np.array(pred_results_ood)[:,0][indices], color='blue')
        plt.plot(np.array(true_targets_ood)[:,0][indices], color='red',alpha=0.75,linewidth=2)
        plt.yticks(range(-1,10,3))
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 2, 4)
        indices = np.argsort(np.array(true_targets_ood)[:,1])
        plt.plot(np.array(pred_results_ood)[:,1][indices], color='blue')
        plt.plot(np.array(true_targets_ood)[:,1][indices], color='red',alpha=0.75,linewidth=2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.savefig(f"{model_name}-ss.pdf", format="pdf", bbox_inches="tight")

    elif model_name == 'bernoulli' or model_name == 'exponential':
        fig = plt.figure(figsize=(4,8))
        plt.subplot(2, 1, 1)
        indices = np.argsort(np.array(true_targets)[:])
        plt.plot(np.array(pred_results)[:][indices], color='blue')
        plt.plot(np.array(true_targets)[:][indices], color='red',alpha=0.75,linewidth=2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 1, 2)
        indices = np.argsort(np.array(true_targets_ood)[:])
        plt.plot(np.array(pred_results_ood)[:][indices], color='blue')
        plt.plot(np.array(true_targets_ood)[:][indices], color='red',alpha=0.75,linewidth=2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.savefig(f"{model_name}-ss.pdf", format="pdf", bbox_inches="tight")




def plot_memorization(model_name, pred_results, true_targets):

    if model_name == 'gaussian-gamma':
        fig = plt.figure(figsize=(30, 3))
        for i in range(10):
            plt.subplot(1, 10, i+1)
            indices = np.argsort(np.array(true_targets)[:,i])
            plt.plot(np.array(pred_results)[:,i][indices], color='blue')
            plt.plot(np.array(true_targets)[:,i][indices], color='red',alpha=0.75,linewidth=2)
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
        plt.savefig(f"{model_name}-token-memorization.pdf", format="pdf", bbox_inches="tight")




def plot_moments(model_name, pred_results, true_targets, pred_results_ood, true_targets_ood):
    # plot moments
    matplotlib.rcParams['axes.linewidth'] = 2

    if model_name in ['gaussian-gamma', 'mixed']:
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        indices = np.argsort(np.array(true_targets)[:,0])
        plt.plot(np.array(pred_results)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\log \tau]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 2)
        indices = np.argsort(np.array(true_targets)[:,1])
        plt.plot(np.array(pred_results)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 3)
        indices = np.argsort(np.array(true_targets)[:,2])
        plt.plot(np.array(pred_results)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau \mu]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 4)
        indices = np.argsort(np.array(true_targets)[:,3])
        plt.plot(np.array(pred_results)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau \mu^2]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 5)
        indices = np.argsort(np.array(true_targets_ood)[:,0])
        plt.plot(np.array(pred_results_ood)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\log \tau]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 6)
        indices = np.argsort(np.array(true_targets_ood)[:,1])
        plt.plot(np.array(pred_results_ood)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 7)
        indices = np.argsort(np.array(true_targets_ood)[:,2])
        plt.plot(np.array(pred_results_ood)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau \mu]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 8)
        indices = np.argsort(np.array(true_targets_ood)[:,3])
        plt.plot(np.array(pred_results_ood)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\tau \mu^2]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.savefig("gaussian-gamma-moments-linear.pdf", format="pdf", bbox_inches="tight")

    elif model_name == 'exponential':
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        indices = np.argsort(np.array(true_targets)[:,0])
        plt.plot(np.array(pred_results)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 2)
        indices = np.argsort(np.array(true_targets)[:,1])
        plt.plot(np.array(pred_results)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^2]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 3)
        indices = np.argsort(np.array(true_targets)[:,2])
        plt.plot(np.array(pred_results)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^3]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 4)
        indices = np.argsort(np.array(true_targets)[:,3])
        plt.plot(np.array(pred_results)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^4]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 5)
        indices = np.argsort(np.array(true_targets_ood)[:,0])
        plt.plot(np.array(pred_results_ood)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 6)
        indices = np.argsort(np.array(true_targets_ood)[:,1])
        plt.plot(np.array(pred_results_ood)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^2]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 7)
        indices = np.argsort(np.array(true_targets_ood)[:,2])
        plt.plot(np.array(pred_results_ood)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^3]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 8)
        indices = np.argsort(np.array(true_targets_ood)[:,3])
        plt.plot(np.array(pred_results_ood)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[\lambda^4]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.savefig(f"{model_name}-moments-linear.pdf", format="pdf", bbox_inches="tight")

    elif model_name == 'bernoulli':
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        indices = np.argsort(np.array(true_targets)[:,0])
        plt.plot(np.array(pred_results)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 2)
        indices = np.argsort(np.array(true_targets)[:,1])
        plt.plot(np.array(pred_results)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^2]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 3)
        indices = np.argsort(np.array(true_targets)[:,2])
        plt.plot(np.array(pred_results)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^3]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 4)
        indices = np.argsort(np.array(true_targets)[:,3])
        plt.plot(np.array(pred_results)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^4]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 5)
        indices = np.argsort(np.array(true_targets_ood)[:,0])
        plt.plot(np.array(pred_results_ood)[:,0][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,0][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 6)
        indices = np.argsort(np.array(true_targets_ood)[:,1])
        plt.plot(np.array(pred_results_ood)[:,1][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,1][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^2]$', x=0.5, y=0.9)
        #plt.yticks(range(-1,3,1))
        #plt.ylim(-0.5,2)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 7)
        indices = np.argsort(np.array(true_targets_ood)[:,2])
        plt.plot(np.array(pred_results_ood)[:,2][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,2][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^3]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.subplot(2, 4, 8)
        indices = np.argsort(np.array(true_targets_ood)[:,3])
        plt.plot(np.array(pred_results_ood)[:,3][indices], color='blue',alpha=1)
        plt.plot(np.array(true_targets_ood)[:,3][indices], color='red',linewidth=2,alpha=0.75)
        plt.title(r'$E[p^4]$', x=0.5, y=0.9)
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)

        plt.savefig(f"{model_name}-moments-linear.pdf", format="pdf", bbox_inches="tight")

