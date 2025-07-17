import torch
import torch.distributions as D
import numpy as np
import math
import matplotlib.pyplot as plt

class DataSampler:

    def __init__(self, model_name, hyperparam, T):

        '''
        model_name: {bernoulli, gaussian, exponential, hmm} (lower-case string)
        T: sequence length (int)

        if model_name is bernoulli, hyperparam should be [alpha, beta]
        if model_name is exponential, hyperparam should be [concentration, rate]
        if model_name is gaussian, hyperparam should be a list with: prior mean, prior precision, likelihood (known) precision
        if model_name is gaussian-gamma, hyperparam should be a list with: prior mean, prior lambda, prior alpha, prior beta
        if model_name is hmm, hyperparam should be a list with: num_classes, dirichlet gamma (generates transition mat), V (vocab size), dirichlet delta (generates emission probs)
        '''

        self.model_name = model_name.lower()
        self.hyperparam = hyperparam
        self.T = T
        self.mode = 'default'

        if self.model_name == 'bernoulli':
            self.prior = D.beta.Beta(torch.tensor(hyperparam[0]), torch.tensor(hyperparam[1]))
        elif self.model_name == 'exponential':
            self.prior = D.gamma.Gamma(torch.tensor(hyperparam[0]), torch.tensor(hyperparam[1]))
        elif self.model_name == 'gaussian':
            self.prior = D.normal.Normal( torch.tensor(hyperparam[0]), (1/torch.tensor(hyperparam[1]))**0.5 )
        elif self.model_name == 'gaussian-gamma':
            self.prior = D.gamma.Gamma(torch.tensor(hyperparam[2]), torch.tensor(hyperparam[3]))
        elif self.model_name == 'hmm':
            self.num_classes = torch.tensor(hyperparam[0], dtype=torch.int64)
            self.prior_trans = D.dirichlet.Dirichlet(torch.zeros((self.num_classes)) + hyperparam[1])
            self.V = torch.tensor(hyperparam[2], dtype=torch.int64)
            self.prior_emit = D.dirichlet.Dirichlet(torch.zeros((self.V)) + hyperparam[3])

    def plot_emit_mat(self):

        emit_mat = self.emit_mat

        fig = plt.figure(figsize=(12, 12))

        plt.subplot(2, 2, 1)
        plt.stem(emit_mat[0].numpy())
        plt.subplot(2, 2, 2)
        plt.stem(emit_mat[1].numpy())
        plt.subplot(2, 2, 3)
        plt.stem(emit_mat[2].numpy())
        plt.subplot(2, 2, 4)
        plt.stem(emit_mat[3].numpy())
        plt.show()

        return

    def get_data(self, n):

        if self.model_name != 'hmm':
            self.prior_param = self.prior.sample([n])

            if self.model_name == 'bernoulli':
                self.likelihood = D.bernoulli.Bernoulli(probs=self.prior_param)
                samples = self.likelihood.sample([self.T]).T
                return samples.type(torch.LongTensor)

            if self.model_name == 'exponential':
                self.likelihood = D.exponential.Exponential(self.prior_param)
                samples = self.likelihood.sample([self.T]).T
                return torch.unsqueeze(samples, dim=2)

            elif self.model_name == 'gaussian':
                self.likelihood = D.normal.Normal( self.prior_param, (1/torch.tensor(self.hyperparam[2]))**0.5 )
                samples = self.likelihood.sample([self.T]).T
                return torch.unsqueeze(samples, dim=2)

            elif self.model_name == 'gaussian-gamma':
                self.prior_mu = D.normal.Normal(torch.tensor(self.hyperparam[0]), (1/(torch.tensor(self.hyperparam[1])*self.prior_param))**0.5 )
                self.mu_samp = torch.squeeze(self.prior_mu.sample([1]))
                self.likelihood = D.normal.Normal(self.mu_samp, (1/self.prior_param)**0.5)
                samples = self.likelihood.sample([self.T]).T
                self.prior_param = torch.concat([torch.unsqueeze(self.mu_samp,1),
                                                torch.unsqueeze(self.prior_param,1)],
                                                dim=1)
                return torch.unsqueeze(samples, dim=2)

        else:
            self.samples = torch.zeros((n, self.T), dtype=torch.int64)
            self.class_assignments = torch.zeros((n, self.T), dtype=torch.int64)
            self.trans_mat = self.prior_trans.sample([self.num_classes])
            self.emit_mat = self.prior_emit.sample([self.num_classes])
            print('Transition matrix')
            print(self.trans_mat)
            # print('Emission matrix')
            # self.plot_emit_mat()
            for i in range(n):
                for j in range(self.T):
                    if j == 0:
                        c = np.random.randint(0, self.num_classes)
                        self.class_assignments[i,j] = c
                        self.samples[i,j] = D.categorical.Categorical(self.emit_mat[c]).sample([1])[0]
                    else:
                        c = D.categorical.Categorical(self.trans_mat[c]).sample([1])[0]
                        self.class_assignments[i,j] = c
                        self.samples[i,j] = D.categorical.Categorical(self.emit_mat[c]).sample([1])[0]
                if i % 100 == 0:
                    print(f'Generating doc {i}')

            return self.samples



class DataSamplerHypothesis:

    def __init__(self, T, size, mode='easy'):

        self.T = T
        self.size = size
        self.model_name = 'hypothesis'
        self.mode = mode

    def get_data(self, n):

        spaces = []
        if self.mode == 'easy':
            axis = list(range(self.size))
        else:
            axis = [0]
            for i in range(self.size-1):
                if i % 2 == 0:
                    axis.append(i+0.4)
                else:
                    axis.append(i+1)
        for i in range(self.size):
            for j in range(self.size):
                upper_left = np.array([axis[i], axis[j]])
                for k in range(i+1, self.size):
                    for l in range(j+1, self.size):
                        lower_right = np.array([axis[k], axis[l]])
                        space = np.array([[axis[i], axis[j]], [axis[k], axis[l]]])
                        spaces.append(space)

        self.spaces = spaces

        indices = np.random.randint(0, len(spaces), n)

        self.indices = indices

        sequences = np.zeros((n, self.T, 2))

        for i in range(n):
            idx = indices[i]
            space = spaces[idx]

            if space[0,0] == space[1,0]:
                x = np.zeros((self.T,)) + space[0,0]
            else:
                x = np.random.uniform(space[0,0], space[1,0], self.T)

            if space[0,1] == space[1,1]:
                y = np.zeros((self.T,)) + space[0,1]
            else:
                y = np.random.uniform(space[0,1], space[1,1], self.T)

            seq = np.stack([x, y], 0).T

            sequences[i,:,:] = seq

        return sequences




