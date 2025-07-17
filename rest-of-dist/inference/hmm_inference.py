import numpy as np



def get_posterior_hmm(dataset, trans_mat, emit_mat, stop=None):

    trans_mat = trans_mat
    emit_mat = emit_mat
    pi = np.zeros(len(trans_mat)) + 1/len(trans_mat)
    posterior_all = []

    for j in range(len(dataset[0])):
        if stop is None or stop == 0:
            y = dataset[0][j].cpu().numpy()
        else:
            y = dataset[0][j][:stop].cpu().numpy()
        C = trans_mat.shape[0]
        V = emit_mat.shape[1]
        N = len(y)
        gamma = np.zeros((N, C))
        a = np.zeros((N, C))
        b = np.zeros((N, C))

        for i in range(N):
            if i == 0:
                a[i,:] = pi * emit_mat[:, y[i]]
            else:
                a_prev = np.expand_dims(a[i-1,:], 1)
                a[i,:] = emit_mat[:, y[i]] * np.sum( trans_mat * a_prev, 0)
            a[i,:] = a[i,:] / np.sum(a[i,:])

        for i in range(N-1, -1, -1):
            if i == N-1:
                b[i,:] = emit_mat[:, y[i]]
            else:
                b[i,:] = np.sum(trans_mat * (b[i+1, :] * emit_mat[:, y[i+1]]), 1)
            b[i,:] = b[i,:] / np.sum(b[i,:])

        unnorm_posterior = a * b
        posterior = unnorm_posterior / np.sum(unnorm_posterior, 1, keepdims=True)
        posterior_all.append(posterior)

        if j % 200 == 0:
            print(j)

    posterior_all = np.stack(posterior_all, 0)

    return posterior_all



def get_posterior_next_class(posteriors, trans_mat, emit_mat):

    trans_mat = trans_mat
    emit_mat = emit_mat
    N = posteriors.shape[0]
    C = posteriors.shape[2]

    posteriors = posteriors[:,-1,:]
    posteriors = np.expand_dims(posteriors, 2)
    prodct = np.multiply(posteriors, trans_mat)

    return np.sum(prodct, 1)



def run_viterbi(dataset, A, B):
    pi = np.zeros(len(A)) + 1/len(A)
    seqs = []

    for data_id in range(len(dataset[0])):
        y = dataset[0][data_id].cpu().numpy()

        num_classes = B.shape[0] # number of classes
        x_seq = np.zeros([num_classes, 0])
        V = np.log(B[:, y[0]] * pi)

        # forward to compute the optimal value function V
        for y_ in y[1:]:
            _V = np.log(np.tile(B[:, y_], reps=[num_classes, 1]).T) + np.log( A.T ) + np.tile(V, reps=[num_classes, 1])
            x_ind = np.argmax(_V, axis=1)
            x_seq = np.hstack([x_seq, np.c_[x_ind]])
            V = _V[np.arange(num_classes), x_ind]

        x_T = np.argmax(V)

        # backward to fetch optimal sequence
        x_seq_opt, i = np.zeros(x_seq.shape[1]+1), x_seq.shape[1]
        prev_ind = x_T
        while i >= 0:
            x_seq_opt[i] = prev_ind
            i -= 1
            prev_ind = x_seq[int(prev_ind), i]
        seqs.append(x_seq_opt)

        if data_id % 100 == 0:
            print(data_id)

    return np.stack(seqs, 0)

