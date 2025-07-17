import numpy as np
import scipy

def get_posterior_hypothesis(data, spaces):

    # Assuming uniform prior over spaces

    unnorm_log_posteriors = np.zeros((len(data), len(spaces)))
    posteriors = np.zeros((len(data), len(spaces)))
    for i in range(len(data)):

        seq = data[i]
        T = len(seq)

        for j, space in enumerate(spaces):
            validity = (seq[:,0] >= space[0,0]).astype(np.float32) + (seq[:,0] <= space[1,0]).astype(np.float32) + \
                (seq[:,1] >= space[0,1]).astype(np.float32) + (seq[:,1] <= space[1,1]).astype(np.float32)
            validity = np.all(validity==4)

            if not validity:
                unnorm_log_posteriors[i, j] = -np.inf
            else:
                if space[0,0] == space[1,0] and space[0,1] == space[1,1]:
                    unnorm_log_posteriors[i, j] = 0.
                elif space[0,0] == space[1,0]:
                    unnorm_log_posteriors[i, j] = np.log(1/(space[1,1] - space[0,1])) * T
                elif space[0,1] == space[1,1]:
                    unnorm_log_posteriors[i, j] = np.log(1/(space[1,0] - space[0,0])) * T
                else:
                    area = (space[1,0] - space[0,0]) * (space[1,1] - space[0,1])
                    unnorm_log_posteriors[i, j] = np.log(1/area) * T

        posteriors[i,:] = scipy.special.softmax(unnorm_log_posteriors[i,:])

        if i % 200 == 0:
            print(i)

    return posteriors