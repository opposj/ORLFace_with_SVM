import numpy as np
# from scipy.optimize import minimize, Bounds


class SVM(object):
    def __init__(self, num_input, sigma):
        self.num_input = num_input
        self.sigma = sigma
        self.num_sample = 0
        self.core = lambda x1, x2: np.exp(-np.linalg.norm(x1 - x2) / (2 * self.sigma ** 2))
        self.alpha = None
        self.bias = 0
        self.x_train = None
        self.label_train = None

    def assign_alpha_and_bias(self, x, label):
        self.num_sample = x.shape[1]

        '''
        def op_func(a):
            return np.sum(a) - 0.5 * sum([sum([a[p] * a[q] * label[p] * label[q] * self.core(x[:, p], x[:, q])
                                               for q in range(self.num_sample)]) for p in
                                          range(self.num_sample)])

        bounds = Bounds([0] * self.num_sample, [np.inf] * self.num_sample)
        cons = {'type': 'eq', 'fun': lambda a: np.array([sum([label[p] * a[p] for p in range(self.num_sample)])])}
        alpha = 1 * np.ones(self.num_sample)
        op_res = minimize(op_func, alpha, method='SLSQP', constraints=cons, bounds=bounds)
        self.alpha = op_res.x
        '''

        def get_core_matrix(sample, lb):
            cm = np.zeros([self.num_sample, self.num_sample])
            for row in range(self.num_sample):
                for col in range(self.num_sample):
                    coefficient = 1. if row == col else 0.5
                    cm[row, col] = coefficient * lb[row] * lb[col] * self.core(sample[:, row], sample[:, col])
            return cm

        core_matrix = get_core_matrix(x, label)
        self.alpha = np.linalg.inv(core_matrix) @ np.expand_dims(np.ones(self.num_sample), 1)
        for idx, temp in enumerate(self.alpha):
            if temp > 0:
                self.bias = 1 / label[idx] - sum([self.alpha[p] * label[p] * self.core(x[:, p], x[:, idx])
                                                  for p in range(self.num_sample)])
                break

        self.x_train = x
        self.label_train = label

    def inference(self, x):
        return np.sign(sum([self.label_train[p] * self.alpha[p] * self.core(self.x_train[:, p], x)
                            for p in range(self.num_sample)]) + self.bias)
