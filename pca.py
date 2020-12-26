import numpy as np
import sys
eps = sys.float_info.epsilon


def to_zero_mean(x):
    return x - np.expand_dims(np.mean(x, axis=1), axis=1)


def cci_pca(x, dim_out):
    assert x.shape[0] >= dim_out, "The dimension of input should be no less than the dimension of output!"
    num_sample = x.shape[1]
    x_mean = 0
    base_vector = np.zeros((x.shape[0], dim_out))

    for t in range(num_sample):
        x_mean = t * x_mean / (t + 1) + x[:, t] / (t + 1)
        x_iter = x[:, t] - x_mean
        for i in range(min(t + 1, dim_out)):
            if i == t and t == 0:
                base_vector[:, i] = x[:, i]
            elif i == t:
                base_vector[:, i] = x_iter
            else:
                base_vector[:, i] = \
                    t * base_vector[:, i] / (t + 1) + ((np.expand_dims(x_iter, axis=1)
                                                        @ np.expand_dims(x_iter, axis=0))
                                                       @ np.expand_dims(base_vector[:, i] / (np.linalg.norm
                                                                                             (base_vector[:, i]) + eps),
                                                                        axis=1))[:, 0] / (t + 1)
                base_norm = base_vector[:, i] / (np.linalg.norm(base_vector[:, i]) + eps)
                x_iter = x_iter - np.dot(x_iter, base_norm) * base_norm

    for i in range(dim_out):
        base_vector[:, i] = base_vector[:, i] / (np.linalg.norm(base_vector[:, i]) + eps)

    return base_vector.T @ x, base_vector
