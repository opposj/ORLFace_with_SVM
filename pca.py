import numpy as np
import heapq
import sys
eps = sys.float_info.epsilon


def to_zero_mean(x):
    return x - np.expand_dims(np.mean(x, axis=1), axis=1)


def trivial_pca(x, dim_out):
    assert x.shape[0] >= dim_out, "The dimension of input should be no less than the dimension of output!"
    c_x = (x @ x.T) / x.shape[1]
    eig, f_vector = np.linalg.eig(c_x)
    base_vector = f_vector[:, heapq.nlargest(dim_out, range(len(eig)), eig.take)]
    return base_vector.T @ x, base_vector


def neural_pca(x, dim_out, eta, batch, epoch):
    assert x.shape[0] >= dim_out, "The dimension of input should be no less than the dimension of output!"
    num_sample = x.shape[1]
    assert num_sample % batch == 0, "The number of samples should be divisible by batch size!"
    num_batch = num_sample // batch
    base_vector = np.random.randn(x.shape[0], dim_out)
    for i in range(dim_out):
        base_vector[:, i] = base_vector[:, i] / np.linalg.norm(base_vector[:, i])

    for i in range(dim_out):
        for _ in range(epoch):
            for b in range(num_batch):
                batch_x = x[:, b * batch:(b + 1) * batch]
                batch_y = base_vector.T @ batch_x
                estimate_value = base_vector[:, 0:i + 1] @ batch_y[0:i + 1, :]
                delta_base = eta * np.sum(np.expand_dims(batch_y[i, :], axis=0) * (batch_x - estimate_value), axis=1)
                delta_result = base_vector[:, i] + delta_base
                base_vector[:, i] = delta_result / np.linalg.norm(delta_result)

    return base_vector.T @ x, base_vector


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


def reconstruct_error(x, y, base_vector):
    err = x - base_vector @ y
    return np.linalg.norm(err) / np.sqrt(x.size)


if __name__ == '__main__':
    in_put = to_zero_mean(np.random.randn(4, 30))
    # in_put = to_zero_mean(np.array([[1, 4, 2, 6], [2, 1, 4.2, 4], [9, 1, 2.3, 1], [1.2, 3.2, -1.2, 8]]))
    output, base = trivial_pca(in_put, 3)
    output_2, base_2 = neural_pca(in_put, 3, 0.01, 1, 500)
    output_3, base_3 = cci_pca(in_put, 3)
    error = reconstruct_error(in_put, output, base)
    error_2 = reconstruct_error(in_put, output_2, base_2)
    error_3 = reconstruct_error(in_put, output_3, base_3)
    print("OK")
