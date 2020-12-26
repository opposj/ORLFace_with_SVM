# noinspection PyUnresolvedReferences
from dataset import ORLFace
import argparse
import numpy as np
from SVM import svm
import itertools
import copy


TRAIN_NUM = 10


def parse_train():
    parser = argparse.ArgumentParser(description='Parser for training parameters.')
    parser.add_argument('-R', metavar='Radio', type=float, default=0.3,
                        help='The radio of training data with default 0.3.')
    parser.add_argument('-P', metavar='PCA', type=int, default=100,
                        help='The output dimension of PCA with default 100.')
    parser.add_argument('-S', metavar='Sigma', type=float, default=0.5,
                        help='The Gaussian sigma of svm with default 0.5.')
    return parser.parse_args()


def test_model(model, dataset):
    if isinstance(model, dict):
        train_test_pred = np.zeros((40, 400))
        for gp, md in model.items():
            output = []
            for data in np.concatenate([dataset.train_data, dataset.test_data], axis=1).T:
                output.append(md.inference(data).astype(int))
            output = np.concatenate(output)
            class_1_idx = np.where(output == 1)
            class_2_idx = np.where(output == -1)
            train_test_pred[(gp[0] - 1), class_1_idx] += 1
            train_test_pred[(gp[1] - 1), class_2_idx] += 1
        train_test_pc = np.argmax(train_test_pred, axis=0)
        train_pred_label = train_test_pc[:dataset.total_train_num]
        test_pred_label = train_test_pc[dataset.total_train_num:]
        train_pos = np.sum((train_pred_label + 1) == dataset.train_label)
        test_pos = np.sum((test_pred_label + 1) == dataset.test_label)
        return 100 * train_pos / dataset.total_train_num, 100 * test_pos / dataset.total_test_num
    else:
        model.forward(dataset.test_data)
        pos = np.sum((np.argmax(model.y if isinstance(model, perception.MultiPerception)
                                else model.output, axis=0) + 1) == dataset.test_label)
        return (pos / dataset.total_test_num) * 100


def train(dataset, **kwargs):
    class_combination = itertools.combinations(range(1, 41), 2)
    models, pos = {}, 0
    for class_1, class_2 in copy.deepcopy(class_combination):
        model = svm.SVM(kwargs['P'], sigma=kwargs['S'])
        data_gen = dataset.generate_tr_data(tr_method='one_to_one', batch_size=dataset.total_train_num // 20,
                                            class_pos=class_1, class_neg=class_2)
        train_rep, label = next(data_gen)
        model.assign_alpha_and_bias(train_rep, label[0])
        models[(class_1, class_2)] = copy.deepcopy(model)
    train_acc, test_acc = test_model(models, dataset)
    return train_acc, test_acc, 0, 0


if __name__ == '__main__':
    args = vars(parse_train())
    dt = ORLFace('D:\\PyCharm\\My_project\\Dataset\\ORLFace', args['R'], args['P'])
    max_mtr = 0
    max_etr = 0
    max_mte = 0
    max_ete = 0
    for i in range(TRAIN_NUM):
        mtr, mte, etr, ete = train(dt, **args)
        if mtr > max_mtr:
            max_mtr = mtr
            max_etr = etr
        if mte > max_mte:
            max_mte = mte
            max_ete = ete
        print("Train {} finished".format(i + 1))
    print('The max test acc in {} trains is {}% in {} and the max train acc is {}% in {}'
          .format(TRAIN_NUM, max_mte, max_ete, max_mtr, max_etr))
