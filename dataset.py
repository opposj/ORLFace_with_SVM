import glob
import os
import numpy as np
from PIL import Image
from PCA import pca


class ORLFace(object):

    def __init__(self, root_path, tr_radio, pca_num):
        self.root_path = root_path
        self.tr_radio = tr_radio
        self.pca_num = pca_num
        self.class_names = os.listdir(root_path)[2:]
        self.total_train_num = int(tr_radio * 400)
        self.total_test_num = 400 - self.total_train_num
        train_num_pc = int(tr_radio * 10)
        test_num_pc = 10 - train_num_pc
        train_data, test_data, train_label, test_label = [], [], [], []
        for cn in self.class_names:
            all_data = np.stack([np.array(Image.open(p).resize((23, 28))).flatten() for p in
                                 glob.glob(os.path.join(self.root_path, cn, '*.bmp'))], axis=1)
            random_idx = np.random.permutation(np.arange(10))
            train_data.append(all_data[:, random_idx[:train_num_pc]])
            test_data.append(all_data[:, random_idx[train_num_pc:]])
            train_label.append(np.array([int(cn[1:])] * train_num_pc))
            test_label.append(np.array([int(cn[1:])] * test_num_pc))
        self.train_data, self.test_data = np.split(pca.cci_pca(pca.to_zero_mean(
            (np.concatenate(train_data + test_data, axis=1)) / 255), pca_num)[0], [int(400 * tr_radio)], axis=1)
        self.train_label = np.concatenate(train_label)
        self.test_label = np.concatenate(test_label)

    def generate_tr_data(self, tr_method='one_hot', batch_size=2, class_pos=0, class_neg=0):
        if tr_method == 'one_hot':
            assert self.total_train_num % batch_size == 0, \
                "The batch_size should be divisible by the length of dataset, which is {}".format(self.total_train_num)
            train_sample_idx = np.random.permutation(np.array(range(self.total_train_num)))
            for i in range(self.total_train_num // batch_size):
                img_read = self.train_data[:, train_sample_idx[i * batch_size:(i + 1) * batch_size]]
                label = []
                for lb in self.train_label[train_sample_idx[i * batch_size:(i + 1) * batch_size]]:
                    row_label = 40 * [0]
                    row_label[lb - 1] = 1
                    label.append(np.array(row_label))
                yield img_read, np.stack(label, axis=1)
        elif tr_method == 'one_to_one':
            sub_train_idx = np.concatenate([np.where(self.train_label == class_pos)[0],
                                            np.where(self.train_label == class_neg)[0]])
            sub_train_data = self.train_data[:, sub_train_idx]
            sub_train_label = self.train_label[sub_train_idx]
            sub_train_num = self.total_train_num // 20
            assert sub_train_num % batch_size == 0, \
                "The batch_size should be divisible by the length of dataset, which is {}".format(sub_train_num)
            sub_train_sample_idx = np.random.permutation(np.array(range(sub_train_num)))
            for i in range(sub_train_num // batch_size):
                img_read = sub_train_data[:, sub_train_sample_idx[i * batch_size:(i + 1) * batch_size]]
                label = []
                for lb in sub_train_label[sub_train_sample_idx[i * batch_size:(i + 1) * batch_size]]:
                    if lb == class_pos:
                        label.append(np.array([1]))
                    else:
                        label.append(np.array([-1]))
                yield img_read, np.stack(label, axis=1)
        elif tr_method == 'one_to_other':
            assert self.total_train_num % batch_size == 0, \
                "The batch_size should be divisible by the length of dataset, which is {}".format(self.total_train_num)
            train_sample_idx = np.random.permutation(np.array(range(self.total_train_num)))
            for i in range(self.total_train_num // batch_size):
                img_read = self.train_data[:, train_sample_idx[i * batch_size:(i + 1) * batch_size]]
                label = []
                for lb in self.train_label[train_sample_idx[i * batch_size:(i + 1) * batch_size]]:
                    if lb == class_pos:
                        label.append(np.array([1]))
                    else:
                        label.append(np.array([-1]))
                yield img_read, np.stack(label, axis=1)


if __name__ == '__main__':
    dataset = ORLFace('D:\\PyCharm\\My_project\\Dataset\\ORLFace', 0.3, 3)
    data_gen = dataset.generate_tr_data(tr_method='one_to_other', class_pos=2, class_neg=3)
    for data in data_gen:
        print("OK")
