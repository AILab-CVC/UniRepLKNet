import os
import h5py
import numpy as np
import random
from collections import OrderedDict, defaultdict
import torch
from torch.utils.data import Dataset
from dataset.utils import Datum


class ScanObjectNN(Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.root = config.data_root
        self.subset = subset
        self.cat2id = {
            "bag": 0, "bed": 1, "bin": 2, "box": 3, "cabinet": 4, "chair": 5,
            "desk": 6, "display": 7, "door": 8, "pillow": 9, "shelf": 10, 
            "sink": 11, "sofa": 12, "table": 13, "toilet": 14
        }
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        self.classnames = np.unique(self.labels)

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = self.points[idx, pt_idxs].copy()
        current_points = current_points[:, [2, 0, 1]] * np.array([[-1, -1, 1]])

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]

        return current_points, label

    def __len__(self):
        return self.points.shape[0]


class ScanObjectNNFewShot(Dataset):
    def __init__(self, config, subset):
        super().__init__()
        self.root = config.data_root
        self.subset = subset
        self.num_shots = config.num_shots

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'train_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        self.classnames = np.unique(self.labels)
        self.data = self.read_data(self.classnames, self.points, self.labels)
        if self.subset == 'train':
            self.data = self.generate_fewshot_dataset(self.data, num_shots=self.num_shots)

        print(f'The size of {subset} data is {len(self.data)}')

    def __getitem__(self, index):
        item = self.data[index]
        points, label = item.impath, item.label

        pt_idxs = np.arange(0, points.shape[0])
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)

        current_points = points[pt_idxs].copy()
        current_points = current_points[:, [2, 0, 1]] * np.array([[-1, -1, 1]])

        current_points = torch.from_numpy(current_points).float()

        return current_points, label

    def __len__(self):
        return len(self.data)

    def read_data(self, classnames, datas, labels):
        items = []

        for i, data in enumerate(datas):
            label = int(labels[i])
            classname = classnames[label]

            item = Datum(
                impath=data,
                label=label,
                classname=classname
            )
            items.append(item)

        return items

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.

        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)
        # NOTE 根据不同的label来划分数据集
        for item in data_source:
            output[item.label].append(item)

        return output

    def generate_fewshot_dataset(
            self, *data_sources, num_shots=-1, repeat=True
    ):
        """Generate a few-shot dataset (typically for the training set).

        This function is useful when one wants to evaluate a model
        in a few-shot learning setting where each class only contains
        a few number of images.

        Args:
            data_sources: each individual is a list containing Datum objects.
            num_shots (int): number of instances per class to sample.
            repeat (bool): repeat images if needed.
        """
        if num_shots < 1:
            if len(data_sources) == 1:
                return data_sources[0]
            return data_sources

        print(f'Creating a {num_shots}-shot dataset')

        output = []

        for data_source in data_sources:
            tracker = self.split_dataset_by_label(data_source)  # NOTE ModelNet正好40个类
            dataset = []

            for label, items in tracker.items():  # NOTE 从每个类中抽取num_shots个样本，构成小样本数据集
                if len(items) >= num_shots:
                    sampled_items = random.sample(items, num_shots)
                else:
                    if repeat:
                        sampled_items = random.choices(items, k=num_shots)
                    else:
                        sampled_items = items
                dataset.extend(sampled_items)

            output.append(dataset)

        if len(output) == 1:
            return output[0]

        return output
