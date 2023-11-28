import os
import numpy as np
import warnings
import pickle

import torch
from torch.utils.data import Dataset

import random
from collections import OrderedDict, defaultdict
from dataset.utils import Datum
from typing import List

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class ModelNet(Dataset):
    def __init__(self, config, split):
        self.root = config.data_root
        self.use_normals = config.use_normals
        self.num_category = config.classes
        self.split = split

        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_{}.txt'.format(split)))]
        print('The size of %s data is %d' % (split, len(self.shape_ids)))

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, 8192))
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.shape_ids)

    def _get_item(self, index):
        point_set, label = self.list_of_points[index], self.list_of_labels[index]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
            point_set = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
        else:
            point_set[:, :3] = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
            point_set[:, 3:] = point_set[:, [5, 3, 4]] * np.array([[-1, -1, 1]])
        return point_set, label[0]

    def __getitem__(self, index):
        points, label = self._get_item(index)
        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)
        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()
        return current_points, label


class ModelNetFewShot(Dataset):
    def __init__(self, config, split):
        super().__init__()
        self.root = config.data_root
        self.use_normals = config.use_normals
        self.num_category = config.classes
        self.num_shots = config.num_shots
        self.split = split

        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]

        self.classnames = OrderedDict()
        for i, line in enumerate(self.cat):
            classname = line.strip()
            self.classnames[i] = classname

        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_{}.txt'.format(split)))]

        self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, 8192))
        with open(self.save_path, 'rb') as f:
            self.list_of_points, self.list_of_labels = pickle.load(f)

        self.data = self.read_data(self.classnames, self.list_of_points, self.list_of_labels)

        if split == 'train':
            self.data = self.generate_fewshot_dataset(self.data, num_shots=self.num_shots)

        print(f'The size of {split} data is {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def _get_item(self, index):
        item = self.data[index]
        point_set, label = item.impath, item.label

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]
            point_set = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
        else:
            point_set[:, :3] = point_set[:, [2, 0, 1]] * np.array([[-1, -1, 1]])
            point_set[:, 3:] = point_set[:, [5, 3, 4]] * np.array([[-1, -1, 1]])
        return point_set, label

    def __getitem__(self, index):
        points, label = self._get_item(index)

        pt_idxs = np.arange(0, points.shape[0])
        if self.split == 'train':
            np.random.shuffle(pt_idxs)

        current_points = points[pt_idxs].copy()
        current_points = torch.from_numpy(current_points).float()

        return current_points, label

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


if __name__ == "__main__":
    
    classes = {
        'airplane': 0,
        'bathtub': 1,
        'bed': 2, 
        'bench': 3, 
        'bookshelf': 4,
        'bottle': 5, 
        'bowl': 6, 
        'car': 7, 
        'chair': 8, 
        'cone': 9, 
        'cup': 10,
        'curtain': 11,
        'desk': 12,
        'door': 13,
        'dresser': 14,
        'flower_pot': 15,
        'glass_box': 16,
        'guitar': 17,
        'keyboard': 18,
        'lamp': 19,
        'laptop': 20,
        'mantel': 21,
        'monitor': 22,
        'night_stand': 23,
        'person': 24,
        'piano': 25,
        'plant': 26,
        'radio': 27,
        'range_hood': 28,
        'sink': 29,
        'sofa': 30,
        'stairs': 31,
        'stool': 32,
        'table': 33,
        'tent': 34,
        'toilet': 35,
        'tv_stand': 36,
        'vase': 37,
        'wardrobe': 38,
        'xbox': 39
    }
