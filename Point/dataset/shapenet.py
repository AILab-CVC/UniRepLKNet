import os
import sys
import glob
import h5py
import numpy as np
import random
import torch
import json
import cv2
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict, defaultdict
from dataset.utils import Datum

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def load_data_partseg(data_root, partition):
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(data_root, 'shapenet_part_seg_hdf5_data', '*train*.h5')) \
               + glob.glob(os.path.join(data_root, 'shapenet_part_seg_hdf5_data', '*val*.h5'))
    else:
        file = glob.glob(os.path.join(data_root, 'shapenet_part_seg_hdf5_data', '*%s*.h5'%partition))
    for h5_name in file:
        with h5py.File(h5_name,'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
        # f = h5py.File(h5_name, 'r+')
        # data = f['data'][:].astype('float32')
        # label = f['label'][:].astype('int64')
        # seg = f['pid'][:].astype('int64')
        # f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


class ShapeNetClsFewShot(Dataset):
    def __init__(self, data_root, num_points, partition='trainval', class_choice=None, num_shots=16):
        self.points, self.labels, self.seg = load_data_partseg(data_root, partition)
        self.cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 
                       'laptop': 9, 'motor': 10, 'mug': 11, 'pistol': 12,
                       'rocket': 13, 'skateboard': 14, 'table': 15}
        self.class2num = {
            self.cat2id[t]: t for t in self.cat2id
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice
        self.num_shots = num_shots
        
        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.labels == id_choice).squeeze()
            self.points = self.points[indices]
            self.labels = self.labels[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0
        
        self.data = self.read_data(self.class2num, self.points, self.labels)
        if self.partition == 'trainval':
            self.data = self.generate_fewshot_dataset(self.data, num_shots=self.num_shots)
        
        print(f'The size of {partition} data is {len(self.data)}')
      
    def __getitem__(self, index):
        item = self.data[index]
        pointcloud, label = item.impath, item.label
        pointcloud = pointcloud[:self.num_points]
        pointcloud[:, 0:3] = pc_normalize(pointcloud[:, 0:3])

        if self.partition == 'trainval':
            # pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]

        return pointcloud, label

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