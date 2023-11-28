import os
import torch
import numpy as np
import torch.utils.data as data

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)


class ShapeNet(data.Dataset):
    def __init__(self, config, split, whole):
        self.data_root = config.data_root
        self.pc_path = config.data_root
        self.split = split
        self.npoints = config.npoints
        
        self.data_list_file = os.path.join(self.data_root, f'{split}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = whole

        print(f'[DATASET] sample out {self.sample_points_num} points')
        print(f'[DATASET] Open file {self.data_list_file}')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print(f'[DATASET] Open file {test_data_list_file}')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print(f'[DATASET] Shapenet55 {len(self.file_list)} instances were loaded')

        self.permutation = np.arange(self.npoints)
        
    # def pc_norm(self, pc):
    #     """ pc: NxC, return NxC """
    #     centroid = np.mean(pc, axis=0)
    #     pc = pc - centroid
    #     m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    #     pc = pc / m
    #     return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # data = self.random_sample(data, self.sample_points_num)
        data = data[:self.sample_points_num]
        data = pc_normalize(data)
        indices = list(range(data.shape[0]))
        np.random.shuffle(indices)
        data = data[indices]
        data = torch.from_numpy(data).float()
        
        # return sample['taxonomy_id'], sample['model_id'], data
        return data, sample['model_id']

    def __len__(self):
        return len(self.file_list)