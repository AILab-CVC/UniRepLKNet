import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
from torchvision import transforms

warnings.filterwarnings('ignore')


class Global_Temp(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "temp_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)  # (17519, 34040, 3)
        self.raw_time = np.load(os.path.join(self.root_path,
                                             "data_time_" + self.flag + ".npy"), allow_pickle=True)  # (17519)
        raw_data = self.raw_data
        raw_time = self.raw_time
        print(self.raw_data.shape)
        print("==== " + self.flag + " data sorted load finished ====")
        if self.features == 'S':
            raw_data = raw_data[:, :, :1]
        if self.features == 'S_station':
            raw_data = raw_data[:, self.target:(self.target + 1), :1]
        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)  # (17519, 34040*3)
        data = raw_data.astype(np.float)

        df_stamp = raw_time
        df_stamp = pd.to_datetime(df_stamp)
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


class Global_Wind(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        self.flag = flag
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.raw_data = np.load(os.path.join(self.root_path,
                                             "wind_global_hourly_" + self.flag + ".npy"),
                                allow_pickle=True)  # (17519, 34040, 3)
        self.raw_time = np.load(os.path.join(self.root_path,
                                             "data_time_" + self.flag + ".npy"), allow_pickle=True)  # (17519)
        raw_data = self.raw_data
        raw_time = self.raw_time
        print(self.raw_data.shape)
        print("==== " + self.flag + " data sorted load finished ====")

        data_len, station, feat = raw_data.shape
        raw_data = raw_data.reshape(data_len, station * feat)  # (17519, 34040*3)
        data = raw_data.astype(np.float)

        df_stamp = raw_time
        df_stamp = pd.to_datetime(df_stamp)
        data_stamp = time_features(pd.to_datetime(df_stamp), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data
        self.data_y = data
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
