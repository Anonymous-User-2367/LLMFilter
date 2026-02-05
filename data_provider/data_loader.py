import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings

warnings.filterwarnings('ignore')
   
    
class Dataset_SingleFiltering(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='selkov.csv',
                 scale=True, num_steps=200):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.num_steps = num_steps

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0] 
        df_stamp = pd.read_csv(os.path.join(self.root_path, f'{data_name}_obs.csv'))
        print('++++++++++++++Filtering!')

        cols_data_stamp = df_stamp.columns[1:]
        df_data_stamp = df_stamp[cols_data_stamp]
        data_stamp = df_data_stamp.values
        
      
        self.data_stamp_x = data_stamp[border1:border2,:]
        self.data_stamp_y = data_stamp[border1:border2,:]
        self.data_x = data[border1:border2,:]
        self.data_y = data[border1:border2,:]
        
    def __getitem__(self, index):
        insample = np.zeros((self.pred_len, self.data_stamp_x.shape[1]))
        insample_mask = np.zeros((self.pred_len, self.data_stamp_x.shape[1]))
        outsample = np.zeros((self.pred_len, self.data_x.shape[1]))
        outsample_mask = np.zeros((self.pred_len, self.data_x.shape[1])) 

        max_index = self.data_stamp_x.shape[0] - self.pred_len
        if index > max_index:
            return insample, outsample, insample_mask, outsample_mask
        
        insample = self.data_stamp_x[index:index + self.pred_len]
        insample_mask = self.data_stamp_x[index:index + self.pred_len]
        outsample = self.data_x[index:index + self.pred_len]
        outsample_mask = self.data_x[index:index + self.pred_len]

        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return self.data_x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries and the next window for prediction.
                Shapes: insample "timeseries, insample size", 
                        onsample "timeseries, onsample size"
        """

        insample = self.data_stamp_x.reshape(1, -1, self.data_stamp_x.shape[1])
        insample_mask = self.data_stamp_y.reshape(1, -1, self.data_stamp_y.shape[1])
        onsample = self.data_x.reshape(1, -1, self.data_x.shape[1])
        onsample_mask = self.data_y.reshape(1, -1, self.data_y.shape[1])
        return insample, insample_mask, onsample, onsample_mask

