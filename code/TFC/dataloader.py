import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft

def generate_freq(dataset, config):
    X_train = dataset["samples"]
    y_train = dataset['labels']
    # shuffle
    data = list(zip(X_train, y_train))
    np.random.shuffle(data)
    data = data[:10000] # take a subset for testing.
    X_train, y_train = zip(*data)
    X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

    if len(X_train.shape) < 3:
        X_train = X_train.unsqueeze(2)

    if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
        X_train = X_train.permute(0, 2, 1)

    """Align the TS length between source and target datasets"""
    X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

    if isinstance(X_train, np.ndarray):
        x_data = torch.from_numpy(X_train)
    else:
        x_data = X_train

    """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
    the output shape is half of the time window."""

    x_data_f = fft.fft(x_data).abs() #/(window_length) # rfft for real value inputs.
    return (X_train, y_train, x_data_f)

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :1, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10 #30 #7 # 60*1
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]
            print('Using subset for debugging, the datasize is:', y_train.shape[0])

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        self.len = X_train.shape[0]

        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, subset=True):
    train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))
    finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))  # train.pt
    test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))  # test.pt
    """In pre-training: 
    train_dataset: [371055, 1, 178] from SleepEEG.    
    finetune_dataset: [60, 1, 178], test_dataset: [11420, 1, 178] from Epilepsy"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset) # for self-supervised, the data are augmented here
    finetune_dataset = Load_Dataset(finetune_dataset, configs, training_mode, target_dataset_size=configs.target_batch_size, subset=subset)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode,
                                target_dataset_size=configs.target_batch_size, subset=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    finetune_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, finetune_loader, test_loader


class Load_Dataset_motion(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64):
        super(Load_Dataset_motion, self).__init__()
        self.training_mode = training_mode
        ts_data = dataset['ts_data']
        fft_data = dataset['fft_data']
        # self.seq_len = dataset.shape[1] # (count, seq_len, fea_num)
        X_train = torch.from_numpy(ts_data)
        X_train_fft = torch.from_numpy(fft_data)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)
        if len(X_train_fft.shape) < 3:
            X_train_fft = X_train_fft.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)
        if X_train_fft.shape.index(min(X_train_fft.shape)) != 1:  # make sure the Channels in second dim
            X_train_fft = X_train_fft.permute(0, 2, 1)

        self.len = X_train.shape[0]

        """Augmentation"""
        self.x_data = X_train
        self.x_data_f = X_train_fft
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]

    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.aug1[index], self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len


def data_generator_motion(taindata_path_set, valdata_path_set, configs, training_mode, sample_count=-1, sample_count_val=-1):
    train_data_dir = taindata_path_set['train_data_dir']
    train_motion_names_dir = taindata_path_set['train_motion_names_dir']
    train_fft_dir = taindata_path_set['train_fft_dir']
    val_data_dir = valdata_path_set['val_data_dir']
    val_motion_names_dir = valdata_path_set['val_motion_names_dir']
    val_fft_dir = valdata_path_set['val_fft_dir']

    print('loading train_ts')
    train_ts = np.load(train_data_dir)
    if sample_count > 0: 
        train_ts = train_ts[:sample_count, :, :]
    train_motion_names = pd.read_parquet(train_motion_names_dir)
    if sample_count > 0: 
        train_motion_names = train_motion_names.iloc[:sample_count]
    print('loading train_fft')
    train_fft = np.load(train_fft_dir)
    if sample_count > 0: 
        train_fft = train_fft[:sample_count, :, :]

    print('loading val_ts')
    val_ts = np.load(val_data_dir)
    if sample_count_val > 0: 
        val_ts = val_ts[:sample_count_val, :, :]
    val_motion_names = pd.read_parquet(val_motion_names_dir)
    if sample_count_val > 0:
        val_motion_names = val_motion_names.iloc[:sample_count_val]
    print('loading val_fft')
    val_fft = np.load(val_fft_dir)
    if sample_count_val > 0:
        val_fft = val_fft[:sample_count_val, :, :]
    
    train_ts = np.nan_to_num(train_ts, neginf=0)
    train_fft = np.nan_to_num(train_fft, neginf=0)
    val_ts = np.nan_to_num(val_ts, neginf=0)
    val_fft = np.nan_to_num(val_fft, neginf=0)

    train_dataset = {'ts_data': train_ts, 'fft_data': train_fft}
    val_dataset = {'ts_data': val_ts, 'fft_data': val_fft}

    train_dataset = Load_Dataset_motion(train_dataset, configs, training_mode, target_dataset_size=configs.batch_size) # for self-supervised, the data are augmented here
    val_dataset = Load_Dataset_motion(val_dataset, configs, training_mode, target_dataset_size=configs.batch_size)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size, 
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=configs.batch_size,
                                             shuffle=True, drop_last=configs.drop_last, num_workers=0)

    return train_loader, val_loader, train_motion_names, val_motion_names
