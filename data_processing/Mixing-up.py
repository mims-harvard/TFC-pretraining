import torch
import numpy as np
import os

dataset_lst = ['SleepEEG', 'Epilepsy', 'FD-A', 'FD-B', 'HAR', 'Gesture', 'ecg', 'emg']

for dataset_name in dataset_lst:
    train_dict = torch.load(os.path.join('datasets', dataset_name, 'train.pt'))
    val_dict = torch.load(os.path.join('datasets', dataset_name, 'val.pt'))
    test_dict = torch.load(os.path.join('datasets', dataset_name, 'test.pt'))

    train_input = train_dict['samples']
    test_input = test_dict['samples']
    train_output = np.expand_dims(train_dict['labels'], axis=1)
    test_output = np.expand_dims(test_dict['labels'], axis=1)

    savepath = os.path.join(basepath, 'code', 'baselines', 'Mixing-up', 'data', dataset_name)
    if os.path.isdir(savepath) == False:
        os.makedirs(savepath)
    np.save(os.path.join(savepath, 'train_input.npy'), train_input)
    np.save(os.path.join(savepath, 'test_input.npy'), test_input)
    np.save(os.path.join(savepath, 'train_output.npy'), train_output)
    np.save(os.path.join(savepath, 'test_output.npy'), test_output)