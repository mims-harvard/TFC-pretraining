import torch
import numpy as np
import os
import pickle

alias_lst = ['sleepEDF', 'epilepsy', 'pFD_A', 'pFD_B', 'HAR', 'AHAR', 'physionet2017', 'emg']
dirname_lst = ['SleepEEG', 'Epilepsy', 'FD-A', 'FD-B', 'HAR', 'Gesture', 'ecg', 'emg']
trial_lst = ['contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss', 'contrastive_ms', 'contrastive_ss']
phase_lst = ['train','val','test']
modality_lst = ['eeg', 'eeg', 'other', 'other', 'other', 'other', 'ecg', 'emg']
fraction = 1
term = 'All Terms'
desired_leads = ['I']

for alias, dirname, trial, modality in zip(alias_lst, dirname_lst, trial_lst, modality_lst):
    train_dict = torch.load(os.path.join('datasets', dirname, 'train.pt'))
    val_dict = torch.load(os.path.join('datasets', dirname, 'val.pt'))
    test_dict = torch.load(os.path.join('datasets', dirname, 'test.pt'))

    input_dict = {}
    output_dict = {}
    pid_dict = {}
    input_dict[modality] = {}
    output_dict[modality] = {}
    pid_dict[modality] = {}
    input_dict[modality][fraction] = {}
    output_dict[modality][fraction] = {}
    pid_dict[modality][fraction] = {}

    for phase in phase_lst:
        input_dict[modality][fraction][phase] = {}
        output_dict[modality][fraction][phase] = {}
        pid_dict[modality][fraction][phase] = {}

    input_dict[modality][fraction]['train'][term] = train_dict['samples'][:,0,:]
    input_dict[modality][fraction]['test'][term] = test_dict['samples'][:,0,:]
    input_dict[modality][fraction]['val'][term] = val_dict['samples'][:,0,:]

    output_dict[modality][fraction]['train'][term] = np.expand_dims(train_dict['labels'], axis=1)
    output_dict[modality][fraction]['test'][term] = np.expand_dims(test_dict['labels'], axis=1)
    output_dict[modality][fraction]['val'][term] = np.expand_dims(val_dict['labels'], axis=1)

    ctr = 0
    pid_dict[modality][fraction]['train'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(train_dict['labels'])), axis=1)
    ctr += len(train_dict['labels'])
    pid_dict[modality][fraction]['test'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(test_dict['labels'])), axis=1)
    ctr += len(test_dict['labels'])
    pid_dict[modality][fraction]['val'][term] = \
        np.expand_dims(np.arange(ctr, ctr+len(val_dict['labels'])), axis=1)

    savepath = os.path.join('code', 'baselines', 'CLOCS', 'data', alias, trial,'leads_%s' % str(desired_leads))
    if os.path.isdir(savepath) == False:
        os.makedirs(savepath)

    """ Save Frames and Labels Dicts """
    with open(os.path.join(savepath,'frames_phases_%s.pkl' % alias),'wb') as f:
        pickle.dump(input_dict,f)

    with open(os.path.join(savepath,'labels_phases_%s.pkl' % alias),'wb') as g:
        pickle.dump(output_dict,g)

    with open(os.path.join(savepath,'pid_phases_%s.pkl' % alias),'wb') as h:
        pickle.dump(pid_dict,h)

    print(f'Final Frames Saved for {alias}!')
