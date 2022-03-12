#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:13:14 2020

@author: XXXX-1
"""
import numpy as np
import os
import pickle
import wfdb
from tqdm import tqdm
#%%
# basepath = '/mnt/SecondaryHDD/PhysioNetChallenge2020_Training_CPSC/Training_WFDB'
basepath = f'{os.getcwd()}/data/PhysioNetChallenge2020_Training_CPSC/Training_WFDB'

fs = 500
files = os.listdir(basepath)
files = [file.split('.hea')[0] for file in files if '.hea' in file]

def generate_inputs_and_outputs(leads_of_interest,samples_to_take,trial):
    """ Create Input and Output Dicts 
    Args:
        leads_of_interest (list): ECG leads of interest
        samples_to_take (int): number of samples in frame
        trial (str): experiment trial
    Outputs:
    inputs (dict): dict with keys = patient number, values = frames
    outputs (dict): dict with keys = patient number, values = labels
    """
    inputs = dict()
    outputs = dict()
    all_labels = []
    for file in tqdm(files):
        data = wfdb.rdsamp(os.path.join(basepath,file))
        leads_data = data[0].transpose()
        leads_names = data[1]['sig_name']
        labels = data[1]
        label = labels['comments'][2].split('Dx: ')[1] #index 2 for Dx, split to remove 'Dx'
        label = label.split(',') #split to account for multiple labels
        
        inputs[file] = []
        outputs[file] = []
        all_labels.append(label)
        nframes = labels['sig_len']//samples_to_take

        lead_indices = np.where(np.in1d(leads_names,leads_of_interest))[0]
        leads_data = leads_data[lead_indices,:]
        
        if trial in ['contrastive_ml','contrastive_msml']:
            for i in range(nframes):
                lead_frames = leads_data[:,i*samples_to_take:(i+1)*samples_to_take]
                lead_frames = lead_frames.transpose() #2500x12
                inputs[file].append(lead_frames)
                outputs[file].append(label)
        else:
            for name,lead in zip(leads_of_interest,leads_data):
                for i in range(nframes):
                    frame = lead[i*samples_to_take:(i+1)*samples_to_take]
                    inputs[file].append(frame)
                    outputs[file].append(label)
        
        inputs[file] = np.array(inputs[file])
        outputs[file] = np.array(outputs[file])
    
    return inputs,outputs,all_labels

#%%
def obtain_samples_to_take(trial):
    if trial == 'contrastive_ms':
        samples_to_take = 5000
    elif trial == 'contrastive_ml':
        samples_to_take = 2500
    elif trial == 'contrastive_msml':
        samples_to_take = 5000
    elif trial == 'contrastive_ss':
        samples_to_take = 2500
    else: #default setting
        samples_to_take = 2500
    return samples_to_take

def encode_outputs(all_labels,outputs):
    """ Encode Labels 
    Args:
        outputs (dict): labels dictionary with keys = patient number, values = labels
    Output:
        outputs (dict): encoded labels 
    """
    unique_labels = np.unique(all_labels)
    unique_labels = [label for label in unique_labels if len(label)==1]
    
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    enc.fit(unique_labels)
    for file,values in outputs.items():
        label_vector = np.zeros((len(values),len(unique_labels)))
        indices = [[enc.transform([el]).item() for el in value] for value in values] #convert arrhythmia label to number
        label_vector[:,indices] = 1 #assign 1 to label index
        outputs[file] = label_vector
    
    return outputs

#%%
def makepath_and_save_data(inputs,outputs,leads_of_interest,trial=''):
    savepath = os.path.join(basepath,'patient_data',trial,'leads_%s' % str(leads_of_interest))
    try:
        os.chdir(savepath)
    except:
        os.makedirs(savepath)
    
    with open(os.path.join(savepath,'ecg_signal_frames_physionet2020.pkl'),'wb') as f:
        pickle.dump(inputs,f)
    with open(os.path.join(savepath,'ecg_signal_labels_physionet2020.pkl'),'wb') as f:
        pickle.dump(outputs,f)
    print('Saved!')
#%%
leads_of_interest = [['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']] #[['II','V2','aVL','aVR']] #[['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']] #list of lists regardless of number of leads
trial = 'contrastive_ss' # 'contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' | 'contrastive_ss' | '' #default
for leads in leads_of_interest:
    samples_to_take = obtain_samples_to_take(trial)
    inputs,outputs,all_labels = generate_inputs_and_outputs(leads,samples_to_take,trial)
    outputs = encode_outputs(all_labels,outputs)
    makepath_and_save_data(inputs,outputs,leads,trial=trial)

