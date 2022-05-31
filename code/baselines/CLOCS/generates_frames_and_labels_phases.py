#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 14:17:07 2019

@author: XXXX-1

Arrange Desired Dataset into Train/Val/Test Dicts for Training

Inputs:
    ECG and PPG Frames and Labels

Outputs:
    Dicts of ECG and PPG Split According to Training Phase
"""

import os
import pickle
import random
from operator import itemgetter
import numpy as np
from tqdm import tqdm

from sklearn.decomposition import PCA
#%%

basepath = 'data'

dataset = 'physionet2020'
trials = ['contrastive_ms'] #contrastive_msml' # contrastive_ms' | 'contrastive_ml' | 'contrastive_msml' 'contrastive_ss' | '' #default
print('Dataset: %s' % dataset)
peak_detection = False

def load_path(dataset,leads=['ii']):
    if dataset == 'bidmc':
        path = os.path.join(basepath,'BIDMC v1')
        label = ''
    elif dataset == 'physionet':
        path = os.path.join(basepath,'PhysioNet v2')
        label = ''
    elif dataset == 'mimic':
        shrink_factor = str(0.1)
        path = os.path.join(basepath,'MIMIC3_WFDB','frame-level',shrink_factor)
        label = 'mortality_'
    elif dataset == 'cipa':
        lead = ['II','aVR']
        path = os.path.join(basepath,'cipa-ecg-validation-study-1.0.0','leads_%s' % lead)
        label = 'drug_'
    elif dataset == 'cardiology':
        classification = 'binary' #'all'
        path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data','%s_classes' % classification)
        label = 'arrhythmia_'
    elif dataset == 'physionet2017':
        path = os.path.join(basepath,'PhysioNet 2017','patient_data')
        label = ''
    elif dataset == 'tetanus':
        path = '/media/XXXX-2/TertiaryHDD/new_tetanus_data/patient_data'
        label = ''
    elif dataset == 'ptb':
        leads = [leads] #['ii']
        path = os.path.join(basepath,'ptb-diagnostic-ecg-database-1.0.0','patient_data','leads_%s' % leads)
        label = ''
    elif dataset == 'fetal':
        abdomen = 'Abdomen_4'
        path = os.path.join(basepath,'non-invasive-fetal-ecg-arrhythmia-database-1.0.0','patient_data',abdomen)
        label = ''
    elif dataset == 'physionet2016':
        path = os.path.join(basepath,'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0')
        label = ''
    elif dataset == 'physionet2020':
        basepath = 'data'
        leads = leads
        path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data',trial,'leads_%s' % leads)
        label = ''
    elif dataset == 'cpsc2018':
        basepath = '/mnt/SecondaryHDD'
        leads = leads
        path = os.path.join(basepath,'CPSC2018','patient_data',trial,'leads_%s' % leads)
        label = ''

    return path,label

def determine_classification_setting(dataset_name):
    if dataset_name == 'physionet':
        classification = '5-way'
    elif dataset_name == 'bidmc':
        classification = '2-way'
    elif dataset_name == 'mimic': #change this accordingly
        classification = '2-way'
    elif dataset_name == 'cipa':
        classification = '7-way'
    elif dataset_name == 'cardiology':
        classification = '12-way'
    elif dataset_name == 'physionet2017':
        classification = '4-way'
    elif dataset_name == 'tetanus':
        classification = '2-way'
    elif dataset_name == 'ptb':
        classification = '2-way'
    elif dataset_name == 'fetal':
        classification = '2-way'
    elif dataset_name == 'physionet2016':
        classification = '2-way'
    elif dataset_name == 'physionet2020':
        classification = '9-way' #because binary multilabel
    elif dataset_name == 'emg':
        classification = '3-way'
    elif dataset_name == 's':

    return classification

def load_frames_and_labels(path,dataset,peak_detection,label=''):
    if peak_detection == True:
        """ Load ECG Frames and Labels """
        with open(os.path.join(path,'ecg_signal_frames_heartpy_%s.pkl' % dataset),'rb') as f:
            ecg_frames = pickle.load(f)

        with open(os.path.join(path,'ecg_signal_%slabels_heartpy_%s.pkl' % (label,dataset)),'rb') as g:
            ecg_labels = pickle.load(g)

        """ Load PPG Frames and Labels """
        with open(os.path.join(path,'ppg_signal_frames_heartpy_%s.pkl' % dataset),'rb') as f:
            ppg_frames = pickle.load(f)

        with open(os.path.join(path,'ppg_signal_%slabels_heartpy_%s.pkl' % (label,dataset)),'rb') as g:
            ppg_labels = pickle.load(g)
    else:
        try:
            """ Load ECG Frames and Labels """
            print(os.path.join(path,'ecg_signal_frames_%s.pkl' % dataset))
            with open(os.path.join(path,'ecg_signal_frames_%s.pkl' % dataset),'rb') as f:
                ecg_frames = pickle.load(f)

            with open(os.path.join(path,'ecg_signal_%slabels_%s.pkl' % (label,dataset)),'rb') as g:
                ecg_labels = pickle.load(g)
        except:
            print("cannot load ecg frames and labels!")
            ecg_frames = None
            ecg_labels = None

        try:
            """ Load PPG Frames and Labels """
            with open(os.path.join(path,'ppg_signal_frames_%s.pkl' % dataset),'rb') as f:
                ppg_frames = pickle.load(f)

            with open(os.path.join(path,'ppg_signal_%slabels_%s.pkl' % (label,dataset)),'rb') as g:
                ppg_labels = pickle.load(g)
        except:
            print("cannot load ppg frames and labels!")
            ppg_frames = None
            ppg_labels = None

    return ecg_frames, ecg_labels, ppg_frames, ppg_labels

#%%
def remove_patients_with_empty_frames(ecg_frames):
    patients_with_empty_frames = [name for name,frames in ecg_frames.items() if np.array(frames).shape[0] == 0]

    if ecg_frames is not None:
        [ecg_frames.pop(key) for key in patients_with_empty_frames]
        [ecg_labels.pop(key) for key in patients_with_empty_frames]

    if ppg_frames is not None:
        [ppg_frames.pop(key) for key in patients_with_empty_frames]
        [ppg_labels.pop(key) for key in patients_with_empty_frames]

#%%
def obtain_train_test_split(ecg_frames):
    """ Split Patients Into Train, Val, and Test """
    patient_numbers = list(ecg_frames.keys())

    """ Obtain Test Set """
    test_ratio = 0.2
    test_length = int(len(patient_numbers)*test_ratio)
    random.seed(0)
    patient_numbers_test = random.sample(patient_numbers,test_length)
    patient_numbers_train = list(set(patient_numbers) - set(patient_numbers_test))

    """ Obtain Train and Test Split """
    val_ratio = 0.2
    val_length = int(len(patient_numbers_train)*val_ratio)
    random.seed(0)
    patient_numbers_val = random.sample(patient_numbers_train,val_length)
    patient_numbers_train = list(set(patient_numbers_train) - set(patient_numbers_val))

    return patient_numbers_train,patient_numbers_val,patient_numbers_test

#%%
def obtain_patient_number_fraction_dict(fractions,patient_numbers_train,patient_numbers_val,patient_numbers_test):
    """ Obtain Patient-Level Fraction of Training Set As Labelled """
    labelled_patient_dict = {}
    unlabelled_patient_dict = {}
    labelled_patient_numbers_prev = 0
    for n,fraction in enumerate(fractions):
        if n == 0:
            labelled_length = int(len(patient_numbers_train)*fraction)
            random.seed(0)
            labelled_patient_numbers = random.sample(patient_numbers_train,labelled_length)
            unlabelled_patient_numbers = list(set(patient_numbers_train) - set(labelled_patient_numbers))
        else:
            patient_numbers_to_choose = list(set(patient_numbers_train) - set(labelled_patient_numbers_prev))
            current_fraction = fraction - fractions[n-1]
            labelled_length = int(len(patient_numbers_train)*current_fraction)
            random.seed(0)
            labelled_patient_numbers = random.sample(patient_numbers_to_choose,labelled_length)
            labelled_patient_numbers = labelled_patient_numbers + labelled_patient_numbers_prev
            unlabelled_patient_numbers = list(set(patient_numbers_train) - set(labelled_patient_numbers))

        labelled_patient_dict[fraction] = labelled_patient_numbers
        unlabelled_patient_dict[fraction] = unlabelled_patient_numbers

        labelled_patient_numbers_prev = labelled_patient_numbers

    return fractions, labelled_patient_dict, unlabelled_patient_dict

def change_labels(dataset_name,header,noise_level,noise_type,frames,labels):
    """ Introduce Noise to Labels @ Different Intensity Levels
    Frames represent all frames for the 'unlabelled' dataset
    Labels represent all labels for the 'unlabelled' dataset """
    if header == 'unlabelled' and noise_type is not None:
        nlabels = labels.shape[0]
        nlabels_to_switch = int(nlabels*noise_level)
        random.seed(0)
        label_indices_to_switch = random.sample(list(np.arange(nlabels)),nlabels_to_switch)
        for index in label_indices_to_switch:
            original_label = labels[index]
            classification = determine_classification_setting(dataset_name)
            nclasses = int(classification.split('-')[0])
            class_set = set(np.arange(nclasses))
            remaining_class_set = list(class_set - set([original_label]))
            if noise_type == 'random':
                random.seed(0)
                new_label = random.sample(remaining_class_set,1)[0]
            elif noise_type == 'nearest_neighbour':
                pca = PCA(n_components=2)
                pca_frames = pca.fit_transform(frames)
                distance_matrix = np.linalg.norm(pca_frames - pca_frames[:,None], axis=-1)
                distance_matrix[distance_matrix==0] = 1e9 #to avoid choosing diagonal entry
                closest_indices = np.argmin(distance_matrix,1)
                new_labels = labels[closest_indices]
                new_label = new_labels[index]

            labels[index] = new_label

    return labels

#%%
def obtain_arrays(dataset_name,fractions,ecg_frames,ecg_labels,ppg_frames,ppg_labels,labelled_patient_dict,unlabelled_patient_dict,noise_type=None,noise_level=None):
    """ Split Data Into Phases and Save Into Dicts """
    modalities = ['ecg']#,'ppg'] #change depending on modality (or both)
    #phases = ['train','val','test']

    frames_dict = dict()
    labels_dict = dict()
    pid_dict = dict()

    for modality in modalities:

        frames_dict[modality] = dict()
        labels_dict[modality] = dict()
        pid_dict[modality] = dict()

        if modality == 'ecg':
            modality_frames = ecg_frames
            modality_labels = ecg_labels
        elif modality == 'ppg':
            modality_frames = ppg_frames #this is a dict
            modality_labels = ppg_labels #this is a dict

        nframes_per_patient = [array.shape[0] for array in list(modality_labels.values())]
        nframes_per_patient_dict = dict(zip(modality_labels.keys(),nframes_per_patient))

        for fraction in tqdm(fractions):

            train_labelled_patients = labelled_patient_dict[fraction]
            train_unlabelled_patients = unlabelled_patient_dict[fraction]

            frames_dict[modality][fraction] = dict()
            labels_dict[modality][fraction] = dict()
            pid_dict[modality][fraction] = dict()

            frames_dict[modality][fraction]['train'] = dict()
            labels_dict[modality][fraction]['train'] = dict()
            pid_dict[modality][fraction]['train'] = dict()

            train_headers = ['labelled','unlabelled']
            train_patients = [train_labelled_patients,train_unlabelled_patients]
            for header,patient_numbers in zip(train_headers,train_patients):
                if len(patient_numbers) == 1:
                    frames = np.array(modality_frames[patient_numbers[0]])
                    frames_dict[modality][fraction]['train'][header] = frames

                    labels = np.array(modality_labels[patient_numbers[0]])
                    labels = change_labels(dataset_name,header,noise_level,noise_type,frames,labels)
                    labels_dict[modality][fraction]['train'][header] = labels

                    pid = [patient_numbers[0] for _ in range(nframes_per_patient_dict[patient_numbers[0]])]
                    pid_dict[modality][fraction]['train'][header] = pid
                elif len(patient_numbers) > 1:
                    frames = np.concatenate(list(itemgetter(*patient_numbers)(modality_frames)))
                    frames_dict[modality][fraction]['train'][header] = frames

                    labels = np.concatenate(list(itemgetter(*patient_numbers)(modality_labels)))
                    labels = change_labels(dataset_name,header,noise_level,noise_type,frames,labels)
                    labels_dict[modality][fraction]['train'][header] = labels

                    pid = [patient_number for patient_number in patient_numbers for _ in range(nframes_per_patient_dict[patient_number])]
                    pid_dict[modality][fraction]['train'][header] = pid

            remaining_phases = ['val','test']
            remaining_patients = [patient_numbers_val,patient_numbers_test]
            remaining_content = dict(zip(remaining_phases,remaining_patients))

            for phase,patient_numbers in remaining_content.items():
                if len(patient_numbers) == 1:
                    frames_dict[modality][fraction][phase] = np.array(modality_frames[patient_numbers[0]])
                    labels_dict[modality][fraction][phase] = np.array(modality_labels[patient_numbers[0]])
                    pid = [patient_numbers[0] for _ in range(nframes_per_patient_dict[patient_numbers[0]])]
                    pid_dict[modality][fraction][phase] = pid
                elif len(patient_numbers) > 1:
                    frames_dict[modality][fraction][phase] = np.concatenate(list(itemgetter(*patient_numbers)(modality_frames))) #indices #list(itemgetter(*indices)(patient_number_list))
                    labels_dict[modality][fraction][phase] = np.concatenate(list(itemgetter(*patient_numbers)(modality_labels)))
                    pid = [patient_number for patient_number in patient_numbers for _ in range(nframes_per_patient_dict[patient_number])]
                    pid_dict[modality][fraction][phase] = pid

    return frames_dict,labels_dict,pid_dict
#%%
def make_directory(path,noise_level=None):
    if noise_level is not None:
        path = os.path.join(path,'noise_level_%.2f' % noise_level)
        try:
            os.chdir(path)
        except:
            os.makedirs(path)
    return path

def save_final_frames_and_labels(frames_dict,labels_dict,pid_dict,path,peak_detection,noise_level=None,label=''):
    if peak_detection == True:
        """ Save Frames and Labels Dicts """
        with open(os.path.join(path,'frames_phases_heartpy_%s.pkl' % dataset),'wb') as f:
            pickle.dump(frames_dict,f)

        with open(os.path.join(path,'labels_phases_heartpy_%s.pkl' % (dataset)),'wb') as g:
            pickle.dump(labels_dict,g)

        with open(os.path.join(path,'pid_phases_heartpy_%s.pkl' % (dataset)),'wb') as h:
            pickle.dump(pid_dict,h)
    else:
        """ Save Frames and Labels Dicts """
        with open(os.path.join(path,'frames_phases_%s.pkl' % dataset),'wb') as f:
            pickle.dump(frames_dict,f)

        with open(os.path.join(path,'labels_phases_%s.pkl' % (dataset)),'wb') as g:
            pickle.dump(labels_dict,g)

        with open(os.path.join(path,'pid_phases_%s.pkl' % (dataset)),'wb') as h:
            pickle.dump(pid_dict,h)

    print('Final Frames Saved!')
#%%
if __name__ == '__main__':
    fractions = [1] #[0.1,0.3,0.5,0.7,0.9]
    leads_list = [['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']] #[['II','V2','aVL','aVR']] #[['I','II','III','aVL','aVR','aVF','V1','V2','V3','V4','V5','V6']] #['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6'] #for ptb dataset
    """ Noisy Label Formulation """
    noise_type = None #random' OR 'nearest_neighbour'
    leads = None
    """ End Noisy Label Formulation """
    for trial in trials:
        for leads in leads_list:
        #for noise_level in noise_level_list:
            path, label = load_path(dataset,leads)
            ecg_frames, ecg_labels, ppg_frames, ppg_labels = load_frames_and_labels(path,dataset,peak_detection,label)
            remove_patients_with_empty_frames(ecg_frames)
            patient_numbers_train, patient_numbers_val, patient_numbers_test = obtain_train_test_split(ecg_frames)
            fractions, labelled_patient_dict, unlabelled_patient_dict = obtain_patient_number_fraction_dict(fractions,patient_numbers_train,patient_numbers_val,patient_numbers_test)
            frames_dict, labels_dict, pid_dict = obtain_arrays(dataset,fractions,ecg_frames,ecg_labels,ppg_frames,ppg_labels,labelled_patient_dict,unlabelled_patient_dict)#noise_type,noise_level
            path = make_directory(path)#,noise_level)
            save_final_frames_and_labels(frames_dict,labels_dict,pid_dict,path,peak_detection)#,noise_level)



