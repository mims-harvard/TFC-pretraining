#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:16:38 2020

@author: Dani Kiyasseh
"""
import torch
from torch.utils.data import Dataset
from operator import itemgetter
import os
import pickle
import numpy as np
#import os
import random
#from scipy.stats import entropy
#import pandas as pd
#%%

window_len = 178 // 2

class my_dataset_contrastive(Dataset):
    """ Takes Arrays and Phase, and Returns Sample 
        i.e. use for BIDMC and PhysioNet Datasets 
    """
    
    def __init__(self,basepath_to_data,dataset_name,phase,inference,fractions,acquired_items, modalities=['ecg','ppg'],
                 task='contrastive',input_perturbed=False,perturbation='Gaussian',leads='ii',heads='single',
                 cl_scenario=None,class_pair='',trial='CMC',nviews=1):
        """ This Accounts for 'train1' and 'train2' Phases """
        if 'train' in phase:
            phase = 'train'
        elif 'val' in phase:
            phase = 'val'

        #if 'contrastive' in task:
        #    task = 'contrastive'
        self.task = task #continual_buffer, etc. 
        self.cl_scenario = cl_scenario
        self.basepath_to_data = basepath_to_data
        if not (task == 'contrastive_ss' and trial in ['Linear','Fine-Tuning','Random']):#!= 'multi_task_learning':
            input_array,output_array,pid_array = self.load_raw_inputs_and_outputs(dataset_name,leads)
            self.output_array = output_array #original output dict
        #print(output_array['ecg'][0.9]['train']['labelled'].shape)
        fraction = fractions['fraction'] #needs to be a list when dealing with 'query' or inference = True for CL scenario
        labelled_fraction = fractions['labelled_fraction']
        unlabelled_fraction = fractions['unlabelled_fraction']
        acquired_indices = acquired_items['acquired_indices']
        acquired_labels = acquired_items['acquired_labels']
        #print(len(acquired_indices.values()))
        """ Combine Modalities into 1 Array """
        frame_array = []
        label_array = []
        self.modalities = modalities
        self.dataset_name = dataset_name
        self.heads = heads
        self.acquired_items = acquired_items
        self.fraction = fraction
        self.labelled_fraction = labelled_fraction
        self.leads = leads
        self.class_pair = class_pair
        self.nviews = nviews
        self.trial = trial
        self.input_perturbed = input_perturbed
        self.perturbation = perturbation
#        self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks

        if task == 'self-supervised':
            for modality in modalities:
                if phase == 'train':
                    modality_input = np.concatenate(list(input_array[modality][fraction][phase].values()))
                    modality_output = np.concatenate(list(output_array[modality][fraction][phase].values()))
                else:
                    modality_input = input_array[modality][fraction][phase]
                    modality_output = output_array[modality][fraction][phase]          
                    #modality_input = input_array[modality][fraction][phase][train_key]
                    #modality_output = output_array[modality][fraction][phase][train_key]
    
                frame_array.append(modality_input)
                label_array.append(modality_output)

            self.input_array = np.concatenate(frame_array)
            self.label_array = [i for i in range(len(modalities)) for _ in range(modality_input.shape[0])] 
        
        elif trial in ['Linear','Fine-Tuning','Random']: #task == 'contrastive_ss' and
            if phase == 'train':
                inputs, outputs, pids = self.retrieve_multi_task_train_data()
            else:
                inputs, outputs, pids = self.retrieve_multi_task_val_data(phase)
            keep_indices = list(np.arange(inputs.shape[0])) #filler
            modality_array = list(np.arange(inputs.shape[0])) #filler
            dataset_list = ['All' for _ in range(len(keep_indices))] #filler
            self.dataset_name = dataset_name
            self.dataset_list = dataset_list
            self.input_array = inputs
            self.label_array = outputs
            self.modality_array = modality_array
            self.remaining_indices = keep_indices
            self.pids = pids

        else: #normal training path for CPPC, CMC, etc. 
            self.name = '-'.join((dataset_name,modalities[0],str(fraction),leads,class_pair)) #name for different tasks
            if phase == 'train':
                if inference == False:
                    inputs,outputs,pids = self.expand_labelled_data(input_array,output_array,pid_array,fraction,labelled_fraction,unlabelled_fraction,acquired_indices,acquired_labels)
                    #outputs = self.offset_outputs(dataset_name,outputs)
                    
                    keep_indices = list(np.arange(inputs.shape[0])) #filler
                    modality_array = list(np.arange(inputs.shape[0])) #filler
                elif inference == True: #==> when MC Dropout is Performed
                    inputs,outputs,modality_array,keep_indices = self.retrieve_modified_unlabelled_data(input_array,output_array,fraction,unlabelled_fraction,acquired_indices)
                    #outputs = self.offset_outputs(dataset_name,outputs)
                    
                    #print(keep_indices)
                    #if input_perturbed == True: #perturb for consistency acquisition metrics
                    #    inputs = self.perturb_inputs(inputs,dataset_name)
            else:
                inputs,outputs,pids = self.retrieve_val_data(input_array,output_array,pid_array,phase,fraction,dataset_name=dataset_name)
                #outputs = self.offset_outputs(dataset_name,outputs)
                keep_indices = list(np.arange(inputs.shape[0])) #filler
                modality_array = list(np.arange(inputs.shape[0])) #filler
            
            dataset_list = [self.name for _ in range(len(keep_indices))] #filler
            self.dataset_name = dataset_name
            self.dataset_list = dataset_list
            self.input_array = inputs
            self.label_array = outputs
            self.pids = pids
            self.modality_array = modality_array
            self.remaining_indices = keep_indices
        
        self.input_perturbed = input_perturbed 
        self.phase = phase

    def load_raw_inputs_and_outputs(self,dataset_name,leads='i'):
        """ Load Arrays Based on dataset_name """
        #basepath = '/home/scro3517/Desktop'
        basepath = self.basepath_to_data
        
        if dataset_name == 'bidmc':
            path = os.path.join(basepath,'BIDMC v1')
            extension = 'heartpy_'
        elif dataset_name == 'physionet':
            path = os.path.join(basepath,'PhysioNet v2')
            extension = 'heartpy_'
        elif dataset_name == 'mimic':
            shrink_factor = str(0.1)
            path = os.path.join(basepath,'MIMIC3_WFDB','frame-level',shrink_factor)
            extension = 'heartpy_'
        elif dataset_name == 'cipa':
            lead = ['II','aVR']
            path = os.path.join(basepath,'cipa-ecg-validation-study-1.0.0','leads_%s' % lead)
            extension = ''
        elif dataset_name == 'cardiology':
            leads = 'all' #flexibility to change later 
            path = os.path.join(basepath,'CARDIOL_MAY_2017','patient_data',self.task,'%s_classes' % leads)
            extension = ''
        elif dataset_name == 'physionet2017':
            path = os.path.join(basepath,'physionet2017',self.task)
            extension = ''
        elif dataset_name == 'tetanus':
            path = '/media/scro3517/TertiaryHDD/new_tetanus_data/patient_data'
            extension = ''
        elif dataset_name == 'ptb':
            leads = [leads]
            path = os.path.join(basepath,'ptb-diagnostic-ecg-database-1.0.0','patient_data','leads_%s' % leads)
            extension = ''  
        elif dataset_name == 'fetal':
            abdomen = leads #'Abdomen_1'
            path = os.path.join(basepath,'non-invasive-fetal-ecg-arrhythmia-database-1.0.0','patient_data',abdomen)
            extension = ''
        elif dataset_name == 'physionet2016':
            path = os.path.join(basepath,'classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0')
            extension = ''
        elif dataset_name == 'physionet2020':
            #basepath = '/mnt/SecondaryHDD'
            leads_name = leads
            path = os.path.join(basepath,'PhysioNetChallenge2020_Training_CPSC','Training_WFDB','patient_data',self.task,'leads_%s' % leads_name)
            extension = ''
        elif dataset_name == 'chapman':
            #basepath = '/mnt/SecondaryHDD'
            leads = leads
            path = os.path.join(basepath,'chapman_ecg',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'chapman_pvc':
            #basepath = '/mnt/SecondaryHDD'
            leads = leads
            path = os.path.join(basepath,'PVCVTECGData',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'emg':
            leads = leads
            path = os.path.join(basepath,'emg',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'sleepEDF':
            leads = leads
            path = os.path.join(basepath,'sleepEDF',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'epilepsy':
            leads = leads
            path = os.path.join(basepath,'epilepsy',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'pFD_A':
            leads = leads
            path = os.path.join(basepath,'pFD_A',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'pFD_B':
            leads = leads
            path = os.path.join(basepath,'pFD_B',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'HAR':
            leads = leads
            path = os.path.join(basepath,'HAR',self.task,'leads_%s' % leads)
            extension = ''
        elif dataset_name == 'AHAR':
            leads = leads
            path = os.path.join(basepath,'AHAR',self.task,'leads_%s' % leads)
            extension = ''

        if self.cl_scenario == 'Class-IL':
            dataset_name = dataset_name + '_' + 'mutually_exclusive_classes'
        """ Dict Containing Actual Frames """ # damn!!! it's a dict, not array
        with open(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as f:
            print(os.path.join(path,'frames_phases_%s%s.pkl' % (extension,dataset_name)),'rb')
            input_array = pickle.load(f)
#             print(input_array['ecg'][1]['train']['labelled'].shape)
        """ Dict Containing Actual Labels """
        with open(os.path.join(path,'labels_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as g:
            output_array = pickle.load(g)
        """ Dict Containing Patient Numbers """
        with open(os.path.join(path,'pid_phases_%s%s.pkl' % (extension,dataset_name)),'rb') as h:
            pid_array = pickle.load(h) #needed for CPPC (ours)
        return input_array,output_array,pid_array

    def offset_outputs(self,dataset_name,outputs,t=0): #t tells you which class pair you are on now (used rarely and only for MTL)
        """ Offset Label Position in case of Single Head """
        dataset_and_offset = self.acquired_items['noutputs']
        if self.heads == 'single':
            """ Changed March 17th 2020 """
            offset = dataset_and_offset[dataset_name] #self.dataset_name
            """ End """
            if dataset_name == 'physionet2020': #multilabel situation 
                """ Option 1 - Expansion """
                #noutputs = outputs.shape[1] * 12 #9 classes and 12 leads
                #expanded_outputs = np.zeros((outputs.shape[0],noutputs))
                #expanded_outputs[:,offset:offset+9] = outputs
                #outputs = expanded_outputs
                """ Option 2 - No Expansion """
                outputs = outputs 
            else: 
                if dataset_name == 'cardiology' and self.task == 'multi_task_learning':
                    outputs = outputs + 2*t
                elif dataset_name == 'chapman' and self.task == 'multi_task_learning':
                    outputs = outputs
                else:
                    outputs = outputs + offset #output represents actual labels
                #print(offset)
        return outputs

    def retrieve_buffered_data(self,buffer_indices_dict,fraction,labelled_fraction):
        input_buffer = []
        output_buffer = []
        task_indices_buffer = []
        dataset_buffer = []
        #print(fraction_list)
        #for fraction,(task_name,indices) in zip(fraction_list[:-1],buffer_indices_dict.items()):
        for task_name,indices in buffer_indices_dict.items():
            #name = '-'.join((task,modality,leads,str(fraction))) #dataset, modality, fraction, leads
            dataset = task_name.split('-')[0]
            fraction = float(task_name.split('-')[2])
            leads = task_name.split('-')[3]
            if self.cl_scenario == 'Class-IL':
                self.class_pair = '-'.join(task_name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
            elif self.cl_scenario == 'Time-IL':
                self.class_pair = task_name.split('-')[-1] 
            elif self.cl_scenario == 'Task-IL' and 'chapman' in dataset: #chapman ecg as task in Task-IL setting
                self.class_pair = task_name.split('-')[-1]
            input_array,output_array = self.load_raw_inputs_and_outputs(dataset,leads)
            input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=dataset)
            """ Offset Applied to Each Dataset """
            if self.heads == 'single':#'continual_buffer':
                output_array = self.offset_outputs(dataset,output_array)
                #offset = self.dataset_and_offset[dataset]
                #output_array = output_array + offset
            current_input_buffer,current_output_buffer = input_array[indices,:], output_array[indices]
            input_buffer.append(current_input_buffer)
            output_buffer.append(current_output_buffer)
            task_indices_buffer.append(indices) #will go 1-10K, 1-10K, etc. not cumulative indices
            dataset_buffer.append([task_name for _ in range(len(indices))])
        #print(input_buffer)
        input_buffer = np.concatenate(input_buffer,axis=0)
        output_buffer = np.concatenate(output_buffer,axis=0)
        task_indices_buffer = np.concatenate(task_indices_buffer,axis=0)
        dataset_buffer = np.concatenate(dataset_buffer,axis=0)
        return input_buffer,output_buffer,task_indices_buffer,dataset_buffer
                
    def expand_labelled_data_with_buffer(self,input_array,output_array,buffer_indices_dict,fraction,labelled_fraction):
        """ function arguments are raw inputs and outputs """
        input_buffer,output_buffer,task_indices_buffer,dataset_buffer = self.retrieve_buffered_data(buffer_indices_dict,fraction,labelled_fraction)
        if self.cl_scenario == 'Class-IL':
            self.class_pair = '-'.join(self.name.split('-')[-2:]) #b/c e.g. '0-1' you need last two
        elif self.cl_scenario == 'Time-IL':
            self.class_pair = self.name.split('-')[-1]
        input_array,output_array = self.retrieve_labelled_data(input_array,output_array,fraction,labelled_fraction,dataset_name=self.dataset_name)
        dataset_list = [self.name for _ in range(input_array.shape[0])]
        #print(max(output_array))
        """ Offset Applied to Current Dataset """
        if self.heads == 'single':#'continual_buffer':
            output_array = self.offset_outputs(self.dataset_name,output_array)
            #offset = self.dataset_and_offset[self.dataset_name]
            #print('Offset')
            #print(offset)
            #output_array = output_array + offset
        print(input_array.shape,input_buffer.shape)
        input_array = np.concatenate((input_array,input_buffer),0)
        output_array = np.concatenate((output_array,output_buffer),0)
        dataset_list = np.concatenate((dataset_list,dataset_buffer),0)
        #print(input_array.shape)
        #print(max(output_array),max(output_buffer))
        return input_array,output_array,dataset_list
    
    def retrieve_val_data(self,input_array,output_array,pid_array,phase,fraction,labelled_fraction=1,dataset_name=''):#,modalities=['ecg','ppg']):
        frame_array = []
        label_array = []
        pids = []
        #if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or (self.cl_scenario == 'Task-IL' and self.dataset_name == 'chapman'):        
        if dataset_name in ['chapman', 'emg', 'sleepEDF', 'epilepsy', 'pFD_A', 'pFD_B', 'HAR', 'AHAR', 'physionet2017']:
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase][self.class_pair]
                modality_output = output_array[modality][fraction][phase][self.class_pair]
                modality_pids = pid_array[modality][fraction][phase][self.class_pair]
                frame_array.append(modality_input)
                label_array.append(modality_output)
                pids.append(modality_pids)
        else:
            """ Obtain Modality-Combined Unlabelled Dataset """
            for modality in self.modalities:
                modality_input = input_array[modality][fraction][phase]
                modality_output = output_array[modality][fraction][phase]
                modality_pids = pid_array[modality][fraction][phase]
                frame_array.append(modality_input)
                label_array.append(modality_output) 
                pids.append(modality_pids)                
        
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)
        pids = np.concatenate(pids)
        
        inputs,outputs,pids,_ = self.shrink_data(inputs,outputs,pids,labelled_fraction)
        return inputs,outputs,pids          
    
    def shrink_data(self,inputs,outputs,pids,fraction,modality_array=None):
        nframes_to_sample = int(inputs.shape[0]*fraction)
        random.seed(0) #to make sure we always obtain SAME shrunken dataset for reproducibility 
        indices = random.sample(list(np.arange(inputs.shape[0])),nframes_to_sample)
        inputs = np.array(list(itemgetter(*indices)(inputs)))
        outputs = np.array(list(itemgetter(*indices)(outputs)))
        pids = np.array(list(itemgetter(*indices)(pids)))
        if modality_array is not None:
            modality_array = np.array(list(itemgetter(*indices)(modality_array)))
        return inputs,outputs,pids,modality_array
    
    def remove_acquired_data(self,inputs,outputs,modality_array,acquired_indices):
        keep_indices = list(set(list(np.arange(inputs.shape[0]))) - set(acquired_indices))
        inputs = np.array(list(itemgetter(*keep_indices)(inputs)))
        outputs = np.array(list(itemgetter(*keep_indices)(outputs)))
        modality_array = np.array(list(itemgetter(*keep_indices)(modality_array)))
        return inputs,outputs,modality_array,keep_indices
    
    def retrieve_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []
        modality_array = []
        
        """ Obtain Modality-Combined Unlabelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase]['unlabelled']
            modality_output = output_array[modality][fraction][phase]['unlabelled']
            modality_name = [modality for _ in range(modality_input.shape[0])]
            frame_array.append(modality_input)
            label_array.append(modality_output)
            modality_array.append(modality_name)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)   
        modality_array = np.concatenate(modality_array)         
        
        inputs,outputs,modality_array = self.shrink_data(inputs,outputs,unlabelled_fraction,modality_array)
        
        return inputs,outputs,modality_array
        
    ### This is function you want for MC Dropout Phase ###
    def retrieve_modified_unlabelled_data(self,input_array,output_array,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(input_array,output_array,fraction,unlabelled_fraction)
        inputs,outputs,modality_array,keep_indices = self.remove_acquired_data(inputs,outputs,modality_array,acquired_indices)
        return inputs,outputs,modality_array,keep_indices

    def retrieve_labelled_data(self,input_array,output_array,pid_array,fraction,labelled_fraction,dataset_name=''):#,modalities=['ecg','ppg']):
        phase = 'train'
        frame_array = []
        label_array = []
        pids = []

        if self.cl_scenario == 'Class-IL' or self.cl_scenario == 'Time-IL' or dataset_name in ['chapman', 'emg', 'sleepEDF', 'epilepsy', 'pFD_A', 'pFD_B', 'HAR', 'AHAR', 'physionet2017']:
            header = self.class_pair
        elif self.cl_scenario == 'Task-IL' and dataset_name == 'chapman':
            header = self.class_pair
        else:
            header = 'labelled'
        
        """ Obtain Modality-Combined Labelled Dataset """
        for modality in self.modalities:
            modality_input = input_array[modality][fraction][phase][header]
            print("modality input shape:", modality_input.shape)
            modality_output = output_array[modality][fraction][phase][header]
            modality_pids = pid_array[modality][fraction][phase][header]
            frame_array.append(modality_input)
            label_array.append(modality_output)
            pids.append(modality_pids)
        """ Flatten Datasets to Get One Array """
        inputs = np.concatenate(frame_array)
        outputs = np.concatenate(label_array)
        pids = np.concatenate(pids)
        inputs,outputs,pids,_ = self.shrink_data(inputs,outputs,pids,labelled_fraction)
        return inputs,outputs,pids

    def acquire_unlabelled_samples(self,inputs,outputs,fraction,unlabelled_fraction,acquired_indices):
        inputs,outputs,modality_array = self.retrieve_unlabelled_data(inputs,outputs,fraction,unlabelled_fraction)
        if len(acquired_indices) > 1:
            inputs = np.array(list(itemgetter(*acquired_indices)(inputs)))
            outputs = np.array(list(itemgetter(*acquired_indices)(outputs)))
            modality_array = np.array(list(itemgetter(*acquired_indices)(modality_array)))
        elif len(acquired_indices) == 1:
            """ Dimensions Need to be Adressed to allow for Concatenation """
            inputs = np.expand_dims(np.array(inputs[acquired_indices[0],:]),1)
            outputs = np.expand_dims(np.array(outputs[acquired_indices[0]]),1)
            modality_array = np.expand_dims(np.array(modality_array[acquired_indices[0]]),1)
        return inputs,outputs,modality_array

    def retrieve_multi_task_train_data(self):
        """ Load All Required Tasks for Multi-Task Training Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        pids = []
        for t,(dataset_name,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = all_modalities[t] #list(map(lambda x:x[0],all_modalities))
            #print(self.labelled_fraction)
            current_input, current_output, pid = self.retrieve_labelled_data(current_input,current_output,pid,fraction,self.labelled_fraction,dataset_name=dataset_name)
            #current_output = self.offset_outputs(dataset_name,current_output,t)
            #print(current_output.shape)
            input_array.append(current_input)
            output_array.append(current_output)
            pids.append(pid)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        pids = np.concatenate(pids,axis=0)
        print('Output Dimension: %s' % str(output_array.shape))
        print('Maximum Output Index: %i' % np.max(output_array))
        return input_array,output_array,pids

    def retrieve_multi_task_val_data(self,phase):
        """ Load All Required Tasks for Multi-Task Validation/Testing Setting """
        all_class_pairs = self.class_pair
        all_modalities = self.modalities
        input_array = []
        output_array = []
        pids = []
        for t,(dataset_name,modalities,fraction,leads,class_pair) in enumerate(zip(self.dataset_name,all_modalities,self.fraction,self.leads,all_class_pairs)): #should be an iterable list
            current_input, current_output, pid = self.load_raw_inputs_and_outputs(dataset_name,leads=leads)
            self.class_pair = class_pair
            self.modalities = modalities
            current_input, current_output, pid = self.retrieve_val_data(current_input,current_output,pid,phase,fraction,dataset_name=dataset_name)#,labelled_fraction=1)
            #current_output = self.offset_outputs(dataset_name,current_output,t)
            input_array.append(current_input)
            output_array.append(current_output)
            pids.append(pid)
        input_array = np.concatenate(input_array,axis=0)
        output_array = np.concatenate(output_array,axis=0)
        pids = np.concatenate(pids,axis=0)
        
        return input_array,output_array,pids

    ### This is function you want for training ###
    def expand_labelled_data(self,input_array,output_array,pid_array,fraction,labelled_fraction,unlabelled_fraction,acquired_indices,acquired_labels):
        inputs,outputs,pids = self.retrieve_labelled_data(input_array,output_array,pid_array,fraction,labelled_fraction,self.dataset_name)
        #print(self.remaining_indices)
        #print('Acquired Indices!')
        #print(acquired_indices)
        """ If indices have been acquired, then use them. Otherwise, do not """
        #if isinstance(acquired_indices,list):
        #    condition = len(acquired_indices) > 0
        #elif isinstance(acquired_indices,dict):
        #    condition = len(acquired_indices) > 1
        """ Changed March 5, 2020 """
        #if len(acquired_indices) > 0:
        if len(acquired_indices) > 0:
            acquired_inputs,acquired_outputs,acquired_modalities = self.acquire_unlabelled_samples(input_array,output_array,fraction,unlabelled_fraction,acquired_indices)
            inputs = np.concatenate((inputs,acquired_inputs),0)
            #print(acquired_labels)
            #""" Note - Acquired Labels from Network Predictions are Used, Not Ground Truth """
            #acquired_labels = np.fromiter(acquired_labels.values(),dtype=float)
            acquired_labels = np.array(list(acquired_labels.values()))
            acquired_labels = acquired_labels.reshape((-1,))
            ##""" For cold_gt trials, run this line """
            ##acquired_labels = np.array(list(acquired_labels.values()))
            ##acquired_labels = acquired_labels.reshape((-1,))
            ##print(outputs.shape,acquired_labels.shape)
            ##""" End GT Labels """
            
            #print(acquired_labels)
            outputs = np.concatenate((outputs,acquired_labels),0) 
        return inputs,outputs,pids
    
    def obtain_perturbed_frame(self,frame):
        """ Apply Sequence of Perturbations to Frame 
        Args:
            frame (numpy array): frame containing ECG data
        Outputs
            frame (numpy array): perturbed frame based
        """
        if self.input_perturbed:
            if 'Gaussian' in self.perturbation:
                mult_factor = 1
                if self.dataset_name in ['ptb','physionet2020']:
                    variance_factor = 0.01*mult_factor
                elif self.dataset_name in ['cardiology','chapman']:
                    variance_factor = 10*mult_factor
                elif self.dataset_name in ['physionet','physionet2017']:
                    variance_factor = 100*mult_factor 
                gauss_noise = np.random.normal(0,variance_factor,size=(window_len))
                frame = frame + gauss_noise
            
            if 'FlipAlongY' in self.perturbation:
                frame = np.flip(frame)
            
            if 'FlipAlongX' in self.perturbation:
                frame = -frame
        return frame

    def normalize_frame(self,frame):
        if self.dataset_name not in ['cardiology','physionet2017','physionet2016']:# or self.dataset_name != 'physionet2017':# or self.dataset_name != 'cipa':
            if isinstance(frame,np.ndarray):
                frame = (frame - np.min(frame))/(np.max(frame) - np.min(frame) + 1e-8)
            elif isinstance(frame,torch.Tensor):
                frame = (frame - torch.min(frame))/(torch.max(frame) - torch.min(frame) + 1e-8)
        return frame

    def __getitem__(self,index):
        true_index = self.remaining_indices[index] #this should represent indices in original unlabelled set
        input_frame = self.input_array[index]
        label = self.label_array[index]
        pid = self.pids[index]
        modality = self.modality_array[index]
        dataset = self.dataset_list[index] #task name

        if input_frame.dtype != float:
            input_frame = np.array(input_frame,dtype=float)
        
        nsamples = input_frame.shape[0] #(5000,) for my approach, #2500 for CMC approach (OURS1)
        if self.trial == 'CMSC':
            """ Start My Approach Patient Specific """
            frame = torch.tensor(input_frame,dtype=torch.float)
            label = torch.tensor(label,dtype=torch.float)
            frame = frame.unsqueeze(0) #(1,5000)
            frame_views = torch.empty(1,window_len,self.nviews)
            start = 0
            for n in range(self.nviews):
                current_view = frame[0,start:start+window_len]
                current_view = self.obtain_perturbed_frame(current_view)
                current_view = self.normalize_frame(current_view)
                frame_views[0,:,n] = current_view
                start += window_len
            """ End My Approach Patient Specific """
        elif self.trial == 'CMLC': #contrastive multi-lead coding (OURS2)
            frame = torch.tensor(input_frame,dtype=torch.float)
            label = torch.tensor(label,dtype=torch.float)
            frame = frame.unsqueeze(0) #(1,2500,12) #SxL = Samples x leads
            frame_views = torch.empty(1,window_len,self.nviews)#self.nviews) #nviews = nleads in this case (1x2500x12)
            start = 0
            for n in range(self.nviews): #nviews = # of leads
                current_view = frame[0,:,n]
                current_view = self.obtain_perturbed_frame(current_view)
                current_view = self.normalize_frame(current_view)
                frame_views[0,:,n] = current_view
        elif self.trial == 'CMSMLC': #contrastive multi-segment multi-lead coding (OURS3)
            frame = torch.tensor(input_frame,dtype=torch.float)
            label = torch.tensor(label,dtype=torch.float)
            frame = frame.unsqueeze(0) #(1,5000,12) #SxL = Samples x Leads
            frame_views = torch.empty(1,window_len,self.nviews*2)#self.nviews) #nviews = nleads in this case (1x2500x12*nsegments)
            nsegments = frame.shape[1]//window_len
            fcount = 0
            for n in range(self.nviews): #nviews = # of leads
                for s in range(nsegments):
                    start = s*window_len
                    current_view = frame[0,start:start+window_len,n]
                    current_view = self.obtain_perturbed_frame(current_view)
                    current_view = self.normalize_frame(current_view)
                    frame_views[0,:,fcount] = current_view  
                    fcount += 1
        elif self.trial in ['CMC','SimCLR']:
            frame_views = torch.empty(1,nsamples,self.nviews)
            for n in range(self.nviews):
                """ Obtain Differing 'Views' of Same Instance by Perturbing Input Frame """
                frame = self.obtain_perturbed_frame(input_frame)
                """ Normalize Data Frame """
                frame = self.normalize_frame(frame)
                frame = torch.tensor(frame,dtype=torch.float)
                label = torch.tensor(label,dtype=torch.float)
                """ Frame Input Has 1 Channel """
                frame = frame.unsqueeze(0)
                """ Populate Frame Views """
                frame_views[0,:,n] = frame
        elif self.trial in ['Linear','Fine-Tuning','Random']:
            frame = self.normalize_frame(input_frame)
            frame = torch.tensor(frame,dtype=torch.float)
            label = torch.tensor(label,dtype=torch.float)
            frame = frame.unsqueeze(0) #(1,5000)
            frame_views = frame.unsqueeze(2) #to show that there is only 1 view (1x2500x1)
        if isinstance(pid, np.ndarray):
            pid = pid[0]
        #print("get item returns:", frame_views,label,pid,modality,dataset,true_index)

        return frame_views,label,pid,modality,dataset,true_index
        
    def __len__(self):
        return len(self.input_array)
