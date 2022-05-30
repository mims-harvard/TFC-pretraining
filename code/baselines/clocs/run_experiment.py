#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:28:57 2020

@author: Dani Kiyasseh
"""

import numpy as np
import copy

from prepare_miscellaneous import obtain_criterion, print_metrics, track_metrics, save_metrics, save_config_weights, save_patient_representation
from prepare_models import load_initial_model_contrastive
from prepare_dataloaders import load_initial_data_contrastive
from perform_training import one_epoch_finetuning, one_epoch_contrastive
#%%
""" Functions in this script:
    1) train_model
"""
#%%

def train_model(basepath_to_data,cnn_network_contrastive,second_cnn_network,classification,load_path_dir,save_path_dir,
                seed,batch_size,held_out_lr,fraction,modalities,leads,saved_weights,phases,downstream_dataset,
                downstream_task,class_pair,input_perturbed,perturbation,trial_to_load=None,trial_to_run=None,
                nencoders=1,embedding_dim=256,nviews=1,labelled_fraction=1,num_epochs=250):
    """ Training and Validation For All Epochs """
    best_loss = float('inf')
    metrics_dict = dict()
    patient_rep_dict = dict()
    if phases[0] == 'train':
        phases[0] = 'train1'  # Another hack because I don't understand the distinction b/w train and train1
    if 'test' not in phases:
        phases = ['train1','val']
        inferences = [False,False]
    else:
        inferences = [False] * len(phases)
    
    stop_counter = 0
    patience = 15 #for early stopping criterion
    epoch_count = 0
    
    """ Added April 24th, 2020 """
    criterion = obtain_criterion(classification)
    if 'train1' in phases or 'train' in phases:
        model_path_dir = load_path_dir #original use-case
    elif 'test' in phases:
        model_path_dir = save_path_dir
    """ End """
    model,optimizer,device = load_initial_model_contrastive(cnn_network_contrastive,phases,model_path_dir,saved_weights,held_out_lr,nencoders,embedding_dim,trial_to_run,downstream_task,downstream_dataset,second_network=second_cnn_network,classification=classification)
    
    weighted_sampling = []
    acquired_indices = [] #indices of the unlabelled data
    acquired_labels = dict() #network labels of the unlabelled data
    dataloader,operations = load_initial_data_contrastive(basepath_to_data,phases,fraction,inferences,batch_size,modalities,acquired_indices,acquired_labels,modalities,downstream_dataset,downstream_task=downstream_task,input_perturbed=input_perturbed,perturbation=perturbation,leads=leads,class_pair=class_pair,trial=trial_to_run,nviews=nviews,labelled_fraction=labelled_fraction)
    """ Obtain Number of Labelled Samples """
    #total_labelled_samples = len(dataloaders_list['train1'].batch_sampler.sampler.data_source.label_array)
    
    while stop_counter <= patience and epoch_count < num_epochs:
        if 'train' in phases or 'val' in phases:
            print('-' * 10)
            print('Epoch %i/%i' % (epoch_count,num_epochs-1))
            print('-' * 10)
                            
        """ ACTUAL TRAINING AND EVALUATION """
        for phase,inference in zip(phases,inferences):
            if 'train1' in phase or 'train' in phase:
                model.train()
            elif phase == 'val' or phase == 'test':
                model.eval()
            
            if trial_to_run in ['Linear','Fine-Tuning','Random']:
                results_dictionary, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = one_epoch_finetuning(weighted_sampling,phase,inference,dataloader,model,optimizer,device,criterion,classification,trial=trial_to_run,epoch_count=epoch_count,save_path_dir=save_path_dir)
            else:
                results_dictionary, outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list = one_epoch_contrastive(weighted_sampling,phase,inference,dataloader,model,optimizer,device,trial=trial_to_run,epoch_count=epoch_count,save_path_dir=save_path_dir)
            """ Store Representations for Each Patient """
            unique_pids = np.unique(pids_list)
            pid_dict = dict()
            for pid in unique_pids:
                pid_indices = np.where(np.in1d(pids_list,pid))[0]
                pid_outputs = outputs_list[pid_indices]
                pid_dict[pid] = pid_outputs
            patient_rep_dict[phase] = pid_dict
            
            if inference == False:
                print_metrics(phase,results_dictionary)
                epoch_loss = results_dictionary['epoch_loss']
                if (phase == 'val' and epoch_loss < best_loss) or (phase == 'test' and epoch_loss < best_loss):
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    """ Save Best Finetuned Weights """
                    if 'train1' in phases or 'train' in phases:
                        save_config_weights(save_path_dir,best_model_wts,saved_weights,phases,trial_to_run,downstream_dataset)
                        save_patient_representation(save_path_dir,patient_rep_dict,trial_to_run)
                    stop_counter = 0
                elif phase == 'val' and epoch_loss >= best_loss:
                    stop_counter += 1
                
                metrics_dict = track_metrics(metrics_dict,results_dictionary,phase,epoch_count)                
#                 print(metrics_dict)
        epoch_count += 1
        if 'train1' not in phases and 'train' not in phases:
            break #from while loop
        elif ('train1' in phases or 'train' in phases) and 'obtain_representation' in downstream_task:
            break
            
    #print('Best Val Loss: %.4f.' % best_loss)
    if 'train1' in phases or 'train' in phases:
        prefix = 'train_val'
        save_metrics(save_path_dir,prefix,metrics_dict)
        model.load_state_dict(best_model_wts)
    elif 'val' in phases:
        prefix = 'val'
        save_metrics(save_path_dir,prefix,metrics_dict)
    elif 'test' in phases:
        prefix = 'test'
        save_metrics(save_path_dir,prefix,metrics_dict)