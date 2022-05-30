#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:22:28 2020

@author: Dani Kiyasseh
"""

from torch.utils.data import DataLoader
from prepare_dataset import my_dataset_contrastive

""" Functions in this script:
    1) load_initial_data_contrastive
"""
#%%

def load_initial_data_contrastive(basepath_to_data,phases,fraction,inferences,batch_size,modality,acquired_indices,
                                  acquired_labels,modalities,dataset_name,input_perturbed=False,perturbation='Gaussian',
                                  leads='ii',labelled_fraction=1,unlabelled_fraction=1,downstream_task='contrastive',
                                  class_pair='',trial='CMC',nviews=1):
    """ Control augmentation at beginning of training here """ 
    resize = False
    affine = False
    rotation = False
    color = False    
    perform_cutout = False
    operations = {'resize': resize, 'affine': affine, 'rotation': rotation, 'color': color, 'perform_cutout': perform_cutout}    
    shuffles = {'train': True,
                'train1':True,
                'train2':False,
                'val': False,
                'test': False}
    
    fractions = {'fraction': fraction,
                 'labelled_fraction': labelled_fraction,
                 'unlabelled_fraction': unlabelled_fraction}
    
    acquired_items = {'acquired_indices': acquired_indices,
                      'acquired_labels': acquired_labels}

    #print("dataset_name", dataset_name, "downstream_task", downstream_task)
    dataset = {phase:my_dataset_contrastive(basepath_to_data,dataset_name,phase,inference,fractions,acquired_items,modalities=modalities,task=downstream_task,input_perturbed=input_perturbed,perturbation=perturbation,leads=leads,class_pair=class_pair,trial=trial,nviews=nviews)
                for phase,inference in zip(phases,inferences)}
#     print(dataset[phases[0]].__getitem__(0))
#     exit(1)
#    if 'train' in phases:
#        check_dataset_allignment(mixture,dataset_list)

    dataloader = {phase:DataLoader(dataset[phase],batch_size=batch_size,shuffle=shuffles[phase],drop_last=False) for phase in phases}
    return dataloader,operations