#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:19:32 2020

@author: Dani Kiyasseh
"""

import torch.optim as optim
import torch
import os

#%%
""" Functions in this script:
    1) load_initial_model_contrastive
    2) obtain_noutputs
"""

#%%
def load_initial_model_contrastive(cnn_network,phases,save_path_dir,saved_weights,held_out_lr,nencoders,embedding_dim,trial_to_run,task,dataset_name,second_network='',classification=''):
    """ Load models with maml weights """
    dropout_type = 'drop1d' #options: | 'drop1d' | 'drop2d'
    p1,p2,p3 = 0.1,0.1,0.1 #initial dropout probabilities #0.2, 0, 0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   
    model = cnn_network(dropout_type,p1,p2,p3,nencoders=nencoders,embedding_dim=embedding_dim,trial=trial_to_run,device=device)
    print(device)
    """ Inference Without Meta Learning """
    if trial_to_run in ['Linear','Fine-Tuning','Random']:
        if 'train1' in phases:
            
            if trial_to_run in ['Linear','Fine-Tuning']:
                model.load_state_dict(torch.load(os.path.join(save_path_dir,saved_weights)))
                print('Pretrained Model Loaded')
                
                if trial_to_run == 'Linear':
                    for param in model.parameters(): #freeze representation weights 
                        param.requires_grad_(False)
                    print('Params Frozen')
                
            model.to(device)
            """ Load Second Model for Classification """
            noutputs = obtain_noutputs(classification,dataset_name)
            model = second_network(model,noutputs,embedding_dim)
            print('Model Extended')
            
        elif 'test' in phases and len(phases) == 1 or 'val' in phases and len(phases) == 1:
            model.to(device)
            noutputs = obtain_noutputs(classification,dataset_name)
            model = second_network(model,noutputs,embedding_dim)
            #print(save_path_dir)
            model.load_state_dict(torch.load(os.path.join(save_path_dir,saved_weights)))
    else:
        if 'obtain_representation' in task:
            model.load_state_dict(torch.load(os.path.join(save_path_dir,saved_weights)))
            print('Finetuned Weights Loaded!')            
        
        if 'test' in phases and len(phases) == 1 or 'val' in phases and len(phases) == 1:
            model.load_state_dict(torch.load(os.path.join(save_path_dir,saved_weights)))
            print('Finetuned Weights Loaded!')
    
    model.to(device)
    #print(next(model.parameters()).is_cuda)
    optimizer = optim.Adam(list(model.parameters()),lr=held_out_lr,weight_decay=0) #shouldn't load this again - will lose running average of gradients and so forth

    return model,optimizer,device

def obtain_noutputs(classification,dataset_name):
    if 'physionet2020' in dataset_name:
        noutputs = 9 #int(classification.split('-')[0])
    elif 'ptbxl' in dataset_name:
        noutputs = 12 #71
#     elif classification == '2-way':
#         noutputs = 1
    else:
        noutputs = int(classification.split('-')[0])
    return noutputs
