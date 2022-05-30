#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 23:26:21 2020

@author: Dani Kiyasseh
"""
import pickle
import os
import torch.nn as nn
import torch
import numpy as np
from itertools import combinations
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from tabulate import tabulate
#%%
""" Functions in this script:
    1) flatten_arrays
    2) obtain_contrastive_loss
    3) calculate_auc
    4) change_labels_type
    5) print_metrics
    6) save_metrics
    7) track_metrics
    8) save_config_weights
    9) save_patient_representations
    10) determine_classification_setting
    11) modify_dataset_order_for_multi_task_learning
    12) obtain_saved_weights_name
    13) make_dir
    14) make_saving_directory_contrastive
    15) obtain_information
    16) obtain_criterion
"""
#%%

def flatten_arrays(outputs_list,labels_list,modality_list,indices_list,task_names_list,pids_list):
    outputs_list = np.concatenate(outputs_list)
    labels_list = np.concatenate(labels_list)
    modality_list = np.concatenate(modality_list)
    indices_list = np.concatenate(indices_list)
    task_names_list = np.concatenate(task_names_list)
    pids_list = np.concatenate(pids_list)
    return outputs_list, labels_list, modality_list, indices_list, task_names_list, pids_list

def obtain_contrastive_loss(latent_embeddings,pids,trial):
    """ Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
        pids (list): patient ids of instances in batch
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    """
    if trial in ['CMSC','CMLC','CMSMLC']:
        pids = np.array(pids,dtype=np.object)   
        pid1,pid2 = np.meshgrid(pids,pids)
        pid_matrix = str(pid1) + '-' + str(pid2)
        pids_of_interest = np.unique(str(pid1) + '-' + str(pid2)) #unique combinations of pids of interest i.e. matching
        bool_matrix_of_interest = np.zeros((len(pids),len(pids)))
        for pid in pids_of_interest:
            bool_matrix_of_interest += pid_matrix == pid
        rows1,cols1 = np.where(np.triu(bool_matrix_of_interest,1))
        rows2,cols2 = np.where(np.tril(bool_matrix_of_interest,-1))

    nviews = set(range(latent_embeddings.shape[2]))
    view_combinations = combinations(nviews,2)
    loss = 0
    ncombinations = 0
    for combination in view_combinations:
        view1_array = latent_embeddings[:,:,combination[0]] #(BxH)
        view2_array = latent_embeddings[:,:,combination[1]] #(BxH)
        norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
        norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
        sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
        norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
        temperature = 0.1
        argument = sim_matrix/(norm_matrix*temperature)
        sim_matrix_exp = torch.exp(argument)
        
        if trial == 'CMC':
            """ Obtain Off Diagonal Entries """
            #upper_triangle = torch.triu(sim_matrix_exp,1)
            #lower_triangle = torch.tril(sim_matrix_exp,-1)
            #off_diagonals = upper_triangle + lower_triangle
            diagonals = torch.diag(sim_matrix_exp)
            """ Obtain Loss Terms(s) """
            loss_term1 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,1)))
            loss_term2 = -torch.mean(torch.log(diagonals/torch.sum(sim_matrix_exp,0)))
            loss += loss_term1 + loss_term2 
            loss_terms = 2
        elif trial == 'SimCLR':
            self_sim_matrix1 = torch.mm(view1_array,view1_array.transpose(0,1))
            self_norm_matrix1 = torch.mm(norm1_vector.transpose(0,1),norm1_vector)
            temperature = 0.1
            argument = self_sim_matrix1/(self_norm_matrix1*temperature)
            self_sim_matrix_exp1 = torch.exp(argument)
            self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1,1) + torch.tril(self_sim_matrix_exp1,-1)
            
            self_sim_matrix2 = torch.mm(view2_array,view2_array.transpose(0,1))
            self_norm_matrix2 = torch.mm(norm2_vector.transpose(0,1),norm2_vector)
            temperature = 0.1
            argument = self_sim_matrix2/(self_norm_matrix2*temperature)
            self_sim_matrix_exp2 = torch.exp(argument)
            self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2,1) + torch.tril(self_sim_matrix_exp2,-1)

            denominator_loss1 = torch.sum(sim_matrix_exp,1) + torch.sum(self_sim_matrix_off_diagonals1,1)
            denominator_loss2 = torch.sum(sim_matrix_exp,0) + torch.sum(self_sim_matrix_off_diagonals2,0)
            
            diagonals = torch.diag(sim_matrix_exp)
            loss_term1 = -torch.mean(torch.log(diagonals/denominator_loss1))
            loss_term2 = -torch.mean(torch.log(diagonals/denominator_loss2))
            loss += loss_term1 + loss_term2
            loss_terms = 2
        elif trial in ['CMSC','CMLC','CMSMLC']: #ours #CMSMLC = positive examples are same instance and same patient
            triu_elements = sim_matrix_exp[rows1,cols1]
            tril_elements = sim_matrix_exp[rows2,cols2]
            diag_elements = torch.diag(sim_matrix_exp)
            
            triu_sum = torch.sum(sim_matrix_exp,1)
            tril_sum = torch.sum(sim_matrix_exp,0)
            
            loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
            loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
            
            loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
            loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
            
            loss = loss_diag1 + loss_diag2
            loss_terms = 2

            if len(rows1) > 0:
                loss += loss_triu #technically need to add 1 more term for symmetry
                loss_terms += 1
            
            if len(rows2) > 0:
                loss += loss_tril #technically need to add 1 more term for symmetry
                loss_terms += 1
        
            #print(loss,loss_triu,loss_tril)

        ncombinations += 1
    loss = loss/(loss_terms*ncombinations)
    return loss

def calculate_auc(classification,outputs_list,labels_list,save_path_dir):
    ohe = LabelBinarizer()
    labels_ohe = ohe.fit_transform(labels_list)
    if classification is not None:
        if classification != '2-way':
            all_auc = []
            for i in range(labels_ohe.shape[1]):
                auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                all_auc.append(auc)
            epoch_auroc = np.mean(all_auc)
        elif classification == '2-way':
            if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir:
                """ Use This for MultiLabel Process -- Only for Physionet2020 """
                all_auc = []
                for i in range(labels_ohe.shape[1]):
                    auc = roc_auc_score(labels_ohe[:,i],outputs_list[:,i])
                    all_auc.append(auc)
                epoch_auroc = np.mean(all_auc)
            else:
                epoch_auroc = roc_auc_score(labels_list,outputs_list)
    else:
        print('This is not a classification problem!')
    return epoch_auroc

def calculate_acc(outputs_list,labels_list,save_path_dir):
    if 'physionet2020' in save_path_dir or 'ptbxl' in save_path_dir: #multilabel scenario
        """ Convert Preds to Multi-Hot Vector """
        preds_list = np.where(outputs_list>0.5,1,0)
        """ Indices of Hot Vectors of Predictions """
        preds_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in preds_list]
        """ Indices of Hot Vectors of Ground Truth """
        labels_list = [np.where(multi_hot_vector)[0] for multi_hot_vector in labels_list]
        """ What Proportion of Labels Did you Get Right """
        acc = np.array([np.isin(preds,labels).sum() for preds,labels in zip(preds_list,labels_list)]).sum()/(len(np.concatenate(preds_list)))        
    else: #normal single label setting 
        preds_list = torch.argmax(torch.tensor(outputs_list),1)
        ncorrect_preds = (preds_list == torch.tensor(labels_list)).sum().item()
        acc = ncorrect_preds/preds_list.shape[0]
    return acc

def change_labels_type(labels,criterion):
    if isinstance(criterion,nn.BCEWithLogitsLoss):
        labels = labels.type(torch.float)
    elif isinstance(criterion,nn.CrossEntropyLoss):
        labels = labels.type(torch.long)
    return labels

def print_metrics(phase,results_dictionary):
    metric_name_to_label = {'epoch_loss':'loss','epoch_auroc':'auc','epoch_acc':'acc'}
    items_to_print = dict()
    labels = []
    for metric_name,result in results_dictionary.items():
        label = metric_name_to_label[metric_name]
        labels.append('-'.join((phase,label)))
        items_to_print[label] = ['%.4f' % result]
    print(tabulate(items_to_print,labels))

def save_metrics(save_path_dir,prefix,metrics_dict):
    torch.save(metrics_dict,os.path.join(save_path_dir,'%s_metrics_dict' % prefix))

def track_metrics(metrics_dict,results_dictionary,phase,epoch_count):
    for metric_name,results in results_dictionary.items():
        
        if epoch_count == 0 and ('train' in phase): #or 'test' in phase):
            metrics_dict[metric_name] = dict()
        
        if epoch_count == 0:
            metrics_dict[metric_name][phase] = []
        
        metrics_dict[metric_name][phase].append(results)
    return metrics_dict

def save_config_weights(save_path_dir,best_model_weights,saved_weights_name,phases,trial,downstream_dataset): #which is actually second_dataset
    if trial in ['Linear','Fine-Tuning','Random']:
        saved_weights_name = 'finetuned_weight'
    torch.save(best_model_weights,os.path.join(save_path_dir,saved_weights_name))

def save_patient_representation(save_path_dir,patient_rep_dict,trial):
    if trial not in ['Linear','Fine-Tuning']:
        with open(os.path.join(save_path_dir,'patient_rep'),'wb') as f:
            pickle.dump(patient_rep_dict,f)

def determine_classification_setting(dataset_name,trial):
    #dataset_name = dataset_name[0]
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
        classification = '2-way' #because binary multilabel
    elif dataset_name == 'chapman':
        classification = '4-way'
    elif dataset_name == 'chapman_pvc':
        classification = '2-way'
    elif dataset_name == 'emg':
        classification = '3-way'
    elif dataset_name == 'sleepEDF':
        classification = '5-way'
    elif dataset_name == 'epilepsy':
        classification = '2-way'
    elif dataset_name == 'pFD_A':
        classification = '3-way'
    elif dataset_name == 'pFD_B':
        classification = '3-way'
    elif dataset_name == 'HAR':
        classification = '6-way'
    elif dataset_name == 'AHAR':
        classification = '8-way'
    else: #used for pretraining with contrastive learning
        classification = None
    #print('Original Classification %s' % classification)
    return classification

def modify_dataset_order_for_multi_task_learning(dataset,modalities,leads,class_pairs,fractions):
    dataset = [dataset] #outside of if statement because dataset is original during each iteration
    if not isinstance(fractions,list): #it is already in list format, therefore no need for extra list
        modalities = [modalities]
        leads = [leads]
        class_pairs = [class_pairs]
        fractions = [fractions]
    return dataset,modalities,leads,class_pairs,fractions

def obtain_saved_weights_name(trial,phases):
    if trial not in ['Linear','Fine-Tuning','Random']:
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to save 
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'pretrained_weight' #name of weights to load
    elif trial in ['Linear','Fine-Tuning','Random']:
        if 'train' in phases:
            saved_weights = 'pretrained_weight' #name of weights to load
        elif 'val' in phases and len(phases) == 1 or 'test' in phases and len(phases) == 1:
            saved_weights = 'finetuned_weight' #name of weights to load
    return saved_weights

def obtain_load_path_dir(phases,save_path_dir,trial_to_run,second_dataset,labelled_fraction,leads,max_seed,task,evaluation=False):
    if trial_to_run in ['Linear','Fine-Tuning','Random']:
        labelled_fraction_path = 'training_fraction_%.2f' % labelled_fraction
        leads_path = 'leads_%s' % str(leads[0]) #remember leads is a list of lists 
        if trial_to_run in ['Random']:
            trial_to_run = ''
            if second_dataset in ['chapman','physionet2020']:
                leads_path = 'leads_%s' % str(leads[0]) #only these two datasets have multiple leads
            else:
                leads_path = ''

        if leads[0] == None:
            leads_path = ''

        save_path_dir = os.path.join(save_path_dir,trial_to_run,second_dataset,leads_path,labelled_fraction_path)        
        #print(save_path_dir)
        if 'train' in phases:
            save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=True,evaluation=evaluation) #do NOT change second_pass = True b/c this function is only ever used during second pass
        elif 'test' in phases:
            if 'test_metrics_dict' in os.listdir(save_path_dir):
                save_path_dir = 'do not test'
        
        if save_path_dir in ['do not train','do not test']:
            load_path_dir = save_path_dir
        else:
            split_save_path_dir = save_path_dir.split('/')
            seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
            load_path_dir = '/'.join(split_save_path_dir[:seed_index+1]) #you want to exclude everything AFTER the seed path
    else:
        load_path_dir = save_path_dir

    print(load_path_dir)
    print(save_path_dir)
    
    return load_path_dir, save_path_dir

def make_saving_directory_contrastive(phases,dataset_name,trial_to_load,trial_to_run,seed,max_seed,task,embedding_dim,leads,input_perturbed,perturbation,evaluation=False):
    # base_path = '/mnt/SecondaryHDD/Contrastive Learning Results'
    base_path = f'{os.getcwd()}/results'
    seed_path = 'seed%i' % int(seed)
    dataset_path = dataset_name#[0] #dataset used for training
    if leads is None:
        leads_path = ''
    else:
        leads_path = 'leads_%s' % str(leads) #leads used for training
    embedding_path = 'embedding_%i' % embedding_dim #size of embbedding used
    if trial_to_run in ['Linear','Fine-Tuning']:
        trial_path = trial_to_load
    elif trial_to_run in ['Random']:
        trial_path = trial_to_run
        dataset_path, leads_path = '', ''
    else:
        trial_path = trial_to_run
    
    if input_perturbed == True:
        perturbed_path = 'perturbed'
        perturbation_path = str(perturbation)
    elif input_perturbed == False:
        perturbed_path = ''
        perturbation_path = ''
    
    save_path_dir = os.path.join(base_path,trial_path,dataset_path,leads_path,embedding_path,perturbed_path,perturbation_path,seed_path)
    
    if 'train' in phases:
        save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,evaluation=evaluation)
    elif 'test' in phases:
        if 'test_metrics_dict' in os.listdir(save_path_dir):
            save_path_dir = 'do not test'
    
    return save_path_dir, seed

def make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=False,evaluation=False): #boolean allows you to overwrite if TRUE 
    """ Recursive Function to Make Sure I do Not Overwrite Previous Seeds """
    split_save_path_dir = save_path_dir.split('/')
    seed_index = np.where(['seed' in token for token in split_save_path_dir])[0].item()
    current_seed = int(split_save_path_dir[seed_index].strip('seed'))
    try:
        if second_pass == False:
            condition = ('obtain_representation' not in task) and (trial_to_run not in ['Linear','Fine-Tuning'])
        elif second_pass == True:
            condition = ('obtain_representation' not in task)
        
        if condition:# and trial_to_run not in ['Linear','Fine-Tuning']: #do not skip if you need to do finetuning
            os.chdir(save_path_dir)
            if 'train_val_metrics_dict' in os.listdir() and evaluation == False:
                if current_seed < max_seed-1:
                    print('Skipping Seed!')
                    new_seed = current_seed + 1
                    seed_path = 'seed%i' % new_seed
                    save_path_dir = save_path_dir.replace('seed%i' % current_seed,seed_path)
                    save_path_dir, seed = make_dir(save_path_dir,max_seed,task,trial_to_run,second_pass=second_pass,evaluation=evaluation)
                else:
                    save_path_dir = 'do not train'
    except:
        os.makedirs(save_path_dir)
    
    if os.path.isdir(save_path_dir) == False: #just in case we miss making the directory somewhere
        os.makedirs(save_path_dir)
    
    if current_seed == max_seed:
        current_seed = 0
    
    return save_path_dir, current_seed

def obtain_information(trial,downstream_dataset,second_dataset,data2leads_dict,data2bs_dict,data2lr_dict,data2classpair_dict):
    if trial in ['Linear','Fine-Tuning','Random']:
        training_dataset = second_dataset
    else:
        training_dataset = downstream_dataset #used for contrastive training 
    leads = data2leads_dict[training_dataset]
    batch_size = data2bs_dict[training_dataset]
    held_out_lr = data2lr_dict[training_dataset]
    class_pair = data2classpair_dict[training_dataset]

    if second_dataset == 'emg':
        modalities = ['emg']
    elif second_dataset in ['sleepEDF', 'epilepsy']:
        modalities = ['eeg']
    elif second_dataset in ['pFD_A', 'pFD_B','HAR','AHAR']:
        modalities = ['other']
    else:
        modalities = ['ecg']
    fraction = 1 #1 for chapman, physio2020, and physio2017. Use labelled_fraction for control over fraction of training data used 
    return leads, batch_size, held_out_lr, class_pair, modalities, fraction       

def obtain_criterion(classification):
#     if classification == '2-way':
#         criterion = nn.BCEWithLogitsLoss()
#     else:
    criterion = nn.CrossEntropyLoss()
    return criterion
