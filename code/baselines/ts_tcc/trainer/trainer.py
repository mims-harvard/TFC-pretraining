import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from tqdm import tqdm

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device, logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc, train_metrics = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _, val_metrics = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if training_mode not in ['ts_sd','self_supervised']:  # use scheduler in all other modes.
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}\n'
                     f'{train_metrics}\n'
                     f'{val_metrics}')

    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode not in ["ts_sd", "self_supervised"]:  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _, metrics = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\n'
                     f'{metrics}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config, device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()
    total_preds = []
    total_labels = []
    for batch_idx, (data, labels, aug1, aug2) in tqdm(enumerate(train_loader)):
        total_labels.append(labels.to('cpu'))
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised":
            _, features1 = model(aug1)
            _, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_lstm_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_lstm_feat2 = temporal_contr_model(features2, features1)

            # normalize projection feature vectors
            zis = temp_cont_lstm_feat1 
            zjs = temp_cont_lstm_feat2 

        elif training_mode == "ts_sd": # note, in config files, just use gaussian noise, this is to match the denoising
                                     # task used in the ts_sd paper
            base_signal = aug1
            noisy_signal = aug2
            denoised_signal = temporal_contr_model(noisy_signal)
        elif training_mode == "ts_sd_finetune":
            base_signal = aug1
            output = (temporal_contr_model(base_signal, mode='finetune'), None)

        else: # supervised
            output = model(data)

        # compute loss
        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2

        elif training_mode == "ts_sd":
            #print(base_signal - denoised_signal)
            loss = 100 * ((base_signal - denoised_signal)**2).mean()

        else: # supervised training or fine tuining
            predictions, _ = output
            total_preds.append(predictions.detach().to('cpu') )
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    metrics_dict = {}
    if len(total_preds) > 0:
        total_preds = torch.vstack(tuple(total_preds))
        total_labels = torch.cat(tuple(total_labels))
        pred_prob = total_preds
        pred = pred_prob.argmax(dim=1)
        target = total_labels
        print(target.shape)
        print(target)
        target_prob = F.one_hot(target.long(), num_classes=model.n_classes)
        metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
        metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
        metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
        metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
        metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
    else:
        metrics_dict['Precision'] = None
        metrics_dict['Recall'] = None
        metrics_dict['F1'] = None
        metrics_dict['AUROC'] = None
        metrics_dict['AUPRC'] = None
    return total_loss, total_acc, metrics_dict


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    probs = []
    trgs = np.array([])
    total_preds = []
    total_labels = []
    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            total_labels.append(labels.to('cpu'))
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode in ["self_supervised", "ts_sd"]:
                pass
            elif training_mode == "ts_sd_finetune":
                output = (temporal_contr_model(data, mode='finetune'), None)
            else:
                output = model(data)

            # compute loss
            if training_mode not in ["self_supervised", "ts_sd"]:
                predictions, features = output
                total_preds.append(predictions.detach().to('cpu') )
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                exp_logits = np.exp(predictions.cpu().numpy())
                probs.append(exp_logits / (1 + exp_logits))
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode not in ["self_supervised", "ts_sd"]:
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode in ["self_supervised", "ts_sd"]:
        total_acc = 0
        return total_loss, total_acc, [], [], {}
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
#         scattered_trgs = np.zeros((len(trgs), 3))
#         np.put_along_axis(scattered_trgs, np.expand_dims(trgs.astype(int), axis=1), 1, axis=1)
#         probs = np.vstack(tuple(probs))
#         print('auroc: ', roc_auc_score(scattered_trgs, normalize(probs, axis=1), multi_class='ovr'))

        metrics_dict = {}
        if len(total_preds) > 0:
            total_preds = torch.vstack(tuple(total_preds))
            total_labels = torch.cat(tuple(total_labels))
            pred_prob = total_preds
            pred = pred_prob.argmax(dim=1)
            target = total_labels
            target_prob = F.one_hot(target.long(), num_classes=model.n_classes)
            metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
            metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
            metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
            metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
            metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
        else:
            metrics_dict['Precision'] = None
            metrics_dict['Recall'] = None
            metrics_dict['F1'] = None
            metrics_dict['AUROC'] = None
            metrics_dict['AUPRC'] = None
        return total_loss, total_acc, outs, trgs, metrics_dict
