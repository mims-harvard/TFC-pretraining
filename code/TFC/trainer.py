import os
import sys
sys.path.append("..")

from loss import *
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from model import * 

def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def Trainer(model,  model_optimizer, train_dl, valid_dl, device, config, training_mode, save_model_dir, save_model_or_checkpoints):
    print("Training started ....\n")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    train_loss_set = []
    val_loss_set = []
    if training_mode == 'pre_train':
        for epoch in range(1, config.num_epoch + 1):
            train_loss = model_pretrain(model, model_optimizer, criterion, train_dl, config, device, training_mode)
            train_loss_set.append(train_loss)
            val_loss = model_pretrain_val(model, model_optimizer, criterion, valid_dl, config, device, training_mode)
            val_loss_set.append(val_loss)
            print(f'Pre-training Epoch : {epoch}', f'Train Loss : {train_loss:.4f}')
            print(f'Pre-training Epoch : {epoch}', f'Val Loss : {val_loss:.4f}\n')

            if save_model_or_checkpoints.lower() == 'checkpoints': 
                # chkpoint = {'model_state_dict': model.state_dict()}
                torch.save(model.state_dict(), f"{save_model_dir}/model_{epoch}.pt")

        if save_model_or_checkpoints.lower() == 'model': 
            # chkpoint = {'model_state_dict': model.state_dict()}
            torch.save(model.state_dict(), f"{save_model_dir}/model_final_epoch.pt")
        # os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        # chkpoint = {'model_state_dict': model.state_dict()}
        # torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        # print('Pretrained model is stored at folder:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))
    
    print("Training End")
    return train_loss_set, val_loss_set

def model_pretrain(model, model_optimizer, criterion, train_loader, config, device, training_mode,):
    total_loss = []
    model.train()
    global loss, loss_t, loss_f, l_TF, loss_c, data_test, data_f_test

    # optimizer
    model_optimizer.zero_grad()

    for batch_idx, (data, aug1, data_f, aug1_f) in enumerate(train_loader):
        data = data.float().to(device)
        aug1 = aug1.float().to(device)  # aug1 = aug2 : [128, 1, 178]
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)

        lam = 0.2
        loss = lam*(loss_t + loss_f) + l_TF

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    # print('Pretraining: train loss:{:.3f}, l_t: {:.3f}, l_f:{:.3f}, l_c:{:.3f}'.format(loss, loss_t, loss_f, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss


def model_pretrain_val(model, model_optimizer, criterion, val_loader, config, device, training_mode,):
    total_loss = []
    model.eval()
    global val_loss, val_loss_t, val_loss_f, val_l_TF, val_loss_c, data_test, data_f_test

    # no model update 
    with torch.no_grad():
        for batch_idx, (data, aug1, data_f, aug1_f) in enumerate(val_loader):
            data, aug1 = data.float().to(device), aug1.float().to(device) # [batch, fea_count, ts_count]
            data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)  # aug1 = aug2 : [128, 1, 178]

            """Produce embeddings"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

            """Compute Pre-train loss"""
            """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
            nt_xent_criterion = NTXentLoss_poly(device, config.batch_size, config.Context_Cont.temperature,
                                        config.Context_Cont.use_cosine_similarity) # device, 128, 0.2, True

            val_loss_t = nt_xent_criterion(h_t, h_t_aug)
            val_loss_f = nt_xent_criterion(h_f, h_f_aug)
            val_l_TF = nt_xent_criterion(z_t, z_f) # this is the initial version of TF loss

            l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), nt_xent_criterion(z_t_aug, z_f_aug)
            val_loss_c = (1 + val_l_TF - l_1) + (1 + val_l_TF - l_2) + (1 + val_l_TF - l_3)

            lam = 0.2
            val_loss = lam*(val_loss_t + val_loss_f) + val_l_TF
            total_loss.append(val_loss.item())

        # print('Pretraining: validation loss:{:.3f}, l_t: {:.3f}, l_f:{:.3f}, l_c:{:.3f}'.format(val_loss, val_loss_t, val_loss_f, val_l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss
