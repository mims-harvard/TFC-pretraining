import torch as th
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F
import sklearn

th.manual_seed(42)
np.random.seed(42)

def to_np(x):
    return x.cpu().detach().numpy()

import os
seed = 1
pretrained = 'sleepEDF'
alias = 'epilepsy'
n_classes = 2
basepath = f'{os.getcwd()}/data'

x_tr = np.load(os.path.join(basepath, alias,  f"train_input.npy"))
y_tr = np.load(os.path.join(basepath, alias,  f"train_output.npy"))
x_te = np.load(os.path.join(basepath, alias, f"test_input.npy"))
y_te = np.load(os.path.join(basepath, alias, f"test_output.npy"))


class MyDataset(Dataset):
    def __init__(self, x, y):

        device = 'cuda'
        self.x = th.tensor(x, dtype=th.float, device=device)
        self.y = th.tensor(y, dtype=th.long, device=device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class FCN(nn.Module):
    def __init__(self, n_in):
        super(FCN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(n_in, 128, kernel_size=7, padding=6, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        self.proj_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):

        h = self.encoder(x)
        out = self.proj_head(h)

        return out, h

class FCN_clf(nn.Module):
    def __init__(self, fcn_model):
        super(FCN_clf, self).__init__()
        self.encoder = fcn_model
        self.encoder.train()
        self.proj_head = nn.Sequential(
            #nn.Linear(128, 4),
            #nn.BatchNorm1d(4),
            #nn.ReLU(),
            nn.Linear(128, n_classes)
        )


    def forward(self, x):
        _, feats = self.encoder(x)
        return self.proj_head(feats)

def train_mixup_model_epoch(model, training_set, test_set, optimizer, alpha, epochs):

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size_tr = 20 #len(training_set.x)
    LossList, AccList
    criterion = nn.CrossEntropyLoss()

    training_generator = DataLoader(training_set, batch_size=batch_size_tr,
                                    shuffle=True, drop_last=True)

    for epoch in range(epochs):

        for x, y in training_generator:
            model.train()
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y[:,0])
            loss.backward()
            optimizer.step()
            LossList.append(loss.item())

        AccList.append(test_model(model, training_set, test_set))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {LossList[-1]}")
        print(f"Accuracy: {AccList[-1]}")
        print("-"*50)

    return LossList, AccList

def test_model(model, training_set, test_set):

    model.eval()
    N_te = len(test_set.x)
    print(N_te)
    test_generator = DataLoader(test_set, batch_size= 1,
                                    shuffle=True, drop_last=False)
    yhat_te = th.zeros((N_te, n_classes))
    y_te = th.zeros((N_te), dtype=th.long)

    for idx_te, (x_te, y_te_i) in enumerate(test_generator):
        with th.no_grad():
            yhat_te_i = model(x_te)
#             print(x_te, yhat_te_i)
            yhat_te[idx_te] = model(x_te) # yhat are pre-softmax values
            y_te[idx_te] = y_te_i

    target = y_te
    target_prob = F.one_hot(target, num_classes=n_classes)
    pred_prob = yhat_te
    pred = pred_prob.argmax(dim=1)
#    print(target)
#   print(pred)
#    print(pred_prob)
    metrics_dict = {}
    metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
    print(metrics_dict)
    return metrics_dict['Accuracy']

# Finally, training the model!!
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

np.random.seed(0)
x_tr, y_tr = unison_shuffled_copies(x_tr, y_tr)
x_te, y_te = unison_shuffled_copies(x_te, y_te)
ntrain = len(x_tr) # set the size of partial training set to use

device = 'cuda' if th.cuda.is_available() else 'cpu'
epochs, LossList, AccList = 100, [], []

alpha = 1.0

training_set = MyDataset(x_tr[0:ntrain,:], y_tr[0:ntrain,:])
test_set = MyDataset(x_te, y_te)
model = th.load(os.path.join(basepath, pretrained, 'model'+str(seed)), map_location=th.device('cuda'))
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)

# reset weights of proj_head
# for name, layer in model.named_children():
#     if name == 'proj_head':
#         for n, l in layer.named_modules():
#             if hasattr(l, 'reset_parameters'):
#                 l.reset_parameters()
model = FCN_clf(model).to(device)

optimizer = th.optim.Adam(model.parameters())
LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)
