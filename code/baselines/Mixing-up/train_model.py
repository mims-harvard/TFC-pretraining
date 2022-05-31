import torch as th
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt


#from IPython.display import clear_output
#from sktime.datasets import load_gunpoint
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import KNeighborsClassifier
seed = 1

def to_np(x):
    return x.cpu().detach().numpy()

import os
window_len = 178
alias = 'sleepEDF'
n_channels = 1
basepath = f'{os.getcwd()}/data'

x_tr = np.load(os.path.join(basepath, alias, f"train_input.npy"))
y_tr = np.load(os.path.join(basepath, alias, f"train_output.npy"))
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

class MixUpLoss(th.nn.Module):

    def __init__(self, device, batch_size):
        super(MixUpLoss, self).__init__()

        self.tau = 0.5
        self.device = device
        self.batch_size = batch_size
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, z_aug, z_1, z_2, lam):

        z_1 = nn.functional.normalize(z_1)
        z_2 = nn.functional.normalize(z_2)
        z_aug = nn.functional.normalize(z_aug)

        labels_lam_0 = lam*th.eye(self.batch_size, device=self.device)
        labels_lam_1 = (1-lam)*th.eye(self.batch_size, device=self.device)

        labels = th.cat((labels_lam_0, labels_lam_1), 1)

        logits = th.cat((th.mm(z_aug, z_1.T),
                         th.mm(z_aug, z_2.T)), 1)

        loss = self.cross_entropy(logits / self.tau, labels)

        return loss

    def cross_entropy(self, logits, soft_targets):
        return th.mean(th.sum(- soft_targets * self.logsoftmax(logits), 1))

def train_mixup_model_epoch(model, training_set, test_set, optimizer, alpha, epochs):

    device = 'cuda' if th.cuda.is_available() else 'cpu'
    batch_size_tr = 32 #len(training_set.x)
    LossList, AccList
    criterion = MixUpLoss(device, batch_size_tr)

    training_generator = DataLoader(training_set, batch_size=batch_size_tr,
                                    shuffle=True, drop_last=True)

    for epoch in range(epochs):

        for x, y in training_generator:

            model.train()

            optimizer.zero_grad()

            x_1 = x
            x_2 = x[th.randperm(len(x))]

            lam = np.random.beta(alpha, alpha)

            x_aug = lam * x_1 + (1-lam) * x_2

            z_1, _ = model(x_1)
            z_2, _ = model(x_2)
            z_aug, _ = model(x_aug)

            loss= criterion(z_aug, z_1, z_2, lam)
            loss.backward()
            optimizer.step()
            LossList.append(loss.item())


        AccList.append(test_model(model, training_set, test_set))

        print(f"Epoch number: {epoch}")
        print(f"Loss: {LossList[-1]}")
        print(f"Accuracy: {AccList[-1]}")
        print("-"*50)

        #if epoch % 10 == 0 and epoch != 0: clear_output()

    return LossList, AccList

def test_model(model, training_set, test_set):

    model.eval()

    N_tr = len(training_set.x)
    N_te = len(test_set.x)

    training_generator = DataLoader(training_set, batch_size=1,
                                    shuffle=True, drop_last=False)
    test_generator = DataLoader(test_set, batch_size= 1,
                                    shuffle=True, drop_last=False)

    H_tr = th.zeros((N_tr, 128))
    y_tr = th.zeros((N_tr), dtype=th.long)

    H_te = th.zeros((N_te, 128))
    y_te = th.zeros((N_te), dtype=th.long)

    for idx_tr, (x_tr, y_tr_i) in enumerate(training_generator):
        with th.no_grad():
            _, H_tr_i = model(x_tr)
            H_tr[idx_tr] = H_tr_i
            y_tr[idx_tr] = y_tr_i

    H_tr = to_np(nn.functional.normalize(H_tr))
    y_tr = to_np(y_tr)


    for idx_te, (x_te, y_te_i) in enumerate(test_generator):
        with th.no_grad():
            _, H_te_i = model(x_te)
            H_te[idx_te] = H_te_i
            y_te[idx_te] = y_te_i

    H_te = to_np(nn.functional.normalize(H_te))
    y_te = to_np(y_te)

    clf = KNeighborsClassifier(n_neighbors=1).fit(H_tr, y_tr)

    return clf.score(H_te, y_te)

# Finally, training the model!!
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

np.random.seed(seed)
x_tr, y_tr = unison_shuffled_copies(x_tr, y_tr)
x_te, y_te = unison_shuffled_copies(x_te, y_te)
ntrain = 100 #len(x_tr) # set the size of partial training set to use

device = 'cuda' if th.cuda.is_available() else 'cpu'
print('running on ', device)
epochs, LossList, AccList = 3, [], []

alpha = 1.0

training_set = MyDataset(x_tr[0:ntrain,:], y_tr[0:ntrain,:])
test_set = MyDataset(x_te, y_te)

model = FCN(n_channels).to(device)

optimizer = th.optim.Adam(model.parameters())
LossListM, AccListM = train_mixup_model_epoch(model, training_set, test_set,
                                              optimizer, alpha, epochs)
th.save(model, os.path.join(basepath, alias, 'model' +  str(seed)))

print(f"Score for alpha = {alpha}: {AccListM[-1]}")


plt.figure(1, figsize=(8, 8))
plt.subplot(121)
plt.plot(LossListM)
plt.title('Loss')
plt.subplot(122)
plt.plot(AccListM)
plt.title('Accuracy')
plt.show()
plt.savefig(os.path.join(basepath, 'loss_acc_plot.png'))
