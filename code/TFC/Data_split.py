"""This file aims to generate a different data split to evaluate the stability of models.
In each split, we select 30 positive and 30 negative samples to form a training set (60 samples in total).
This is an example on Epilepsy dataset. -- Xiang Zhang, Jan 16, 2023"""

import torch
import os
import numpy as np

targetdata_path = f"../../datasets/Epilepsy/"

finetune_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
train_data = torch.load(os.path.join(targetdata_path, "train.pt"))
Samples = torch.cat((finetune_dataset['samples'], train_data['samples']), dim=0)
Labels = torch.cat((finetune_dataset['labels'], train_data['labels']), dim=0)

train_size = 30

"""Generate balanced training set"""
id0 = Labels==0
id1 = Labels==1

Samples_0, Samples_1 = Samples[id0], Samples[id1]
Labels_0, Labels_1 = Labels[id0], Labels[id1]

Samples_train = torch.cat((Samples_0[:train_size], Samples_1[:train_size]))
Samples_test = torch.cat((Samples_0[train_size:], Samples_1[train_size:]))

Labels_train = torch.cat((Labels_0[:train_size], Labels_1[:train_size]))
Labels_test = torch.cat((Labels_0[train_size:], Labels_1[train_size:]))

# """Generate imbalanced training set"""
# data = list(zip(Samples, Labels))
# np.random.shuffle(data)
# X, y = zip(*data)
# X = torch.stack(list(X), dim=0)
# y = torch.stack(list(y), dim=0)
# Samples_train, Samples_test = X[:train_size], X[train_size:]
# Labels_train, Labels_test = y[:train_size], y[train_size:]

train_dic = {'samples':Samples_train, 'labels':Labels_train}
test_dic = {'samples':Samples_test, 'labels':Labels_test}
torch.save(train_dic, os.path.join(targetdata_path,'train.pt'))
torch.save(test_dic, os.path.join(targetdata_path,'test.pt'))

print('Re-split finished. Dataset saved to folder:', targetdata_path)
