from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import OneHotEncoder
import sklearn
import os
import torch
import torch.nn.functional as F
import numpy as np

alias = 'AHAR'
alias2 = 'AHAR'
base_dir = os.path.join('data', alias)
base_dir2 = os.path.join('data',alias2)
train_dict = torch.load(os.path.join(base_dir, 'train.pt'))
test_dict = torch.load(os.path.join(base_dir2, 'test.pt'))
train_X = train_dict['samples'][:,0,:]
train_y = train_dict['labels']
#idrawn_idx = np.random.choice(np.arange(0,), 1000, replace=False)
test_X = test_dict['samples'][:,0,:]
test_y = test_dict['labels'][:]
for i in range(1,2):
    clf = KNeighborsClassifier(n_neighbors = i)
    clf.fit(train_X, train_y)
# enc = OneHotEncoder(handle_unknown = 'ignore')
# enc.fit([[0,0],[1,1]])

    metrics_dict = {}
    pred_prob = clf.predict_proba(test_X)
    pred = pred_prob.argmax(axis=1)
    target = test_y
    target_prob = (F.one_hot(torch.tensor(test_y).long(), num_classes=8)).numpy()
    print(target_prob.shape, pred_prob.shape)
    metrics_dict['Accuracy'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
    print(metrics_dict)
