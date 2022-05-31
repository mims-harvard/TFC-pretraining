import numpy as np
import sklearn
from . import _eval_protocols as eval_protocols
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
def eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='linear'):
    assert train_labels.ndim == 1 or train_labels.ndim == 2
    train_repr = model.encode(train_data, encoding_window='full_series' if train_labels.ndim == 1 else None)
    test_repr = model.encode(test_data, encoding_window='full_series' if train_labels.ndim == 1 else None)

    if eval_protocol == 'linear':
        fit_clf = eval_protocols.fit_lr
    elif eval_protocol == 'svm':
        fit_clf = eval_protocols.fit_svm
    elif eval_protocol == 'knn':
        fit_clf = eval_protocols.fit_knn
    else:
        assert False, 'unknown evaluation protocol'

    def merge_dim01(array):
        return array.reshape(array.shape[0]*array.shape[1], *array.shape[2:])

    if train_labels.ndim == 2:
        train_repr = merge_dim01(train_repr)
        train_labels = merge_dim01(train_labels)
        test_repr = merge_dim01(test_repr)
        test_labels = merge_dim01(test_labels)

    clf = fit_clf(train_repr, train_labels)

    acc = clf.score(test_repr, test_labels)
    print(train_repr.shape, train_labels.shape)
    print(test_repr.shape, test_labels.shape)
    print(test_repr)
    if eval_protocol in ['linear','knn']:
        y_score = clf.predict_proba(test_repr)
    else:
        y_score = clf.decision_function(test_repr)
#     print(train_labels.max())
#     print(train_labels)
#     test_labels_onehot = label_binarize(test_labels, classes=np.arange(train_labels.max()+1))
    test_labels_onehot = (F.one_hot(torch.tensor(test_labels).long(), num_classes=int(train_labels.max()+1))).numpy()
#     print(test_labels_onehot)
#     auprc = average_precision_score(test_labels_onehot, y_score)
#    print(y_score)
#    print(test_labels)
    metrics_dict = {}
    pred_prob = y_score
    pred = pred_prob.argmax(axis=1)
    target = test_labels
    target_prob = test_labels_onehot
    metrics_dict['Acc'] = sklearn.metrics.accuracy_score(target, pred)
    metrics_dict['Precision'] = sklearn.metrics.precision_score(target, pred, average='macro')
    metrics_dict['Recall'] = sklearn.metrics.recall_score(target, pred, average='macro')
    metrics_dict['F1'] = sklearn.metrics.f1_score(target, pred, average='macro')
    metrics_dict['AUROC'] = sklearn.metrics.roc_auc_score(target_prob, pred_prob, multi_class='ovr')
    metrics_dict['AUPRC'] = sklearn.metrics.average_precision_score(target_prob, pred_prob)
    print(metrics_dict)
    return y_score, { 'acc': acc, 'auprc': 0 }
