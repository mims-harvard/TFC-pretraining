"""It's a KNN baseline for prompt sanity check of the dataset. -- Xiang Zhang Jan 16, 2013"""

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
import torch,os
import numpy as np


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

traindata = torch.load(os.path.join('../../datasets/Epilepsy', "test.pt"))
valdata = torch.load(os.path.join('../../datasets/Epilepsy', "val.pt"))

train_fea, train_label = traindata['samples'].detach().cpu().numpy().squeeze(1), traindata['labels'].detach().cpu().numpy()
val_fea, val_label = valdata['samples'].detach().cpu().numpy().squeeze(1), valdata['labels'].detach().cpu().numpy()


# train classifier: KNN
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(train_fea, train_label)  # .detach().cpu().numpy()
knn_acc_train = neigh.score(train_fea, train_label)  # .detach().cpu().numpy()
print('KNN finetune acc:', knn_acc_train)
# test the downstream classifier
# representation_test = emb_test.detach().cpu().numpy()
knn_result = neigh.predict(val_fea)
knn_result_score = neigh.predict_proba(val_fea)
one_hot_label_test = one_hot_encoding(val_label)
print(classification_report(val_label, knn_result, digits=4))
print(confusion_matrix(val_label, knn_result))
knn_acc = accuracy_score(val_label, knn_result)
precision = precision_score(val_label, knn_result, average='macro', )
recall = recall_score(val_label, knn_result, average='macro', )
F1 = f1_score(val_label, knn_result, average='macro')
auc = roc_auc_score(knn_result_score, one_hot_label_test, average="macro", multi_class="ovr")
prc = average_precision_score(knn_result_score, one_hot_label_test, average="macro")
print("KNN Train Acc:{}. '\n' Test: acc {}, precision {}, Recall {}, F1 {}, AUROC {}, AUPRC {}"
      "".format(knn_acc_train, knn_acc, precision, recall, F1, auc, prc))