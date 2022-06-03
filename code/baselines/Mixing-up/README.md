# Mixup Contrastive Learning

Code for mixup contrastive learning framework for time series. The notebook "MixupContrastiveLearningExample.ipynb" illustrates how to perform the learning and test on some simple time series datasets. Note, the notebook has since been deleted because it's unrelated to the baseline experiments in our TF-C paper. If you want to test their model, please refer to the authors' original git repo.

# Results on all univariate and multivariate datasets

The folders UcrResults and UeaResults contains the results on all datasets in the UCR and UEA databases for all methods considered. They also containts the results across all 5 folds for the learning based methods. Note, similarly, these folders have been removed from the codebase.



# Hyperparameters

The range of random seeds used for the five replicates is 0 - 4.

The following table shows hyperparameters that vary between different scenarios that produced the best results we have obtained and reported in the paper:

|                                 | n_epochs for pre-training | n_epochs for fine-tuning |
| ------------------------------- | ------------------------- | ------------------------ |
| Scenario 1: SleepEEG - Epilepsy | 3                         | 10                       |
| Scenario 2: FD-A - FD-B         | 3                         | 10                       |
| Scenario 3: HAR - Gesture       | 3                         | 20                       |
| Scenario 4: ECG - EMG           | 10                        | 5                        |

We increased # of epochs used during the one-to-many experiments, especially when fine-tuning on FD-B using the model pre-trained on SleepEDF, because the convergence is slower than the corresponding one-to-one scenario of FD-A to FD-B.

