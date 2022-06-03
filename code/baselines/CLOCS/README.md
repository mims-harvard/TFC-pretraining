# CLOCS

CLOCS is a patient-specific contrastive learning method that can be used to pre-train medical time-series data. It can improve the generalization performance of downstream supervised tasks.

This method is described in "CLOCS: Contrastive Learning of Cardiac Signals"

# Requirements

The CLOCS code requires

* Python 3.6 or higher
* PyTorch 1.0 or higher

(The authors left out describing what other packages are required... So I provide a working conda `environment.yml` file for easy setup)

# Datasets

## Download

The datasets can be downloaded from the following links:

1) PhysioNet 2020: https://physionetchallenges.github.io/2020/
2) Chapman: https://figshare.com/collections/ChapmanECG/4560497/2
3) Cardiology: https://irhythm.github.io/cardiol_test_set/
4) PhysioNet 2017: https://physionet.org/content/challenge-2017/1.0.0/

## Pre-processing

In order to pre-process the datasets appropriately for CLOCS and the downstream supervised tasks, please refer to the following repository: https://anonymous.4open.science/r/9ecc66f3-e173-4771-90ce-ff35ee29a1c0/

# Training

To train the model(s) in the paper, run this command:

```
python run_experiments.py
```

# Evaluation

To evaluate the model(s) in the paper, run this command:

```
python run_experiments.py
```



# Hyperparameters

Range of random seeds used for the five replicates are 0 - 4.

Pre-training batch size is set to 256 and fine-tuning batch size is 4.

The following table shows hyperparameters that vary between different scenarios that produced the best results we have obtained and reported in the paper:

|          | K1-k3, s1-s3 (for conv layers) | Last 1D max-pooling layer size | Input dimension of view_linear_modules | n_epochs |
| -------- | ------------------------------ | ------------------------------ | -------------------------------------- | -------- |
| SleepEEG | 733, 221                       | 2                              | 32                                     | 5        |
| Epilepsy | 733, 221                       | 8                              | 32                                     | 20       |
| FD-A     | 777, 444                       | 2                              | 128                                    | 20       |
| FD-B     | 777, 444                       | 4                              | 128                                    | 50       |
| HAR      | 444, 221                       | 2                              | 32                                     | 10       |
| Gesture  | 444, 221                       | 6                              | 32                                     | 50       |
| ECG      | 777, 333                       | 2                              | 64                                     | 5        |
| EMG      | 777, 333                       | 6                              | 64                                     | 20       |

