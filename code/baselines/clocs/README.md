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

# Additional notes on how to actually download data and train/fine-tune
1. Run the download_data.py script to download the Physionet2020 data, which will be used for pretraining. The same script will also download the Chapman data, which will be used for fine-tuning during the transfer learning phase. Note you have to manually edit the spreadsheet name in the chapman_ecg folder to `Diagnostics.xlsx` and then export a `.csv` file for yourself (I should probably do this in the script).  
2. For the data preparation stage, we also have to run some of the authors' scripts. For Physionet2020 first run `load_physionet2020.py` and then run `generates_frames_and_lables_phases.py`. For Chapman, just run `load_chapman_ecg.py`. I think I have modified these scripts to automatically detect our cwd to do the file manipulations. If not, please check it for yourself.
3. Finally we can train the model. For pretraining, first open the `run_experiments.py` and modify the parameters at the bottom the file. Specifically, change `trial_to_run_list` to contain just `CMSC` or `CMLC` or `CMSMLC`. The to_load list doesn't matter. For the downstream_dataset and second_dataset use both `physionet2020`. Finally, modify the `labelled_fraction` to be 1. Additionally, you can provide a list of embedding dimensions to train models of multiple sizes.
4. After pretraining, we have to modify the script to do fine-tuning. Change trial_to_run to `Fine-Tuning` and the second_dataset to `chapman`. Then lower the labelled_fraction to 0.5 or lower, otherwise you will get spuriously good results. Then run the script again and it will produce some fine-tuned models. By default it should run for about 80 epochs.

