# timeseries
Contrastive learning for self-supervised domain adaptation in time series


## CLOCS Reproducing
1. Run download_data.py
2. Run generates_Frames_and_labels_phrases.py to get labels
3. Run load_phyionet2020.py to process data
4. Run run_experiments.py for pretraining and finetuning. 
- Keep trial_to_load_list = ['CMSC']
- In pretraining, use trial_to_run_list =  ['CMSC'] 
- Fine tunine, use  trial_to_run_list =  ['Fine-Tuning']
- downstream_dataset_list = ['physionet2020'] #dataset for pretraininng # 'chapman' | 'physionet2020'
- second_dataset_list = ['chapman'] # dataset for fine-tuning
