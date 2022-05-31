mkdir code/baselines/TS2vec/datasets/{SleepEEG,Epilepsy,FD-A,FD-B,HAR,Gesture,ECG,ECG}
ln -s code/baselines/Mixing-up/data/SleepEEG/{train,val,test}.pt code/baselines/TS2vec/datasets/SleepEEG
ln -s code/baselines/Mixing-up/data/Epilepsy/{train,val,test}.pt code/baselines/TS2vec/datasets/Epilepsy
ln -s code/baselines/Mixing-up/data/FD-A/{train,val,test}.pt code/baselines/TS2vec/datasets/FD-A
ln -s code/baselines/Mixing-up/data/FD-B/{train,val,test}.pt code/baselines/TS2vec/datasets/FD-B
ln -s code/baselines/Mixing-up/data/HAR/{train,val,test}.pt code/baselines/TS2vec/datasets/HAR
ln -s code/baselines/Mixing-up/data/Gesture/{train,val,test}.pt code/baselines/TS2vec/datasets/Gesture
ln -s code/baselines/Mixing-up/data/ECG/{train,val,test}.pt code/baselines/TS2vec/datasets/ECG
ln -s code/baselines/Mixing-up/data/EMG/{train,val,test}.pt code/baselines/TS2vec/datasets/EMG