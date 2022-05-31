wget -O SleepEEG.zip https://figshare.com/ndownloader/articles/19930178/versions/1
wget -O Epilepsy.zip https://figshare.com/ndownloader/articles/19930199/versions/1
wget -O FD-A.zip https://figshare.com/ndownloader/articles/19930205/versions/1
wget -O FD-B.zip https://figshare.com/ndownloader/articles/19930226/versions/1
wget -O HAR.zip https://figshare.com/ndownloader/articles/19930244/versions/1
wget -O Gesture.zip https://figshare.com/ndownloader/articles/19930247/versions/1
wget -O ECG.zip https://figshare.com/ndownloader/articles/19930253/versions/1
wget -O EMG.zip https://figshare.com/ndownloader/articles/19930250/versions/1

tar -xf SleepEEG.zip -C datasets/SleepEEG/
tar -xf Epilepsy.zip -C datasets/Epilepsy/
tar -xf FD-A.zip -C datasets/FD-A/
tar -xf FD-B.zip -C datasets/FD-B/
tar -xf HAR.zip -C datasets/HAR/
tar -xf Gesture.zip -C datasets/Gesture/
tar -xf ECG.zip -C datasets/ECG/
tar -xf EMG.zip -C datasets/EMG/

rm {SleepEEG,Epilepsy,FD-A,FD-B,HAR,Gesture,ECG,EMG}.zip 




