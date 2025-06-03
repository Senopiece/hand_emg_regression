TODO: how to setup, emg2pose dependency, emg2pose full dataset requirements, repacked dataset, how to run train

python -m emg_hand_tracking.train -d ../datasets/0.zip -v c0 -n --val_usage 1 --train_sample_ratio 0.4 --val_window 9000 --batch_size 256 --frames_per_item 256