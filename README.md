## Main Usage Scenario

This repo is for training a semg-handpose model

> NOTE: using python 3.10

### 1. Emg2Pose dependency

This repository is suited to be evaluated both on proprietary dataset and on emg2pose dataset as for reference.

Hence follow the instructions of [emg2pose](https://github.com/facebookresearch/emg2pose) to install

From here you are interested to install it with this commands into your preferred env (pyenv/venv/conda/etc...)
```bash
# Install the emg2pose package
pip install -e .

# Install the UmeTrack package (for forward kinematics and mesh skinning)
pip install -e emg2pose/UmeTrack
```

### 2. Emg2Pose full dataset

If you wish to train on emg2pose dataset

1. Follow the instruction on [emg2pose](https://github.com/facebookresearch/emg2pose) and extract the full dataset somewhere
2. Run [notebooks/convert_dataset.ipynb] with selected sessions range and provided dataset path to convert dataset to the format acceptable by train script
3. As the result you should get a bunch of zip files each representing a recording session
4. Run tran (see the section below)

### 3. Custom dataset

If you wish to make a custom dataset, or use the one made along with this study

1. Inspect [dataset capture](https://github.com/Senopiece/emg_hand_pose_dataset_capture) repository
2. Follow it's instructions to make your own hardware and capture your own dataset
3. Or just use prerecorded ones provided in the [readme](https://github.com/Senopiece/emg_hand_pose_dataset_capture/blob/main/README.md)
4. As the result you should get a bunch of zip files each representing a recording session

### 4. Train

Training process is straightforward, having simple requirements of torch, cuda (highly recommended for acceleration). Install with [requirements.txt](requirements.txt).

A example using the training script:

```bash
$ python -m emg_hand_tracking.train -d ../datasets/0.zip -v c0 -n --val_usage 1 --val_window 1000 --train_frames_per_item 320 --val_frames_per_item 320 --epoch_time_limit 10000
```

- That will utilize wandb for logging, so make sure to login first
- Also check [sweep.yaml](sweep.yaml) to make hyper-parameter search