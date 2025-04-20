for ($i = 2; $i -le 31; $i++) {
    python -m emg_hand_tracking.train -d datasets/s${i}.z -v s${i} -n
}
