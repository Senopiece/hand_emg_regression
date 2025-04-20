param (
    [int]$start_index,
    [int]$stop_index
)

# Check if the parameters are provided
if (-not $start_index -or -not $stop_index) {
    Write-Error "Usage: script.ps1 <start_index> <stop_index>"
    exit 1
}

# Loop through the range defined by start_index and stop_index
for ($i = $start_index; $i -le $stop_index; $i++) {
    python -m emg_hand_tracking.train -d datasets/s${i}.z -v s${i} -n
}
