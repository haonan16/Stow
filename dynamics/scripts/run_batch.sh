#!/bin/bash

# Navigate to the directory containing the scripts
cd /home/haonan2/Projects/Stowing/dynamics/scripts/sbatch

# Get all matching files in the directory and store them in an array
scripts=($(ls run_batch_*.swb))

# Get the total number of scripts
total=${#scripts[@]}

# Initialize a counter
count=0

# While there are scripts that have not been run yet
while [ $count -lt $total ]
do
    # Get the number of currently running jobs for the user
    running_jobs=$(squeue -u $USER -h | wc -l)

    # If the number of running jobs is less than 5
    if [ $running_jobs -lt 5 ]
    then
        # Submit the next script using sbatch
        sbatch ${scripts[$count]}

        # Increment the counter
        ((count++))
    else
        # Sleep for a bit before checking again
        sleep 60
    fi
done

# Navigate back to the original directory
cd -
