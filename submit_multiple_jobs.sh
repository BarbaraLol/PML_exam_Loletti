#!/bin/bash

##########
# Perform, in order:
# chmod +x Bayesian.job submit_multiple_jobs.sh
# ./submit_multiple_jobs.sh
##########
# Number of times to submit the job
NUM_JOBS=5  # replace with your desired count, i.e., x

# Path to your SLURM job script
JOB_SCRIPT="Bayesian.job"

# Loop to submit the job NUM_JOBS times, waiting 2 hours between each submission
for ((i=1; i<=NUM_JOBS; i++))
do
  echo "Submitting job $i..."
  sbatch $JOB_SCRIPT

  # Wait for 2 hours (7200 seconds) before submitting the next job, except after the last job
  if [ "$i" -lt "$NUM_JOBS" ]; then
    echo "Waiting 2 hours before the next submission..."
    sleep 7200  # 2 hours in seconds
  fi
done
