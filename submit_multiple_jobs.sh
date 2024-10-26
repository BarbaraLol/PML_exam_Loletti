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

# Loop to submit the job NUM_JOBS times
for ((i=1; i<=NUM_JOBS; i++))
do
  echo "Submitting job $i..."
  sbatch $JOB_SCRIPT
done
