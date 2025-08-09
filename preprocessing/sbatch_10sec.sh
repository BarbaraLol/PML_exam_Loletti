#!/bin/bash
#SBATCH --job-name=10_preprocessing       # Job name
#SBATCH --output=logs/%x_%j.out        # Stdout (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err         # Stderr
#SBATCH --partition=EPYC               # GPU partition
#SBATCH --nodes=2                      # Run on a single node
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=128            # Number of CPU cores per task
#SBATCH --mem=256G                     # Total memory
#SBATCH --time=2:00:00                 # Time limit (HH:MM:SS)


# Load Python module
module load python/3.10

source ~/myenv/bin/activate
pip install numpy pandas matplotlib librosa scipy scikit-learn pyro-ppl
echo "starting"
echo "Node: $(hostname)"
echo "Python: $(which python)"

python3 Bayesian_preprocessing_10sec.py
