#!/bin/bash
#
#SBATCH --job-name=BayesianTrain       # Job name
#SBATCH --output=logs/%x_%j.out        # Stdout (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err         # Stderr
#SBATCH --partition=GPU                # GPU partition
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=12             # Number of CPU cores per task
#SBATCH --mem=32G                      # Total memory
#SBATCH --time=0:30:00                # Time limit (HH:MM:SS)

# --- Load necessary modules (adjust versions as needed) ---
module purge  # Clean environment
module load cuda/12.1
module load cudnn/8.9.7  # Add cuDNN

# --- Create & activate a fresh virtualenv ---
#python -m venv ../../venv
source ~/myenv/bin/activate

# --- Upgrade pip and install dependencies ---
pip install --upgrade pip
pip install numpy pandas matplotlib librosa scipy scikit-learn pyro-ppl
