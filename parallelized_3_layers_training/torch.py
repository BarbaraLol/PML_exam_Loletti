#!/bin/bash
#SBATCH --job-name=install_torch          # Job name
#SBATCH --partition=GPU                   # Partition with sufficient memory
#SBATCH --mem=16G                         # Memory allocation for the job
#SBATCH --time=00:30:00                   # Time allocation (adjust as necessary)
#SBATCH --output=install_torch_output.txt # Output log

# Activate your virtual environment
source ~/my_env/bin/activate

# Install a specific version of PyTorch
pip install torch==1.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
