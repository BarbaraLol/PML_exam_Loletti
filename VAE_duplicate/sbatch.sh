#!/bin/bash
#
#SBATCH --job-name=VAE                 # Job name
#SBATCH --output=logs/%x_%j.out        # Stdout (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err         # Stderr
#SBATCH --partition=GPU                # GPU partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=24              # Number of CPU cores per task
#SBATCH --mem=200G                      # Total memory
#SBATCH --time=2:00:00                # Time limit (HH:MM:SS)

# --- Load necessary modules (adjust versions as needed) ---
module purge  # Clean environment
module load cuda/12.1

# --- Create & activate a fresh virtualenv ---
#python -m venv ../../venv
source ~/myenv/bin/activate
# --- Run your training script ---
# python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments --conditional --batch_size 8 --output_dir vae_results/20sec_chunks --patience 5
# python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments --batch_size 16 --output_dir vae_results/20sec_chunks_simple --patience 5
# python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Processed_Data_10sec/audio_segments --conditional --batch_size 16 --output_dir vae_results/10sec_chunks --patience 5

python train.py \
  --data_dir ../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments \
  --lr 1e-5 \
  --beta 1e-4 \
  --latent_dim 128 \
  --batch_size 8 \
  --epochs 100 \
#   --conditional  # Add this flag for conditional VAE


deactivate
