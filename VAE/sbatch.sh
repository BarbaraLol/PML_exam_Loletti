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
python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments --conditional --batch_size 8 --output_dir vae_results/20sec_chunks --patience 5
# python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Processed_Data_5sec/audio_segments --batch_size 16 --output_dir vae_results/5sec_chunks --patience 5
# python3 train.py --data_dir ../Chicks_Automatic_Detection_dataset/Processed_Data_10sec/audio_segments --conditional --batch_size 16 --output_dir vae_results/10sec_chunks --patience 5

# python plot.py --model_path "./vae_results/10sec_chunks/simple_vae_experiment_20250827_102244/best_model.pth" \
#                                --data_dir "../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments" \
#                                --output_dir "class_analysis" \
#                                --max_samples 1500

deactivate
