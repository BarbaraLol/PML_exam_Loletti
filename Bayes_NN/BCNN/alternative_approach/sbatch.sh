#!/bin/bash
#
#SBATCH --job-name=BCNN                # Job name
#SBATCH --output=logs/%x_%j.out        # Stdout (%x=job-name, %j=job-id)
#SBATCH --error=logs/%x_%j.err         # Stderr
#SBATCH --partition=GPU                # GPU partition
#SBATCH --gres=gpu:1
#SBATCH --nodes=1                      # Run on a single node
#SBATCH --ntasks-per-node=1            # One task per node
#SBATCH --cpus-per-task=24              # Number of CPU cores per task
#SBATCH --mem=128G                      # Total memory
#SBATCH --time=2:00:00                # Time limit (HH:MM:SS)

# --- Load necessary modules (adjust versions as needed) ---
module purge  # Clean environment
module load cuda/12.1
module load cudnn/8.9.7  # Add cuDNN


# --- (Optional) make sure logs directory exists ---
mkdir -p logs

# --- Create & activate a fresh virtualenv ---
#python -m venv ../../venv
source ~/myenv/bin/activate

# --- Upgrade pip and install dependencies ---
#pip install --upgrade pip
#pip install torch torchaudio torchvision pyro-ppl
#pip install -r ../../requirements.txt
# --- Verify GPU ---
echo "===== GPU INFORMATION ====="
nvidia-smi
echo "===== PYTHON GPU CHECK ====="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); \
           print(f'CUDA available: {torch.cuda.is_available()}'); \
           if torch.cuda.is_available(): print(f'GPU: {torch.cuda.get_device_name(0)}')"

# --- Run your training script ---
python3 train.py --data_dir ../../../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/ --batch_size 16 --output_dir results/20sec_chunks
# python3 train.py --data_dir ../../Chicks_Automatic_Detection_dataset/Processed_Data_5sec/audio_segments/ --batch_size 16 --output_dir results/5sec_chunks
# python3 train.py --data_dir ../../../Chicks_Automatic_Detection_dataset/Processed_Data_10sec/audio_segments/ --batch_size 16 --output_dir results/10sec_chunks


