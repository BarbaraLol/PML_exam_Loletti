nvidia-smi
sudo apt-get install nvidia-cuda-toolkit -y
nvcc --version
python3 - <<EOF
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
EOF
