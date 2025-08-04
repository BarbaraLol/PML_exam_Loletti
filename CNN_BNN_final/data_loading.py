import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from torchvision import transforms
import random

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, label_encoder, transform=None):
        self.label_encoder = label_encoder
        self.transform = transform
        self.file_paths = [fp for fp in file_paths if self._is_valid_file(fp)]
        self.scaler = self._fit_scaler()

    def _is_valid_file(self, filepath):
        try:
            data = torch.load(filepath)
            return isinstance(data, dict) and 'spectrogram' in data and 'label' in data
        except:
            print(f"Skipping corrupted file: {filepath}")
            return False

    def _fit_scaler(self, sample_size=100):
        sample_files = self.file_paths[:min(sample_size, len(self.file_paths))]
        sample_data = []
        
        for fp in sample_files:
            data = torch.load(fp)
            spectrogram = data['spectrogram'].float().numpy()
            sample_data.append(spectrogram.reshape(-1, 1))
        
        scaler = StandardScaler()
        if sample_data:
            scaler.fit(np.concatenate(sample_data))
        return scaler

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        spectrogram = data['spectrogram'].float()
        label = data['label']
        
        # Normalization
        if self.scaler is not None:
            original_shape = spectrogram.shape
            spectrogram = self.scaler.transform(spectrogram.numpy().reshape(-1, 1))
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).reshape(original_shape)
        
        # Clamp extreme values
        spectrogram = torch.clamp(spectrogram, min=-5, max=5)
        
        # Apply transforms (data augmentation)
        if self.transform:
            spectrogram = self.transform(spectrogram)
            
        # Return as [1, height, width] (3D tensor)
        return spectrogram.unsqueeze(0), torch.tensor(
            self.label_encoder.transform([label])[0],
            dtype=torch.long
        )


def load_file_paths(data_dir):
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

def encode_labels(file_paths):
    labels = []
    for fp in file_paths:
        try:
            data = torch.load(fp)
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
        except:
            continue
    return labels