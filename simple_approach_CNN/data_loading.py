import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, label_encoder):
        self.label_encoder = label_encoder
        self.file_paths = [fp for fp in file_paths if self._is_valid_file(fp)]
        self.scaler = None
        self._fit_scaler()
    
    def _is_valid_file(self, filepath):
        try:
            data = torch.load(filepath)
            return isinstance(data, dict) and 'spectrogram' in data and 'label' in data
        except:
            print(f"Skipping corrupted file: {filepath}")
            return False
    
    def _fit_scaler(self, sample_size=100):
        """Fit scaler on a small sample of data"""
        sample_files = self.file_paths[:min(sample_size, len(self.file_paths))]
        sample_data = []
        
        for fp in sample_files:
            data = torch.load(fp)
            spectrogram = data['spectrogram'].float().numpy()
            sample_data.append(spectrogram.reshape(-1, 1))
        
        if sample_data:
            self.scaler = StandardScaler()
            self.scaler.fit(np.concatenate(sample_data))
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        spectrogram = data['spectrogram'].float()
        
        if self.scaler is not None:
            original_shape = spectrogram.shape
            spectrogram = self.scaler.transform(spectrogram.numpy().reshape(-1, 1))
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).reshape(original_shape)
        
        spectrogram = torch.clamp(spectrogram, min=-5, max=5)
        encoded_label = self.label_encoder.transform([data['label']])[0]
        return spectrogram, torch.tensor(encoded_label, dtype=torch.long)

def load_file_paths(data_dir):
    """Load all .pt files from directory."""
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]

def encode_labels(file_paths):
    """Extract all labels from files."""
    labels = []
    for fp in file_paths:
        try:
            data = torch.load(fp)
            if isinstance(data, dict) and 'label' in data:
                labels.append(data['label'])
        except:
            continue
    return labels