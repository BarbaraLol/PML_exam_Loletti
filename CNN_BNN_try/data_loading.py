import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import os
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, label_encoder):
        self.label_encoder = label_encoder
        self.scaler = StandardScaler()
        
        # First filter out corrupted files
        self.valid_files = []
        for fp in file_paths:
            try:
                data = torch.load(fp)
                if isinstance(data, dict) and 'spectrogram' in data and 'label' in data:
                    self.valid_files.append(fp)
                else:
                    print(f"Skipping malformed file (missing keys): {fp}")
            except:
                print(f"Skipping corrupted file: {fp}")
        
        if not self.valid_files:
            raise ValueError("No valid .pt files found!")
        
        # Then fit the scaler on valid files
        self._fit_scaler()
    
    def _fit_scaler(self):
        """Fit scaler on all valid spectrograms."""
        all_spectrograms = []
        for fp in self.valid_files:
            data = torch.load(fp)
            spectrogram = data['spectrogram'].numpy()
            all_spectrograms.append(spectrogram)
        
        # Stack and fit scaler
        if all_spectrograms:
            all_spectrograms = np.concatenate([x.reshape(-1, 1) for x in all_spectrograms], axis=0)
            self.scaler.fit(all_spectrograms)
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        fp = self.valid_files[idx]
        data = torch.load(fp)
        spectrogram = data['spectrogram'].float()  # Ensure float type
        
        # Normalize
        original_shape = spectrogram.shape
        spectrogram = self.scaler.transform(spectrogram.numpy().reshape(-1, 1))
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).reshape(original_shape)
        
        # Encode label
        encoded_label = self.label_encoder.transform([data['label']])[0]
        return spectrogram, torch.tensor(encoded_label, dtype=torch.long)

def load_file_paths(data_dir):
    """Load all .pt files from directory."""
    file_paths = []
    for f in os.listdir(data_dir):
        if f.endswith('.pt'):
            file_paths.append(os.path.join(data_dir, f))
    print(f"Found {len(file_paths)} .pt files in {data_dir}")
    return file_paths

def encode_labels(file_paths):
    """Extract all labels from files, skipping corrupted ones."""
    all_labels = []
    corrupted_files = []
    
    for fp in file_paths:
        try:
            data = torch.load(fp)
            if isinstance(data, dict) and 'label' in data:
                all_labels.append(data['label'])
            else:
                print(f"Skipping malformed file (missing label): {fp}")
                corrupted_files.append(fp)
        except:
            print(f"Skipping corrupted file: {fp}")
            corrupted_files.append(fp)
    
    if corrupted_files:
        print(f"\nFound {len(corrupted_files)} corrupted/malformed files")
    
    return all_labels