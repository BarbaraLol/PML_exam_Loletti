import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os
import numpy as np


class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, label_encoder):
        self.file_paths = file_paths
        self.label_encoder = label_encoder

        # Initializing the scaler and fitting it on all spectrograms contained into the trainig dataset
        all_spectrograms = []
        for y in self.file_paths:
            data = torch.load(y)
            spectrogram = data['spectrogram'].numpy().reshape(-1, 1) # Normalizing the single spectrogram
            all_spectrograms.append(spectrogram)
        
        all_spectrograms = np.concatenate(all_spectrograms, axis = 0)
        self.scaler = StandardScaler().fit(all_spectrograms) # Fitting tha scaler on the entire dataset

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        spectrogram, label = data['spectrogram'], data['label']
        spectrogram = spectrogram.numpy().reshape(-1, 1) # Done to flatten the spectrogram
        #spectrogram = torch.tensor(spectrogram).reshape(1025, 938) # Reshaping back after the scaling process
        # Flatten the spectrogram
        #spectrogram = spectrogram.view(-1)  # Flatten to a 1D vector
        # Altre modifiche
        

        # Encode the label as an integer
        encoded_label = self.label_encoder.transform([label])[0]
        return spectrogram, torch.tensor(encoded_label, dtype=torch.long)
    # Possible use of some data augmentation techniques in order to avoid overfitting

def load_file_path(data_dir):
    file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')]
    return file_paths

def encode_lables(file_paths):
    all_labels = []

    # First, gather all the labels to fit the encoder
    for fp in file_paths:
        data = torch.load(fp)  # Use torch.load instead of pickle.load
        all_labels.append(data['label'])
    
    return all_labels
