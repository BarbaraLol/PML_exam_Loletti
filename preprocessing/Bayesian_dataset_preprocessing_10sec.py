import numpy as np
# For reproducibility
np.random.seed(33)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Avoid GUI issues
import matplotlib.pyplot as plt

import librosa
import librosa.display

import os
import shutil
import pickle

from scipy.io import wavfile as wav
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset, DataLoader

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import gc

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 
print(f"Using device: {device}")

# Setting the input and output folders
input_dataset = '../Chicks_Automatic_Detection_dataset_orfeo/Registrazioni/'
output_dataset = '../Chicks_Automatic_Detection_dataset_orfeo/Processed_Data_5sec/'

# Create output directory if it doesn't exist
os.makedirs(output_dataset, exist_ok=True)
print(f"Created output directory: {output_dataset}")


def preparing_directories(base_dir, subdirectory='audio_segments'):
    # Create main output directory
    save_dir = os.path.join(base_dir, subdirectory)
    spectrogram_dir = os.path.join(save_dir, 'spectrograms')
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    
    print(f"Created directories: {save_dir} and {spectrogram_dir}")
    return save_dir, spectrogram_dir


def compute_spectrogram(y, sr):
    # Handle short segments
    n_fft = min(2048, len(y))
    hop_length = n_fft // 2  # Half of n_fft
    
    # Compute STFT and convert to dB scale
    spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)
    
    return torch.tensor(spectrogram, dtype=torch.float32).to(device)


def audio_spectrograms(input_dir, output_dir, chunk_duration=5):
    checkpoint_file = os.path.join(output_dir, "preprocessed_audios.txt")
    
    # Create checkpoint file if missing
    if not os.path.exists(checkpoint_file):
        open(checkpoint_file, 'a').close()
    
    processed_files = set()
    with open(checkpoint_file, "r") as f:
        processed_files = set(line.strip() for line in f)

    # Prepare output directories
    save_dir, spectrogram_dir = preparing_directories(output_dir)
    
    # Recursively find all .wav files
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    for audio_file in audio_files:
        if audio_file in processed_files:
            print(f"Skipping already processed: {audio_file}")
            continue
            
        print(f"Processing: {audio_file}")
        
        try:
            # Load audio file
            y, sampling_rate = librosa.load(audio_file, sr=None)
            
            # Resample to 48kHz if needed
            target_sr = 48000
            if sampling_rate != target_sr:
                y = librosa.resample(y, orig_sr=sampling_rate, target_sr=target_sr)
                sampling_rate = target_sr
                
            # Trim silence
            ytrim, _ = librosa.effects.trim(y, top_db=55)
            trimmed_duration = librosa.get_duration(y=ytrim, sr=sampling_rate)
            print(f"Trimmed duration: {trimmed_duration:.2f} seconds")
            
            # Prepare for segmentation
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            samples_per_chunk = int(chunk_duration * sampling_rate)
            audio_length = len(ytrim)
            audio_done = 0
            counter = 1
            
            # Process chunks
            while audio_done < audio_length:
                # Calculate current chunk size
                chunk_size = min(samples_per_chunk, audio_length - audio_done)
                segment = ytrim[audio_done:audio_done + chunk_size]
                segment_duration = librosa.get_duration(y=segment, sr=sampling_rate)
                
                # Skip segments shorter than chunk_duration
                if segment_duration < chunk_duration:
                    print(f"Skipping short segment: {segment_duration:.2f}s")
                    break
                
                # Compute spectrogram
                spectrogram = compute_spectrogram(segment, sampling_rate)
                
                # Create filename base
                segment_base = f"{base_name}_segment_{counter}"
                
                # Save spectrogram tensor
                tensor_path = os.path.join(save_dir, f"{segment_base}.pt")
                call_type = base_name.split('_')[0]  # Extract call type
                torch.save({
                    'spectrogram': spectrogram.cpu().clone(),
                    'label': call_type
                }, tensor_path)
                
                # Save spectrogram image
                image_path = os.path.join(spectrogram_dir, f"{segment_base}.png")
                plt.figure(figsize=(10, 4))
                librosa.display.specshow(spectrogram.cpu().numpy(), 
                                        sr=sampling_rate,
                                        x_axis='time', 
                                        y_axis='log', 
                                        cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram: {segment_base}')
                plt.tight_layout()
                plt.savefig(image_path)
                plt.close()
                
                print(f"Saved segment {counter} ({segment_duration:.2f}s)")
                counter += 1
                audio_done += chunk_size
                
            # Update processed files
            with open(checkpoint_file, "a") as f:
                f.write(audio_file + "\n")
                
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
        
        # Clean up memory
        del y, ytrim
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Main processing
print("Starting audio processing")
audio_spectrograms(input_dataset, output_dataset, chunk_duration=5)
print("Processing completed successfully")