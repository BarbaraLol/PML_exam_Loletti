#!pip install numpy 
#!pip install pandas 
#!pip install matplotlib
#!pip install librosa
#!pip install scipy
#!pip install sklearn 
#!pip install torch 
##!pip install torch.nn
#!pip install torch.utils
#!pip install pyro-ppl

import numpy as np
# For reproducibility
np.random.seed(33)
import pandas as pd
import matplotlib
matplotlib.use('Agg')                              # In order to avoid GUI related issues when using Matplotlib while multithreading
import matplotlib.pyplot as plt

import librosa                                     # To manage the audio files
import librosa.display

import os
import shutil                                      # For deleting the files and folders

import pickle

from scipy.io import wavfile as wav

from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.model_selection import train_test_split 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

import pyro
from pyro.distributions import Normal, Categorical
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

import gc

# Ensure GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting the input and output folders
# implementare funzione per prendere solo input e produrre cartella di output rinominandola come l'input ed eliminando la cartella in input
input_dataset = './Chicks_Automatic_Detection_dataset/Registrazioni/'
output_dataset = './Chicks_Automatic_Detection_dataset/Registrazioni_prova/'


# Change the file name and location
def file_name(input_dataset, output_dataset):
    for subdir, dirs, files in os.walk(input_dataset):
        for file in files:
            if file.endswith('.wav'):
                # Current file path
                current_file_path = os.path.join(subdir, file)
                # Extract the relevant subdirectories to form part of the new file name
                relative_subdir = os.path.relpath(subdir, input_dataset)
                relative_subdir = relative_subdir.replace(os.sep, '_')  # Replace directory separators with underscores
                
                # Form the new file name
                new_file_name = f"{relative_subdir}_{file}"
                
                # New file path in the output directory with the new name
                new_file_path = os.path.join(output_dataset, new_file_name)
                
                # Create the output directory if it doesn't exist
                os.makedirs(output_dataset, exist_ok=True)
                
                # Move and rename the file
                os.rename(current_file_path, new_file_path)
                print(f"Moved and renamed: {current_file_path} -> {new_file_path}")

    # Deleting the folder with the unnecessary stuff
    for item in os.listdir(output_dataset):
        item_path = os.path.join(output_dataset, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove the folder and its contents
            print(f"Deleted folder and its contents: {item_path}")


# Checking and creating the directories to save the spectrograms
def preparing_directories(output_dataset, subdirectory='audio_segments'):
    save_dir = os.path.join(output_dataset, subdirectory) # Where the audio segments spectrograms will be saved
    spectrogram_dir = os.path.join(save_dir, 'spectrograms') # Where the spectrograms images will be saved (inside the audio_segments folder)
    # To make sure that both directories exist
    # Ensure the save_dir exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    # Ensure the spectrogram_dir exists
    if not os.path.exists(spectrogram_dir):
        os.makedirs(spectrogram_dir)
        print(f"Created directory: {spectrogram_dir}")

    return save_dir, spectrogram_dir


def compute_spectrogram(y, sr):
    # n_fft determines the window size over which the Fourier Transform is computed
    # Adjust n_fft based on segment length
    n_fft = min(2048, len(y))
    hop_length = n_fft // 2  # Typically, hop_length is half of n_fft
    spectrogram = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    spectrogram = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max)

    return torch.tensor(spectrogram, dtype=torch.float32).to(device)


def audio_spectrograms(output_dataset, checkpoint_file = "preprocessed_audios.txt"):
    # Setting the checkpoints' saving process
    processed_files = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        # creation of the checkpoint file if it doesn't exist
        with open(checkpoint_file, "w") as f:
            pass

    # Get a list of all audio files in the directory
    audio_files = [os.path.join(output_dataset, f) for f in os.listdir(output_dataset) if f.endswith('.wav')]

    # Preparing the directories
    save_dir, spectrogram_dir = preparing_directories(output_dataset)

    ########## Steps ##########
    # 1. Silences removal at the beginning and at the end 
    # 2. Division of the audio in 20sec long segments
    # 3. Creation of a spectrogram for each segment

    #Loop through all the files
    for audio_file in audio_files:
        if audio_file not in processed_files:
            # Load the audio file
            y, sampling_rate = librosa.load(audio_file, sr=None)

            # Resample the audio to a uniform sampling rate (e.g., 44.1 kHz)
            target_sr = 48000  # You can set this to any desired uniform rate (e.g., 44100 or 48000 Hz)
            if sampling_rate != target_sr:  # Only resample if the original sampling rate is different
                y = librosa.resample(y, orig_sr=sampling_rate, target_sr=target_sr)
                sampling_rate = target_sr
                print(f"Resampled {audio_file} to {target_sr} Hz")

            # Step n°1
            # It's either I choose a dynamic trimming process...
            # rmse = librosa.feature.rms(y=y, frame_length=256, hop_length=64)[0]

            # Debug: print the RMSE values and check their range
            #print("RMSE values (first 10):", rmse[:10])
            
            #specs = librosa.power_to_db(rmse**2, ref=np.max) 

            # top_db = int(min(specs)) - 2
            # ...or a static one
            ytrim, _ = librosa.effects.trim(y, frame_length=256, hop_length=64, top_db=55)

            trimmed_duration = librosa.get_duration(y=ytrim, sr=sampling_rate)
            # print(f"Trimmed duration for{audio_file}: {trimmed_duration}") 

            # Step n°2
            # Extract the base name of the audio file (without extension)
            base_name = os.path.splitext(os.path.basename(audio_file))[0]

            chunk_duration = 20 # Every chunck is 20sec long
            # Get number of samples for 20 seconds
            buffer = int(chunk_duration * sampling_rate)

            audio_length = len(ytrim)
            audio_done = 0 # Tracks the alredy prcessed portion of the audio
            counter = 1

            while audio_done < audio_length:
                # Check if the chunck duration is actually contained inside the audio
                if buffer > (audio_length - audio_done):
                    buffer = audio_length - audio_done
                
                # Extracting the next chunck from the audio
                segments = ytrim[audio_done : (audio_done + buffer)]

                # Check the duration of the current chunk
                chunk_duration_sec = librosa.get_duration(y=segments, sr=sampling_rate)
                # print(f"Chunk duration (segment {counter}) for {audio_file}: {chunk_duration_sec:.2f} seconds")

                # Skip chunks less than 20 seconds
                if chunk_duration_sec < 20:
                    break  # Exit the loop if no more valid chunks
                    # print(f"Chunk duration (segment {counter}) blocked for duration of {chunk_duration}") 

                # Step n°3
                segment_spectrogram = compute_spectrogram(segments, sampling_rate)

                # Saving the spectrogram with its corresponding lable
                segment_spectrogram_name = f"{base_name}_segment_{counter}.pt"
                segment_spectrogram_path = os.path.join(save_dir, segment_spectrogram_name)
                # Extracting the type of call
                call_type = base_name.split('_')[0]

                # Save the normalized spectrogram and its corresponding label as a Torch tensor
                data = {
                    'spectrogram': segment_spectrogram.clone().detach(),
                    'label': call_type
                }
                torch.save(data, segment_spectrogram_path)

                # Save the spectrogram as an image (.png)
                image_file_name = f"{base_name}_segment_{counter}.png"
                image_file_path = os.path.join(spectrogram_dir, image_file_name)

                plt.figure(figsize=(10, 4))
                #plt.xlim([0, 20])  # Force the x-axis to show exactly 20 seconds
                librosa.display.specshow(segment_spectrogram.cpu().numpy(), x_axis='time', y_axis='log', cmap='viridis')
                plt.colorbar(format='%+2.0f dB')
                plt.title(f'Spectrogram of {base_name}_segment_{counter}')
                plt.tight_layout()
                plt.savefig(image_file_path)  # Save the spectrogram as an image file
                plt.close()  # Close the plot to free up memory

                counter += 1
                audio_done += buffer # Update the position in the audio

            # Saving processed file to checkpoint
            with open(checkpoint_file, "a") as f:
                f.write(audio_file + "\n")

            # Explicitly delete variables to free memory
            del y, ytrim, segment_spectrogram, segments
            torch.cuda.empty_cache()  # Clears GPU cache to free GPU memory
            gc.collect()  # Collects garbage to free CPU memory

print ("Starting processing the dataset")

#file_name(input_dataset, output_dataset) # To reorder the dataset

audio_spectrograms(output_dataset)

print("Finish processing")