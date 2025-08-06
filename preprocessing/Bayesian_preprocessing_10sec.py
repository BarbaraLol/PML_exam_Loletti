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

import threading                                   # To menage different processing working in parallel
import queue

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
input_dataset = '../Chicks_Automatic_Detection_dataset/Registrazioni/'
output_dataset = '../Chicks_Automatic_Detection_dataset/Processed_Data_10sec/'  # New separate folder for processed data


# Change the file name and location - MODIFIED to not move .wav files
def file_name(input_dataset, output_dataset):
    """This function is now optional since we're not moving .wav files"""
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


def preparing_directories(output_dataset, subdirectory='audio_segments'):
    # Create the main output directory if it doesn't exist
    if not os.path.exists(output_dataset):
        os.makedirs(output_dataset)
        print(f"Created main output directory: {output_dataset}")
    
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


# MODIFIED: Function to process a single audio file with separate output directory
def audio_spectrograms(audio_file, output_dataset, save_dir, spectrogram_dir, checkpoint_file = "preprocessed_audios.txt"):
    # Setting the checkpoints' saving process - save checkpoint in output directory
    checkpoint_path = os.path.join(output_dataset, checkpoint_file)
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        # creation of the checkpoint file if it doesn't exist
        with open(checkpoint_path, "w") as f:
            pass
    
    if audio_file not in processed_files:

        ########## Steps ##########
        # 1. Silences removal at the beginning and at the end 
        # 2. Division of the audio in 20sec long segments
        # 3. Creation of a spectrogram for each segment

        # Load the audio file
        y, sampling_rate = librosa.load(audio_file, sr=None)

        # Resample the audio to a uniform sampling rate (e.g., 44.1 kHz)
        target_sr = 48000  # You can set this to any desired uniform rate (e.g., 44100 or 48000 Hz)
        if sampling_rate != target_sr:  # Only resample if the original sampling rate is different
            y = librosa.resample(y, orig_sr=sampling_rate, target_sr=target_sr)
            sampling_rate = target_sr
            # print(f"Resampled {audio_file} to {target_sr} Hz")

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
        # print(f"Trimmed duration for {audio_file}: {trimmed_duration}") 

        # Step n°2
        # Extract the base name of the audio file (without extension)
        base_name = os.path.splitext(os.path.basename(audio_file))[0]

        chunk_duration = 10 # Every chunck is 20sec long
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

            segments = ytrim[audio_done : (audio_done + buffer)]

            # Step n°3
            segment_spectrogram = compute_spectrogram(segments, sampling_rate)
            print(segment_spectrogram.shape)

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

            counter += 1
            audio_done += buffer # Update the position in the audio

        # Saving processed file to checkpoint
        with open(checkpoint_path, "a") as f:
            f.write(audio_file + "\n")

        # Explicitly delete variables to free memory
        del y, ytrim, segment_spectrogram, segments
        torch.cuda.empty_cache()  # Clears GPU cache to free GPU memory
        gc.collect()  # Collects garbage to free CPU memory

# MODIFIED: Plotting spectrograms with checkpoint in output directory
def spectrograms_plotting(output_dataset, save_dir, spectrogram_dir, checkpoint_file = "preprocessed_segments.txt"):
    # Setting the checkpoints' saving process - save checkpoint in output directory
    checkpoint_path = os.path.join(output_dataset, checkpoint_file)
    processed_files = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            processed_files = set(f.read().splitlines())
    else:
        # creation of the checkpoint file if it doesn't exist
        with open(checkpoint_path, "w") as f:
            pass
    
    for file in os.listdir(save_dir):
        if file not in processed_files and file.endswith('.pt'):
            # Load the saved spectrogram data
            data = torch.load(os.path.join(save_dir, file), weights_only=True)
            spectrogram = data['spectrogram']
            
            # Plot the spectrogram
            plt.figure(figsize=(10, 4))
            #plt.xlim([0, 20])  # Force the x-axis to show exactly 20 seconds
            librosa.display.specshow(spectrogram.cpu().numpy(), x_axis='time', y_axis='log', cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()

            # Save the plot as an image
            image_file_name = file.replace('.pt', '.png')
            image_file_path = os.path.join(spectrogram_dir, image_file_name)
            plt.savefig(image_file_path)
            plt.close()

            # Saving processed file to checkpoint
            with open(checkpoint_path, "a") as f:
                f.write(f"{file} with shape: {data['spectrogram'].shape}\n")

# Function to get all .wav files from input directory (including subdirectories)
def get_audio_files_from_input(input_dataset):
    """Recursively find all .wav files in the input directory"""
    audio_files = []
    for subdir, dirs, files in os.walk(input_dataset):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(subdir, file))
    return audio_files

# Function to assign and process a single audio file within a thread
def thread_audio_file_form_queue(file_queue, output_dataset, save_dir, spectrogram_dir):
    while not file_queue.empty():
        try:
            audio_file = file_queue.get_nowait() # Getting an audio file from the queue w/out waiting
        except queue.Empty:
            break # This means that all the audio files have been processed and the thread is being closed

        # Starting the processing
        audio_spectrograms(audio_file, output_dataset, save_dir, spectrogram_dir)

        # Marking the task as done when compleated
        file_queue.task_done()

# MODIFIED: Parallel process function to read from input and save to output
def parallel_audio_processing(input_dataset, output_dataset, workers):
    # Get a list of all audio files in the INPUT directory
    audio_files = get_audio_files_from_input(input_dataset)
    print(f"Found {len(audio_files)} audio files to process")

    # Preparing the directories in the OUTPUT directory
    save_dir, spectrogram_dir = preparing_directories(output_dataset)

    # Creating a queue to menage the audio files
    file_queue = queue.Queue()

    for audio_file in audio_files:
        file_queue.put(audio_file)

    # Now that all files are added to the queue, delete the list to free memory
    del audio_files  # This deletes the list of audio files to free up memory
    print("Deleted 'audio_files' array from memory") 

    # Create and menage threads
    threads = []
    for _ in range(workers):
        thread = threading.Thread(target = thread_audio_file_form_queue, args = (file_queue, output_dataset, save_dir, spectrogram_dir))
        thread.start()
        threads.append(thread)

    # Wait for all tasks to be processed
    file_queue.join() # This will block until all tasks are marked as done

    # Waiting for all the thread to finish
    for thread in threads:
        thread.join()
    
    # Now that all threads have finished processing, plot the spectrograms
    spectrograms_plotting(output_dataset, save_dir, spectrogram_dir)
    
    print("All audio files processed.")


# Main entry point to run the parallel processing
if __name__ == "__main__":
    print("Starting parallel processing with GPU enhancement and intermediate result savings")

    # Optional: uncomment if you want to reorganize .wav files first
    # file_name(input_dataset, input_dataset)  # Keep .wav files in input directory

    # Run parallel audio processing with a specified number of workers
    # The function now takes both input and output directories
    # Set the 'workers' variable according to how many processes you want to execute in parallel
    parallel_audio_processing(input_dataset, output_dataset, workers=128) # To change based on how many task you want to perform in parallel

    print("Finish processing")