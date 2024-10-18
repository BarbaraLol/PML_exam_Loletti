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
import matplotlib.pyplot as plt

import librosa                                     # To manage the audio files
import librosa.display

import os

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


# Function to save the model 
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='bnn_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'pyro_params': {name: pyro.param(name).detach().cpu() for name in pyro.get_param_store().get_all_param_names()},
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f} and accuracy {accuracy:.4f}")

# Function to load the model
def load_checkpoint(model, optimizer, filename='bnn_checkpoint.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load Pyro parameters
        pyro_param_store = pyro.get_param_store()
        for name, param in checkpoint['pyro_params'].items(): 
            pyro_param_store[name] = param

        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        accuracy = checkpoint['accuracy']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch} with loss {loss:.4f} and accuracy {accuracy:.4f}")
        return start_epoch, loss, accuracy
    else:
        print("No checkpoint found. Starting training from scratch.")
        return 0, None, None

# Bayesian Neural Network implementation
print("model definition")
class BNN(nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super (BNN, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.out = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    output = F.relu(self.fc1(x))
    output = self.out(output)
    return output

  def model(self, x_data, y_data = None):
    #Prior distributions for weightes and biases
    fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight), scale=torch.ones_like(self.fc1.weight)).to_event(self.fc1.weight.dim())
    fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias), scale=torch.ones_like(self.fc1.bias)).to_event(self.fc1.bias.dim())

    fc2w_prior = Normal(loc=torch.zeros_like(self.fc2.weight), scale=torch.ones_like(self.fc2.weight)).to_event(self.fc2.weight.dim())
    fc2b_prior = Normal(loc=torch.zeros_like(self.fc2.bias), scale=torch.ones_like(self.fc2.bias)).to_event(self.fc2.bias.dim())

    outputw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight)).to_event(self.out.weight.dim())
    outputb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)).to_event(self.out.bias.dim())

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, 'output.weight': outputw_prior, 'output.bias': outputb_prior}
    
    # Lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", self, priors)
    lifted_reg_model = lifted_module()

    with pyro.plate("data", len(x_data)):
      logits = lifted_reg_model(x_data)
      obs = pyro.sample("obs", Categorical(logits=logits), obs=y_data)

    return logits

  def guide(self, x_data, y_data = None):
    # Variable for the softplus function
    softplus = torch.nn.Softplus()
    # Weight and bias variational distribution priors
    # First layer weight distribution priors
    fc1w_mu = torch.randn_like(self.fc1.weight)
    fc1w_sigma = torch.randn_like(self.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param).to_event(1)

    # First layer bias distribution priors
    fc1b_mu = torch.randn_like(self.fc1.bias)
    fc1b_sigma = torch.randn_like(self.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param).to_event(1)

    # Second layer weight distribution priors
    fc2w_mu = torch.randn_like(self.fc2.weight)
    fc2w_sigma = torch.randn_like(self,fc2.weight)
    fc2w_mu_param = pyro.param("fc2w_mu", fc2w_mu)
    fc2w_sigma_param = softplus(pyro.param("fc2w_sigma", fc2w_sigma))
    fc2w_prior = Normal(loc=fc2w_mu_param, scale=fc2w_sigma_param).to_event(1)

    # Second layer bias distribution priors
    fc2b_mu = torch.randn_like(self.fc2.bias)
    fc2b_sigma = torch.randn_like(self,fc2.bias)
    fc2b_mu_param = pyro.param("fc2b_mu", fc2b_mu)
    fc2b_sigma_param = softplus(pyro.param("fc2b_sigma", fc2b_sigma))
    fc2b_prior = Normal(loc=fc2b_mu_param, scale=fc2b_sigma_param).to_event(1)

    # Output layer weight distribution priors
    outputw_mu = torch.randn_like(self.out.weight)
    outputw_sigma = torch.randn_like(self.out.weight)
    outputw_mu_param = pyro.param("outputw_mu", outputw_mu)
    outputw_sigma_param = softplus(pyro.param("outputw_sigma", outputw_sigma))
    outputw_prior = Normal(loc=outputw_mu_param, scale=outputw_sigma_param).to_event(1)
    
    # Output layer bias distribution priors
    outputb_mu = torch.randn_like(self.out.bias)
    outputb_sigma = torch.randn_like(self.out.bias)
    outputb_mu_param = pyro.param("outputb_mu", outputb_mu)
    outputb_sigma_param = softplus(pyro.param("outputb_sigma", outputb_sigma))
    outputb_prior = Normal(loc=outputb_mu_param, scale=outputb_sigma_param).to_event(1)

    priors = {'fc1.weight': fc1w_prior, 'fc1.bias': fc1b_prior, 'fc2.weight': fc2w_prior, 'fc2.bias': fc2b_prior, 'output.weight': outputw_prior, 'output.bias': outputb_prior}
    lifted_module = pyro.random_module("module", self, priors)

    return lifted_module()

# Dataset loading
# Define the list of file paths
file_path = './Chicks_Automatic_Detection/audio_segments/'
file_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.pt')]

# Encode labels as integers
label_encoder = LabelEncoder()
all_labels = []

# First, gather all the labels to fit the encoder
for fp in file_paths:
    data = torch.load(fp)  # Use torch.load instead of pickle.load
    _, label = data['spectrogram'], data['label']
    all_labels.append(label)

label_encoder.fit(all_labels)

class SpectrogramDataset(Dataset):
    def __init__(self, file_paths, label_encoder):
        self.file_paths = file_paths
        self.label_encoder = label_encoder
        self.scaler = StandardScaler()

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = torch.load(self.file_paths[idx])
        spectrogram, label = data['spectrogram'], data['label']
        spectrogram = spectrogram.numpy().reshape(-1, 1) # Done to flatten the spectrogram
        spectrogram = self.scaler.fit_transform(spectrogram) # To normalize the spectrogram
        spectrogram = torch.tensor(spectrogrma).reshape(128, 64) # Reshaping back after the scaling process
        # Encode the label as an integer
        encoded_label = self.label_encoder.transform([label])[0]
        return spectrogram, torch.tensor(encoded_label, dtype=torch.long)
    # Possible use of some data augmentation techniques in order to avoid overfitting

# Actually loading the dataset
dataset = SpectrogramDataset(file_paths, label_encoder)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Splittong the data into train, test and validation sets
train_size = int(0.7*len(dataset))
validation_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - validation_size
train_dataset, test_dataset, validation_dataset = random_split(dataset, [train_size, test_size, validation_size])

# Creating the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Training and validation phase
# Function to log values to a text file after each epoch
def log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy, model, filename='epoch_logs.txt'):
    with open(filename, 'a') as f:
        f.write(f"Epoch {epoch+1}:\n")
        f.write(f"Train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {avg_epoch_accuracy:.4f}\n")
        f.write(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}\n")
        f.write("Weight and Bias Distributions:\n")
        
        # Log weight and bias distributions from Pyro parameters
        pyro_param_store = pyro.get_param_store()
        for name in pyro_param_store.get_all_param_names():
            param_value = pyro_param_store[name].detach().cpu().numpy()
            f.write(f"{name}: {param_value}\n")
        f.write("\n")

# Training step
print("Starting training")

# Model instantiation
# output_size = 3 which are the different types of chicks calls
# Load a sample spectrogram to determine the input size
sample_data = torch.load(file_paths[0])
sample_spectrogram = sample_data['spectrogram']
input_size = spectrogram.numel()  # Flatten the spectrogram into a single vector (numel gives the total number of elements)
BNN_model = BNN(input_size=input_size, hidden_size=256, output_size=3)

# Define the optimizer (before loading checkpoint)
optimizer = torch.optim.Adam(BNN_model.parameters(), lr=0.01)

# SVI setup
optim = Adam({"lr": 0.01})
svi = SVI(BNN_model.model, BNN_model.guide, optim, loss=Trace_ELBO())

# Load checkpoint if available
start_epoch, _, _ = load_checkpoint(BNN_model, optimizer)

num_epoch = 20 # Maybe more(?)

# Create a logs directory if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')

# Define the log file name
log_file = os.path.join('logs', 'training_logs.txt')

def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

for epoch in range(start_epoch, num_epoch):
    BNN_model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    # Training loop
    for x_train, y_train in train_loader:
        epoch_loss += svi.step(x_train, y_train)
        predictions = BNN_model(x_train)
        epoch_accuracy += calculate_accuracy(predictions, y_train)
    
    # Averaging the loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader.dataset)
    avg_epoch_accuracy = epoch_accuracy / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")

    # Validation step
    BNN_model.eval()
    validation_loss = 0.0
    validation_accuracy = 0.0

    with torch.no_grad():
        for x_val, y_val in validation_loader:
            val_loss = svi.evaluate_loss(x_val, y_val)
            validation_loss += val_loss
            predictions = BNN_model(x_val)
            validation_accuracy += calculate_accuracy(predictions, y_val)
    
    avg_val_loss = validation_loss/len(validation_loader.dataset)
    avg_val_accuracy = validation_accuracy / len(validation_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

    # Log the values and parameter distributions for this epoch
    log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy, BNN_model, filename=log_file)

    # Save checkpoint at the end of each epoch
    save_checkpoint(BNN_model, optim, epoch, avg_epoch_loss, avg_epoch_accuracy)
    
print("Training completed")
