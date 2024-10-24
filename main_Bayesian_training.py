import os
from model import BNN
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from data_loading import SpectrogramDataset, load_file_path, encode_lables
from train_utils import save_checkpoint, load_checkpoint, calculate_accuracy, log_epoch_data


# First step: dataset loading by defining a list of file paths
# data_dir = './Chicks_Automatic_Detection_dataset/audio_segments/'
data_dir = './Chicks_Automatic_Detection_dataset/Registrazioni_prova/audio_segments/'
num_epoch = 100 
batch_size = 128

# Loading and processing the data
file_paths = load_file_path(data_dir)
all_labels = encode_lables (file_paths)

# Encode labels as integers
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Creating the dataset instance
dataset = SpectrogramDataset(file_paths, label_encoder)

# Splitting the data into train, test and validation sets
train_size = int(0.7*len(dataset))
validation_size = int(0.15*len(dataset))
test_size = len(dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# Creating the data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Initializing the model
input_size = torch.load(file_paths[0])['spectrogram'].numel() # It loads and flattens the spectrogram into a single vector (numel gives the total number of elements)
print("This is the input_size: ", input_size)
BNN_model = BNN(input_size=input_size, hidden_size=256, output_size=3)
# Define the optimizer (before loading checkpoint)
optimizer = torch.optim.Adam(BNN_model.parameters(), lr=0.01)
# SVI setup
svi = SVI(BNN_model.model, BNN_model.guide, Adam({"lr": 0.01}), loss=Trace_ELBO())

## Training and validation phase
# Ensure the logs directory exists and get the log file path
log_file = ensure_logs_directory(log_dir='logs', log_filename_prefix='training_logs')
printf("Starting training")

# Training step
# Load checkpoint if available
start_epoch, _, _ = load_checkpoint(BNN_model, optimizer)

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

    # Log the epoch results
    log_epoch_data(epoch, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy, filename=log_file)

    # Save checkpoint at the end of each epoch
    save_checkpoint(BNN_model, optimizer, epoch, avg_epoch_loss, avg_epoch_accuracy)
    
# Log and save
log_epoch_data(epoch, avg_train_loss, avg_train_accuracy, avg_val_loss, avg_val_accuracy)
save_checkpoint(BNN_model, optimizer, epoch, avg_train_loss, avg_train_accuracy)

print("Training completed")


