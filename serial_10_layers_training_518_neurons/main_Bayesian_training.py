import os
from model import BNN
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from data_loading import SpectrogramDataset, load_file_path, encode_lables
from train_utils import save_checkpoint, load_checkpoint, calculate_accuracy, log_epoch_data, ensuring_log_directory
import seaborn as sns 
import matplotlib as plt


# First step: dataset loading by defining a list of file paths
# data_dir = './Chicks_Automatic_Detection_dataset/audio_segments/'
data_dir = '../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
num_epoch = 250 
batch_size = 64

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
#input_size = torch.load(file_paths[0])['spectrogram'].numel() # It loads and flattens the spectrogram into a single vector (numel gives the total number of elements)
sample_spectrogram = torch.load(file_paths[0])['spectrogram']
input_size = sample_spectrogram.shape[0] * sample_spectrogram.shape[1]  # 1025 * 938
#print("This is the input_size: ", input_size)
BNN_model = BNN(input_size=961450, hidden_size=518, output_size=3)

# Define the optimizer (before loading checkpoint)
optimizer = torch.optim.Adam(BNN_model.parameters(), lr=0.0001)
# Cosine Annealing scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
# SVI setup
svi = SVI(BNN_model.model, BNN_model.guide, optimizer, loss=Trace_ELBO())

## Training and validation phase
# Ensure the logs directory exists and get the log file path
log_file = ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs')
print("Starting training")

# Training step
# Load checkpoint if available
start_epoch, _, _ = load_checkpoint(BNN_model, optimizer)

# Early stopping parameters
patience = 10  # Number of epochs with no improvement after which training will stop
best_val_loss = float('inf')  # Initialize best validation loss to infinity
epochs_without_improvement = 0  # Track how many epochs have passed without improvement

for epoch in range(start_epoch, num_epoch):
    BNN_model.train()
    epoch_loss, epoch_accuracy = 0.0, 0.0

    # Training loop
    for x_train, y_train in train_loader:
        epoch_loss += svi.step(x_train, y_train)
        predictions = BNN_model(x_train)
        epoch_accuracy += calculate_accuracy(predictions, y_train)
    
    # Averaging the loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    avg_epoch_accuracy = epoch_accuracy / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epoch}, Loss: {avg_epoch_loss:.4f}, Accuracy: {avg_epoch_accuracy:.4f}")

    # Validation step
    BNN_model.eval()
    validation_loss, validation_accuracy = 0.0, 0.0

    with torch.no_grad():
        for x_val, y_val in validation_loader:
            val_loss = svi.evaluate_loss(x_val, y_val)
            scheduler.step(val_loss) # Upgrading the optimizer
            validation_loss += val_loss
            predictions = BNN_model(x_val)
            validation_accuracy += calculate_accuracy(predictions, y_val)
    
    avg_val_loss = validation_loss/len(validation_loader)
    avg_val_accuracy = validation_accuracy / len(validation_loader)
    print(f"Epoch {epoch+1}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

    # Check if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0  # Reset the counter if improvement is found
    else:
        epochs_without_improvement += 1

    # Early stopping check
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. Stopping training after {epoch + 1} epochs.")
        break

    # Log the epoch results
    log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy, filename=log_file)

    # Save checkpoint at the end of each epoch
    save_checkpoint(BNN_model, optimizer, epoch, avg_epoch_loss, avg_epoch_accuracy)
    
# Log and save
log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy)
save_checkpoint(BNN_model, optimizer, epoch, avg_epoch_loss, avg_epoch_accuracy)

# Confusion Matrix and Classification Report after training
print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report for precision, recall, f1-score
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

print("Training completed")


