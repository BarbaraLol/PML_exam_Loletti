import os
from model import HybridCNN_BNN
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from train_utils import save_checkpoint, load_checkpoint, calculate_accuracy, log_epoch_data, ensuring_log_directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Configuration
data_dir = '../audio_segments'
num_epochs = 250
batch_size = 16
learning_rate = 0.0001

# Load and process data
file_paths = load_file_paths(data_dir)
all_labels = encode_labels(file_paths)

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Create dataset
dataset = SpectrogramDataset(file_paths, label_encoder)

# Get input shape from first sample
sample_spectrogram = torch.load(file_paths[0])['spectrogram']
input_shape = sample_spectrogram.shape
num_classes = len(label_encoder.classes_)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model
model = HybridCNN_BNN(input_shape, num_classes)

# Optimizer and SVI
optimizer = Adam({"lr": learning_rate})
svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())

# Training setup
log_file = ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs')
print("Starting training")

# Early stopping
patience = 10
best_val_loss = float('inf')
epochs_without_improvement = 0

# Training loop
#for epoch in range(num_epochs):
#    model.train()
#    train_loss, train_acc = 0.0, 0.0
#    
#    # Training
#    for x_train, y_train in train_loader:
#        loss = svi.step(x_train, y_train)
#        train_loss += loss
#        
#        # Calculate accuracy
#        with torch.no_grad():
#            preds = model(x_train)
#            train_acc += calculate_accuracy(preds, y_train)

# Add this before your training loop
scaler = GradScaler()

for epoch in range(start_epoch, num_epoch):
    model.train()
    epoch_loss, epoch_accuracy = 0.0, 0.0

    # Training loop with mixed precision
    for x_train, y_train in train_loader:
        x_train, y_train = x_train.to(device), y_train.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            loss = svi.step(x_train, y_train)
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy
        with torch.no_grad():
            predictions = model(x_train)
            epoch_accuracy += calculate_accuracy(predictions, y_train)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    
    # Validation
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x_val, y_val in val_loader:
            loss = svi.evaluate_loss(x_val, y_val)
            val_loss += loss
            
            preds = model(x_val)
            val_acc += calculate_accuracy(preds, y_val)
            
            # Store predictions and labels for metrics
            _, predicted = torch.max(preds, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_val.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = val_acc / len(val_loader)
    
    # Print metrics
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
    print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0
        # Save best model
        save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_train_acc, filename='best_model.pth')
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break
    
    # Log epoch data
    log_epoch_data(epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, filename=log_file)

# Final evaluation on test set
model.eval()
test_loss, test_acc = 0.0, 0.0
all_preds = []
all_labels = []

with torch.no_grad():
    for x_test, y_test in test_loader:
        loss = svi.evaluate_loss(x_test, y_test)
        test_loss += loss
        
        preds = model(x_test)
        test_acc += calculate_accuracy(preds, y_test)
        
        _, predicted = torch.max(preds, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = test_acc / len(test_loader)

print(f"\nFinal Test Results:")
print(f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")

# Confusion Matrix and Classification Report
print("\nConfusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

print("Training completed!")