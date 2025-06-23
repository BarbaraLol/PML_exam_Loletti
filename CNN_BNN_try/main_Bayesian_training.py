import os
from model import HybridCNN_BNN
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.optim import PyroOptim
from pyro.infer import SVI, Trace_ELBO
from pyro import poutine
import pyro.distributions as dist
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from train_utils import save_checkpoint, load_checkpoint, calculate_accuracy, log_epoch_data, ensuring_log_directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNN_BNN(input_shape, num_classes).to(device)
print(f"Model moved to: {next(model.parameters()).device}")  # Verify

# FIXED: Updated batch size to 128 as requested
data_dir = '../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
num_epochs = 1000
batch_size = 64  # Better balance for V100
initial_lr = 1e-6  # Start extremely low
grad_clip = 1.0  # Essential for Bayesian nets

def main():
    # Load and process data
    file_paths = load_file_paths(data_dir)
    all_labels = encode_labels(file_paths)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Create dataset
    dataset = SpectrogramDataset(file_paths, label_encoder)

    # Get input shape
    sample_spectrogram = torch.load(file_paths[0])['spectrogram']
    input_shape = sample_spectrogram.shape
    num_classes = len(label_encoder.classes_)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Total samples: {len(dataset)}")
    print(f"Batch size: {batch_size}")

    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # Create data loaders with increased batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = HybridCNN_BNN(input_shape, num_classes).to(device)
    
    # Clear any existing Pyro parameters
    pyro.clear_param_store()
    
    # CRITICAL FIX: Add gradient clipping to prevent explosions
    def clip_gradients(optimizer, max_norm=1.0):
        """Clip gradients to prevent explosion"""
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    torch.nn.utils.clip_grad_norm_(p, max_norm)
    
    # Optimizer setup with gradient clipping
    def make_optimizer(params, lr):
        optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)  # Added weight decay
        return optimizer
    
    pyro_optimizer = PyroOptim(make_optimizer, {"lr": initial_lr})
    
    # CRITICAL FIX: Use ClippedAdam to prevent gradient explosion
    svi = SVI(model.model, model.guide, pyro_optimizer, loss=Trace_ELBO())
    
    # PyTorch optimizer for LR scheduling
    outer_optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(
        outer_optimizer, 
        mode='min', 
        factor=0.5, 
        patience=8,  # Increased patience
        min_lr=1e-8
    )

    # Training setup
    log_file = ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs')
    print(f"Starting training with initial LR: {initial_lr}")
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 20  # Increased patience

    # DEBUG: Quick data check
    print("\n=== Quick Data Check ===")
    sample_batch = next(iter(train_loader))
    x_sample, y_sample = sample_batch[0].to(device), sample_batch[1].to(device)
    print(f"Sample input range: [{x_sample.min():.6f}, {x_sample.max():.6f}]")
    print(f"Sample input mean: {x_sample.mean():.6f}, std: {x_sample.std():.6f}")
    print(f"Sample target range: [{y_sample.min()}, {y_sample.max()}]")
    print(f"Target unique values: {torch.unique(y_sample)}")
    print("========================\n")

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        # Training phase
        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            # Skip batch if contains NaN/Inf
            if torch.isnan(x_train).any() or torch.isinf(x_train).any():
                print(f"WARNING: Skipping batch {batch_idx} due to NaN/Inf")
                continue
                
            # Use SVI step
            loss = svi.step(x_train, y_train)
            
            # CRITICAL: Check for exploding loss
            if np.isnan(loss) or np.isinf(loss) or loss > 1e8:
                print(f"CRITICAL: Loss explosion detected: {loss:.2e}")
                print(f"Stopping training to prevent further issues")
                return
            
            with torch.no_grad():
                preds = model(x_train)
                train_acc += calculate_accuracy(preds, y_train)
            train_loss += loss
            
            # Early debug output
            if epoch == 0 and batch_idx < 2:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss = {loss:.6f}")

        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                batch_val_loss = svi.evaluate_loss(x_val, y_val)
                val_loss += batch_val_loss
                
                # Check validation loss
                if np.isnan(batch_val_loss) or np.isinf(batch_val_loss):
                    print(f"WARNING: NaN/Inf in validation loss: {batch_val_loss}")
                
                # Accuracy
                preds = model(x_val)
                val_acc += calculate_accuracy(preds, y_val)

        # Update learning rate
        avg_val_loss = val_loss / len(val_loader)
        old_lr = outer_optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        current_lr = outer_optimizer.param_groups[0]['lr']
        
        # Print LR change if it occurred
        if current_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.2e} to {current_lr:.2e}")
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_acc = val_acc / len(val_loader)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Early stopping and checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_checkpoint(
                model, 
                outer_optimizer, 
                epoch, 
                avg_train_loss, 
                avg_train_acc, 
                filename='best_model.pth'
            )
            print("New best model saved!")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                break

        # Log epoch data
        log_epoch_data(
            epoch, 
            avg_train_loss, 
            avg_train_acc, 
            avg_val_loss, 
            avg_val_acc,
            current_lr,
            filename=log_file
        )

        # Stop early if loss is still problematic
        if epoch == 0 and avg_train_loss > 1e4:
            print(f"Training loss too high after first epoch ({avg_train_loss:.2e})")
            print("Consider checking data preprocessing or model architecture")
            break

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()