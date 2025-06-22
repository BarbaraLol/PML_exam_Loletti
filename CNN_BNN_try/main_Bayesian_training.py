import os
from model import HybridCNN_BNN
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Remove deprecated autocast import since we'll handle CPU/GPU differently
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

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# data_dir = '../audio_segments'
data_dir = '../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
num_epochs = 1000
batch_size = 16
initial_lr = 0.001  # Start with slightly higher LR for dynamic adjustment

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
    model = HybridCNN_BNN(input_shape, num_classes).to(device)
    
    # Clear any existing Pyro parameters
    pyro.clear_param_store()
    
    # Optimizer and LR scheduler setup
    pyro_optimizer = PyroOptim(optim.Adam, {"lr": initial_lr})
    svi = SVI(model.model, model.guide, pyro_optimizer, loss=Trace_ELBO())
    
    # PyTorch optimizer for LR scheduling (wrapped around Pyro's optimizer)
    outer_optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # Fixed: Remove verbose parameter
    scheduler = ReduceLROnPlateau(
        outer_optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5
    )

    # Training setup
    log_file = ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs')
    print(f"Starting training with initial LR: {initial_lr}")

    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 15  # Longer patience to allow LR reduction to take effect

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        # Training phase
        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            # Use SVI step (which handles gradients internally)
            loss = svi.step(x_train, y_train)
            
            with torch.no_grad():
                preds = model(x_train)
                train_acc += calculate_accuracy(preds, y_train)
            train_loss += loss

        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        kl_divergence = 0.0
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                val_loss += svi.evaluate_loss(x_val, y_val)
                
                # KL divergence calculation (fixed)
                try:
                    trace = poutine.trace(model.guide).get_trace(x_val, y_val)
                    kl = 0.0
                    for name, node in trace.nodes.items():
                        if "fn" in node and "value" in node and hasattr(node["fn"], "log_prob"):
                            kl += node["fn"].log_prob(node["value"]).sum().item()
                    kl_divergence += kl
                except Exception as e:
                    print(f"Warning: Could not calculate KL divergence: {e}")
                    kl_divergence += 0.0
                
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
        avg_kl = kl_divergence / len(val_loader)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"LR: {current_lr:.2e} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")
        print(f"KL Divergence: {avg_kl:.4f}")

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

        # Log epoch data (now includes LR)
        log_epoch_data(
            epoch, 
            avg_train_loss, 
            avg_train_acc, 
            avg_val_loss, 
            avg_val_acc,
            current_lr,  # Add LR to logging
            filename=log_file
        )

    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()