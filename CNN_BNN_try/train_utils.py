import os
import torch
import pyro
import csv
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='hybrid_cnn_bnn_checkpoint.pth'):
    """Save model checkpoint with CNN and BNN parameters."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'pyro_params': {name: pyro.param(name).detach().cpu() for name in pyro.get_param_store().get_all_param_names()},
        'loss': loss,
        'accuracy': accuracy,
        'model_type': 'HybridCNN_BNN'  # Add model type for loading compatibility
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f} and accuracy {accuracy:.4f}")

def load_checkpoint(model, optimizer, filename='hybrid_cnn_bnn_checkpoint.pth'):
    """Load model checkpoint with proper handling of CNN and BNN parameters."""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        
        # Verify model type compatibility
        if checkpoint.get('model_type') != 'HybridCNN_BNN':
            print("Warning: Loading a checkpoint that wasn't saved as HybridCNN_BNN. May cause issues.")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load Pyro parameters if they exist
        if 'pyro_params' in checkpoint:
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

def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy."""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs'):
    """Ensure log directory exists and create timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)  # Simplified directory creation
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f"{log_filename_prefix}_{timestamp}.csv"
    
    return os.path.join(log_dir, log_filename)

def log_epoch_data(epoch, train_loss, train_accuracy, val_loss, val_accuracy, filename='training_logs.csv'):
    """Log training metrics for each epoch."""
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'Epoch', 
                'Train Loss', 
                'Train Accuracy', 
                'Validation Loss', 
                'Validation Accuracy',
                'Timestamp'
            ])
        
        writer.writerow([
            epoch + 1, 
            train_loss, 
            train_accuracy, 
            val_loss, 
            val_accuracy,
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])