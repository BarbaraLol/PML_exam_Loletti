import os
import torch
import csv
from datetime import datetime

####################
# For simple VAE #
###################
# def save_checkpoint(model, optimizer, epoch, loss, accuracy=None, filename='checkpoint.pth'):
#     """Save training checkpoint"""
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
    
#     if accuracy is not None:
#         checkpoint['accuracy'] = accuracy
    
#     torch.save(checkpoint, filename)
#     print(f"Checkpoint saved: {filename}")

# def calculate_accuracy(predictions, labels):
#     """Calculate classification accuracy (for conditional VAE evaluation)"""
#     _, predicted = torch.max(predictions, 1)
#     correct = (predicted == labels).sum().item()
#     return correct / labels.size(0)

# def log_epoch_data(epoch, train_loss, train_acc, val_loss, val_acc, lr, filename):
#     """Log epoch data to CSV file"""
#     file_exists = os.path.isfile(filename)
#     with open(filename, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])
#         writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, lr])

# def log_vae_epoch_data(epoch, train_loss, train_recon, train_kl, 
#                       val_loss, val_recon, val_kl, lr, filename):
#     """Log VAE-specific epoch data to CSV file"""
#     file_exists = os.path.isfile(filename)
#     with open(filename, mode='a', newline='') as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(['Epoch', 'Train_Total', 'Train_Recon', 'Train_KL', 
#                            'Val_Total', 'Val_Recon', 'Val_KL', 'LR'])
#         writer.writerow([epoch+1, train_loss, train_recon, train_kl, 
#                         val_loss, val_recon, val_kl, lr])

#######################
# For conditional VAE #
#######################
# Updated train_utils.py - Add conditional VAE logging with accuracy

def save_checkpoint(model, optimizer, epoch, loss, accuracy=None, filename='checkpoint.pth'):
    """Save training checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    
    if accuracy is not None:
        checkpoint['accuracy'] = accuracy
    
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")

def calculate_accuracy(predictions, labels):
    """Calculate classification accuracy (for conditional VAE evaluation)"""
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def calculate_conditional_vae_accuracy(model, data, labels, device):
    """Calculate reconstruction accuracy for conditional VAE by comparing reconstructions"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        recon_x, mu, logvar = model(data, labels)
        
        # Simple reconstruction accuracy based on MSE threshold
        mse_per_sample = torch.mean((recon_x - data) ** 2, dim=[1, 2, 3])
        # Consider "accurate" if MSE is below median + std
        threshold = torch.median(mse_per_sample) + torch.std(mse_per_sample)
        accurate_reconstructions = (mse_per_sample < threshold).sum().item()
        
        return accurate_reconstructions / data.size(0)

def log_epoch_data(epoch, train_loss, train_acc, val_loss, val_acc, lr, filename):
    """Log epoch data to CSV file"""
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, lr])

def log_vae_epoch_data(epoch, train_loss, train_recon, train_kl, 
                      val_loss, val_recon, val_kl, lr, filename):
    """Log VAE-specific epoch data to CSV file"""
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'Train_Total', 'Train_Recon', 'Train_KL', 
                               'Val_Total', 'Val_Recon', 'Val_KL', 'LR'])
            writer.writerow([epoch+1, train_loss, train_recon, train_kl, 
                            val_loss, val_recon, val_kl, lr])
    except Exception as e:
        print(f"Error writing VAE log: {e}")

def log_conditional_vae_epoch_data(epoch, train_loss, train_recon, train_kl, train_acc,
                                  val_loss, val_recon, val_kl, val_acc, lr, filename):
    """Log Conditional VAE-specific epoch data with accuracy to CSV file"""
    try:
        file_exists = os.path.isfile(filename)
        with open(filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Epoch', 'Train_Total', 'Train_Recon', 'Train_KL', 'Train_Acc',
                               'Val_Total', 'Val_Recon', 'Val_KL', 'Val_Acc', 'LR'])
            writer.writerow([epoch+1, train_loss, train_recon, train_kl, train_acc,
                            val_loss, val_recon, val_kl, val_acc, lr])
    except Exception as e:
        print(f"Error writing Conditional VAE log: {e}")

def calculate_conditional_vae_accuracy(model, data, labels, device):
    """Calculate reconstruction accuracy for conditional VAE by comparing reconstructions"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        recon_x, mu, logvar = model(data, labels)
        
        # Simple reconstruction accuracy based on MSE threshold
        mse_per_sample = torch.mean((recon_x - data) ** 2, dim=[1, 2, 3])
        # Consider "accurate" if MSE is below median + std
        threshold = torch.median(mse_per_sample) + torch.std(mse_per_sample)
        accurate_reconstructions = (mse_per_sample < threshold).sum().item()
        
        return accurate_reconstructions / data.size(0)