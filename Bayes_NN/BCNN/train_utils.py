import os
import torch
import csv
from datetime import datetime

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='checkpoint.pth'):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy
    }, filename)

def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def log_epoch_data(epoch, train_loss, train_acc, val_loss, val_acc, lr, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc', 'LR'])
        writer.writerow([epoch+1, train_loss, train_acc, val_loss, val_acc, lr])