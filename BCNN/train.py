import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms 
from blitz.losses import kl_divergence_from_nn  # Fixed import
from model import BayesianChickCallDetector
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import time
import numpy as np
from datetime import datetime
import csv

def main():
    # Setup
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs")
    parser.add_argument('--output_dir', default='results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping")
    args = parser.parse_args()

    # Create output directory with timestamp
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create log file
    log_file = os.path.join(output_dir, "training_log.csv")
    
    # Data loading
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(encode_labels(file_paths))
    
    # Create datasets
    dataset = SpectrogramDataset(file_paths, label_encoder)
    sample_shape = torch.load(file_paths[0])['spectrogram'].shape
    num_classes = len(label_encoder.classes_)
    print(f"Found {len(dataset)} samples with shape {sample_shape} and {num_classes} classes")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Fix input dimensions
    def fix_dims(batch):
        data, target = batch
        while data.dim() > 4:
            data = data.squeeze(1)
        return data, target

    # Model
    print("Initializing Full Bayesian CNN model...")
    model = BayesianChickCallDetector(sample_shape, num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5) # Low LR for stability
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.7, 
        patience=8, 
        min_lr=1e-7
    )
    
    # Early stopping
    best_val_loss = float('inf') # Initialize with worst possible value
    best_val_acc = 0.0
    patience_counter = 0
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    # Log header
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
            'lr', 'kl_loss', 'time_elapsed'
        ])
    
    for epoch in range(args.epochs):

        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        kl_loss_total = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            data, target = fix_dims(batch)
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            
            # Calculate losses
            nll_loss = criterion(outputs, target)
            kl_loss = kl_divergence_from_nn(model) / len(train_loader.dataset)
            loss = nll_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += nll_loss.item() * data.size(0)
            kl_loss_total += kl_loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # if batch_idx % 20 == 0:
                #print(f"Epoch [{epoch+1}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      # f"Loss: {nll_loss.item():.4f} | KL: {kl_loss.item():.4f} | Acc: {100.*correct/total:.2f}%")
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        kl_loss_total = kl_loss_total / len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                data, target = fix_dims(batch)
                data, target = data.to(device), target.to(device)
                outputs = model(data, sample=False)  # Disabling sampling
                loss = criterion(outputs, target)
                
                val_loss += loss.item() * data.size(0)
                _, predicted = outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Time: {epoch_time:.2f}s | Total: {total_time//60:.0f}m {total_time%60:.0f}s")
        print(f"LR: {current_lr:.2e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"KL Loss: {kl_loss_total:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        # Log epoch data
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, train_acc, val_loss, val_acc,
                current_lr, kl_loss_total, total_time
            ])
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # # Save best model checkpoint
            # torch.save({
            #     'epoch': epoch+1,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'val_loss': val_loss,
            #     'val_acc': val_acc,
            #     'best_val_loss': best_val_loss,
            #     'label_encoder': label_encoder
            # }, os.path.join(output_dir, 'best_model.pth'))
            
            # print(f"Saved best model at epoch {epoch+1} with val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1} after {args.patience} epochs without improvement")
                print(f"Best validation loss: {best_val_loss:.4f} | Best accuracy: {best_val_acc:.2f}%")
                break
    
    # # Save final model
    # torch.save({
    #     'epoch': epoch+1,
    #     'model_state_dict': model.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'val_loss': val_loss,
    #     'val_acc': val_acc,
    #     'label_encoder': label_encoder
    # }, os.path.join(output_dir, 'final_model.pth'))
    
    print(f"\nTraining completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.4f} | Best accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()