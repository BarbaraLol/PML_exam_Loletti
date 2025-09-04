import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from blitz.losses import kl_divergence_from_nn
from model import BayesianChickCallDetector
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
import argparse
import os
import time
import numpy as np
import csv

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training") # Increased
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")  # Reduced
    parser.add_argument('--output_dir', default='results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping") # Reduced
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "training_log.csv")
    
    # Data loading
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(encode_labels(file_paths))
    
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
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)

    def fix_dims(batch):
        data, target = batch
        while data.dim() > 4:
            data = data.squeeze(1)
        return data, target

    # FIXED: Model and training setup
    print("Initializing Bayesian CNN model...")
    model = BayesianChickCallDetector(sample_shape, num_classes).to(device)
    
    # FIXED: Higher learning rate, lower weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # FIXED: More aggressive learning rate scheduling
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # Early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    
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
            outputs = model(data, sample=True)  # Enable sampling during training
            
            # FIXED: Proper KL scaling
            nll_loss = criterion(outputs, target)
            kl_loss = kl_divergence_from_nn(model) / len(train_dataset)  # Scale by dataset size
            
            # FIXED: KL weight scheduling (start low, increase gradually)
            kl_weight = min(1.0, (epoch + 1) / 10)  # Ramp up over 10 epochs
            total_loss = nll_loss + kl_weight * kl_loss
            
            total_loss.backward()
            
            # FIXED: Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            train_loss += nll_loss.item() * data.size(0)
            kl_loss_total += kl_loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_dataset)
        kl_loss_total = kl_loss_total / len(train_dataset)
        train_acc = 100. * correct / total
        
        # FIXED: Validation with sampling enabled (key for BCNN)
        model.eval()  # Still eval mode, but we'll enable sampling manually
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                data, target = fix_dims(batch)
                data, target = data.to(device), target.to(device)
                
                # FIXED: Use multiple samples for robust validation
                batch_predictions = []
                for _ in range(5):  # 5 samples per validation instance
                    outputs = model(data, sample=True)
                    batch_predictions.append(F.softmax(outputs, dim=1))
                
                # Average predictions
                avg_outputs = torch.stack(batch_predictions).mean(dim=0)
                loss = criterion(torch.log(avg_outputs + 1e-8), target)  # Convert back to logits
                
                val_loss += loss.item() * data.size(0)
                _, predicted = avg_outputs.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(val_dataset)
        val_acc = 100. * val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"Time: {epoch_time:5.1f}s | "
              f"Train: {train_acc:5.1f}% | "
              f"Val: {val_acc:5.1f}% | "
              f"LR: {current_lr:.1e} | "
              f"KL: {kl_loss_total:.4f}")
        
        # Log epoch data
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1, train_loss, train_acc, val_loss, val_acc,
                current_lr, kl_loss_total, total_time
            ])
        
        # Early stopping
        if val_acc > best_val_acc:  # Track best accuracy instead of loss
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'label_encoder': label_encoder
            }, os.path.join(args.output_dir, 'best_bcnn_model.pth'))
            
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Training time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()