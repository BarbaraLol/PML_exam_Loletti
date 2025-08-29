# Using k fold cross validation
import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms 
from blitz.losses import kl_divergence_from_nn
from model import ConditionalBayesianChickCallDetector
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import argparse
import os
import time
import numpy as np
from datetime import datetime
import csv
from collections import Counter, defaultdict

def fix_dims(batch):
    """Fix input dimensions"""
    data, target = batch
    while data.dim() > 4:
        data = data.squeeze(1)
    return data, target

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
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
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_kl_loss = kl_loss_total / len(train_loader.dataset)
    accuracy = 100. * correct / total
    
    return avg_loss, avg_kl_loss, accuracy

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            data, target = fix_dims(batch)
            data, target = data.to(device), target.to(device)
            outputs = model(data, sample=False)  # Disable sampling for validation
            loss = criterion(outputs, target)
            
            val_loss += loss.item() * data.size(0)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = val_loss / len(val_loader.dataset)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def analyze_class_distribution(all_labels, train_indices, val_indices, fold):
    """Analyze class distribution for current fold"""
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"\nFold {fold} Class Distribution:")
    print(f"Train: {Counter(train_labels)} ({len(train_labels)} samples)")
    print(f"Val: {Counter(val_labels)} ({len(val_labels)} samples)")

def k_fold_cross_validation(dataset, args):
    """Perform k-fold cross-validation"""
    
    # Extract all labels for stratified k-fold
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item())
    
    # Initialize stratified k-fold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = defaultdict(list)
    
    # Get sample shape and num classes
    sample_shape = torch.load(load_file_paths(args.data_dir)[0])['spectrogram'].shape
    num_classes = len(set(all_labels))
    
    print(f"\nStarting {k_folds}-fold cross-validation...")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample shape: {sample_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {Counter(all_labels)}")
    
    # Create output directory for k-fold results
    kfold_dir = os.path.join(args.output_dir, "k_fold_results")
    os.makedirs(kfold_dir, exist_ok=True)
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(dataset)), all_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        
        # Analyze class distribution
        analyze_class_distribution(all_labels, train_indices, val_indices, fold + 1)
        
        # Create data loaders for this fold
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # Initialize model for this fold
        model = ConditionalBayesianChickCallDetector(sample_shape, num_classes).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.7, patience=8, min_lr=1e-7
        )
        
        # Training variables
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create fold-specific log file
        fold_log_file = os.path.join(kfold_dir, f"fold_{fold+1}_log.csv")
        
        # Training loop for this fold
        start_time = time.time()
        
        # Write header
        with open(fold_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                'lr', 'kl_loss', 'time_elapsed'
            ])
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, kl_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Time tracking
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log epoch data
            with open(fold_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, train_loss, train_acc, val_loss, val_acc,
                    current_lr, kl_loss, total_time
                ])
            
            # Track best performance
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model for this fold
                torch.save({
                    'fold': fold+1,
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }, os.path.join(kfold_dir, f'fold_{fold+1}_best_model.pth'))
                
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, "
                      f"Best Val Acc={best_val_acc:.2f}%")
        
        # Store fold results
        fold_results['fold'].append(fold + 1)
        fold_results['best_val_acc'].append(best_val_acc)
        fold_results['best_val_loss'].append(best_val_loss)
        fold_results['final_train_acc'].append(train_acc)
        fold_results['training_time'].append(total_time)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Training Time: {total_time:.1f}s")
    
    # Calculate and display cross-validation results
    print(f"\n{'='*70}")
    print("K-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'='*70}")
    
    val_accs = fold_results['best_val_acc']
    val_losses = fold_results['best_val_loss']
    training_times = fold_results['training_time']
    
    print(f"Validation Accuracy per fold: {[f'{acc:.2f}%' for acc in val_accs]}")
    print(f"Mean Validation Accuracy: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
    print(f"Mean Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"95% Confidence Interval: [{np.mean(val_accs) - 1.96*np.std(val_accs)/np.sqrt(k_folds):.2f}%, "
          f"{np.mean(val_accs) + 1.96*np.std(val_accs)/np.sqrt(k_folds):.2f}%]")
    print(f"Mean Training Time per fold: {np.mean(training_times):.1f}s")
    
    # Save cross-validation summary
    summary_file = os.path.join(kfold_dir, "cv_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'best_val_acc', 'best_val_loss', 'final_train_acc', 'training_time'])
        for i in range(k_folds):
            writer.writerow([
                fold_results['fold'][i],
                fold_results['best_val_acc'][i],
                fold_results['best_val_loss'][i],
                fold_results['final_train_acc'][i],
                fold_results['training_time'][i]
            ])
        
        # Add summary statistics
        writer.writerow([])
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow(['Mean Val Acc', f"{np.mean(val_accs):.2f}%"])
        writer.writerow(['Std Val Acc', f"{np.std(val_accs):.2f}%"])
        writer.writerow(['Mean Val Loss', f"{np.mean(val_losses):.4f}"])
        writer.writerow(['Std Val Loss', f"{np.std(val_losses):.4f}"])
    
    print(f"\nDetailed results saved to: {kfold_dir}")
    return fold_results

def main():
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per fold")
    parser.add_argument('--output_dir', default='results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=15, help="Patience for early stopping")
    args = parser.parse_args()

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Data loading
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(encode_labels(file_paths))
    
    # Create dataset
    dataset = SpectrogramDataset(file_paths, label_encoder)
    print(f"Found {len(dataset)} samples")
    
    # Run k-fold cross-validation
    results = k_fold_cross_validation(dataset, args)
    
    print("\nK-fold cross-validation completed!")

if __name__ == "__main__":
    main()