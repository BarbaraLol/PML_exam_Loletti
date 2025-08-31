# Memory-optimized training script
import torch
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MEMORY OPTIMIZATION SETTINGS
torch.cuda.empty_cache()  # Clear cache at start
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory

print(f"Using device: {device}")
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms 
from blitz.losses import kl_divergence_from_nn
from model import HierarchicalBayesianChickCallDetector  # Use memory-optimized model
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
import gc  # Garbage collection

def fix_dims(batch):
    """Fix input dimensions"""
    data, target = batch
    while data.dim() > 4:
        data = data.squeeze(1)
    return data, target

def create_call_type_mapping(all_labels, num_call_types=3):
    """Create a mapping from fine-grained labels to broad call types."""
    unique_labels = sorted(set(all_labels))
    num_classes = len(unique_labels)
    
    # Simple strategy: divide classes into roughly equal call type groups
    labels_per_type = num_classes // num_call_types
    
    call_type_mapping = {}
    for i, label in enumerate(unique_labels):
        call_type = min(i // labels_per_type, num_call_types - 1)
        call_type_mapping[label] = call_type
    
    print(f"Call type mapping: {call_type_mapping}")
    return call_type_mapping

def get_call_type_targets(targets, call_type_mapping):
    """Convert fine-grained targets to call type targets"""
    return torch.tensor([call_type_mapping[target.item()] for target in targets], 
                       dtype=torch.long, device=targets.device)

def train_epoch(model, train_loader, optimizer, criterion, call_type_criterion, device, call_type_mapping):
    """Train for one epoch with memory optimization"""
    model.train()
    train_loss = 0.0
    call_type_loss_total = 0.0
    kl_loss_total = 0.0
    correct = 0
    total = 0
    call_type_correct = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # MEMORY OPTIMIZATION: Clear cache every 10 batches
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        data, target = fix_dims(batch)
        data, target = data.to(device), target.to(device)
        
        # Get call type targets
        call_type_targets = get_call_type_targets(target, call_type_mapping)
        
        optimizer.zero_grad()
        
        # Forward pass with gradient accumulation for large batches
        final_output, call_type_output = model(data, call_type_targets)
        
        # Calculate losses
        classification_loss = criterion(final_output, target)
        call_type_loss = call_type_criterion(call_type_output, call_type_targets)
        kl_loss = kl_divergence_from_nn(model) / len(train_loader.dataset)
        
        # Combined loss
        total_loss = classification_loss + 0.3 * call_type_loss + kl_loss
        
        total_loss.backward()
        
        # MEMORY OPTIMIZATION: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        train_loss += classification_loss.item() * data.size(0)
        call_type_loss_total += call_type_loss.item() * data.size(0)
        kl_loss_total += kl_loss.item() * data.size(0)
        
        # Final classification accuracy
        _, predicted = final_output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        # Call type classification accuracy
        _, call_type_predicted = call_type_output.max(1)
        call_type_correct += call_type_predicted.eq(call_type_targets).sum().item()
        
        # MEMORY OPTIMIZATION: Delete intermediate tensors
        del final_output, call_type_output, classification_loss, call_type_loss, kl_loss, total_loss
        
    avg_loss = train_loss / len(train_loader.dataset)
    avg_call_type_loss = call_type_loss_total / len(train_loader.dataset)
    avg_kl_loss = kl_loss_total / len(train_loader.dataset)
    accuracy = 100. * correct / total
    call_type_accuracy = 100. * call_type_correct / total
    
    return avg_loss, avg_kl_loss, accuracy, avg_call_type_loss, call_type_accuracy

def validate_epoch(model, val_loader, criterion, call_type_criterion, device, call_type_mapping):
    """Validate for one epoch with memory optimization"""
    model.eval()
    val_loss = 0.0
    call_type_loss_total = 0.0
    correct = 0
    total = 0
    call_type_correct = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # MEMORY OPTIMIZATION
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            data, target = fix_dims(batch)
            data, target = data.to(device), target.to(device)
            
            call_type_targets = get_call_type_targets(target, call_type_mapping)
            
            # Forward pass without call type targets (pure inference mode)
            final_output = model(data, call_type_targets=None)
            
            # Get call type predictions for loss calculation
            model.train()
            final_output_train, call_type_output = model(data, call_type_targets=None)
            model.eval()
            
            classification_loss = criterion(final_output, target)
            call_type_loss = call_type_criterion(call_type_output, call_type_targets)
            
            val_loss += classification_loss.item() * data.size(0)
            call_type_loss_total += call_type_loss.item() * data.size(0)
            
            _, predicted = final_output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            _, call_type_predicted = call_type_output.max(1)
            call_type_correct += call_type_predicted.eq(call_type_targets).sum().item()
    
    avg_loss = val_loss / len(val_loader.dataset)
    avg_call_type_loss = call_type_loss_total / len(val_loader.dataset)
    accuracy = 100. * correct / total
    call_type_accuracy = 100. * call_type_correct / total
    
    return avg_loss, accuracy, avg_call_type_loss, call_type_accuracy

def analyze_class_distribution(all_labels, train_indices, val_indices, fold):
    """Analyze class distribution for current fold"""
    train_labels = [all_labels[i] for i in train_indices]
    val_labels = [all_labels[i] for i in val_indices]
    
    print(f"\nFold {fold} Class Distribution:")
    print(f"Train: {Counter(train_labels)} ({len(train_labels)} samples)")
    print(f"Val: {Counter(val_labels)} ({len(val_labels)} samples)")

def k_fold_cross_validation(dataset, args):
    """Perform k-fold cross-validation with memory optimization"""
    
    # Extract all labels for stratified k-fold
    all_labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        all_labels.append(label.item())
    
    # Create call type mapping
    num_call_types = 3
    call_type_mapping = create_call_type_mapping(all_labels, num_call_types)
    
    # Initialize stratified k-fold
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    fold_results = defaultdict(list)
    
    # Get sample shape and num classes
    sample_shape = torch.load(load_file_paths(args.data_dir)[0])['spectrogram'].shape
    num_classes = len(set(all_labels))
    
    print(f"\nStarting {k_folds}-fold cross-validation with Memory-Optimized Hierarchical Bayesian CNN...")
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Sample shape: {sample_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of call types: {num_call_types}")
    print(f"Class distribution: {Counter(all_labels)}")
    
    # Create output directory for k-fold results
    kfold_dir = os.path.join(args.output_dir, "k_fold_results")
    os.makedirs(kfold_dir, exist_ok=True)
    
    for fold, (train_indices, val_indices) in enumerate(skf.split(range(len(dataset)), all_labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{k_folds}")
        print(f"{'='*60}")
        
        # MEMORY OPTIMIZATION: Clear cache at start of each fold
        torch.cuda.empty_cache()
        gc.collect()
        
        analyze_class_distribution(all_labels, train_indices, val_indices, fold + 1)
        
        # Create data loaders for this fold with SMALLER batch size
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=2,  # REDUCED from 4 to 2
            pin_memory=False  # DISABLED pin_memory to save GPU memory
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=args.batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=False
        )
        
        # Initialize memory-optimized hierarchical model
        model = HierarchicalBayesianChickCallDetector(
            sample_shape, num_classes, num_call_types=num_call_types
        ).to(device)
        
        # MEMORY OPTIMIZATION: Use smaller learning rate and weight decay
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)  # REDUCED LR
        criterion = nn.CrossEntropyLoss()
        call_type_criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=5, min_lr=1e-7  # MORE AGGRESSIVE
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
                'lr', 'kl_loss', 'call_type_train_loss', 'call_type_train_acc',
                'call_type_val_loss', 'call_type_val_acc', 'time_elapsed'
            ])
        
        for epoch in range(args.epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, kl_loss, train_acc, call_type_train_loss, call_type_train_acc = train_epoch(
                model, train_loader, optimizer, criterion, call_type_criterion, device, call_type_mapping
            )
            val_loss, val_acc, call_type_val_loss, call_type_val_acc = validate_epoch(
                model, val_loader, criterion, call_type_criterion, device, call_type_mapping
            )
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            
            # Log epoch data
            with open(fold_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch+1, train_loss, train_acc, val_loss, val_acc,
                    current_lr, kl_loss, call_type_train_loss, call_type_train_acc,
                    call_type_val_loss, call_type_val_acc, total_time
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
                    'call_type_mapping': call_type_mapping,
                    'train_indices': train_indices,
                    'val_indices': val_indices
                }, os.path.join(kfold_dir, f'fold_{fold+1}_best_model.pth'))
                
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print progress every 5 epochs (reduced frequency)
            if epoch % 5 == 0:
                print(f"Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%, "
                      f"Best Val Acc={best_val_acc:.2f}%")
                print(f"  Call Type - Train Acc={call_type_train_acc:.2f}%, Val Acc={call_type_val_acc:.2f}%")
        
        # Store fold results
        fold_results['fold'].append(fold + 1)
        fold_results['best_val_acc'].append(best_val_acc)
        fold_results['best_val_loss'].append(best_val_loss)
        fold_results['final_train_acc'].append(train_acc)
        fold_results['final_call_type_acc'].append(call_type_val_acc)
        fold_results['training_time'].append(total_time)
        
        print(f"\nFold {fold + 1} Results:")
        print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
        print(f"  Best Val Loss: {best_val_loss:.4f}")
        print(f"  Final Call Type Acc: {call_type_val_acc:.2f}%")
        print(f"  Training Time: {total_time:.1f}s")
        
        # MEMORY OPTIMIZATION: Clean up after each fold
        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()
        gc.collect()
    
    # Calculate and display results (same as before)
    print(f"\n{'='*70}")
    print("K-FOLD CROSS-VALIDATION RESULTS - MEMORY-OPTIMIZED HIERARCHICAL BAYESIAN CNN")
    print(f"{'='*70}")
    
    val_accs = fold_results['best_val_acc']
    val_losses = fold_results['best_val_loss']
    call_type_accs = fold_results['final_call_type_acc']
    training_times = fold_results['training_time']
    
    print(f"Final Classification Accuracy per fold: {[f'{acc:.2f}%' for acc in val_accs]}")
    print(f"Call Type Classification Accuracy per fold: {[f'{acc:.2f}%' for acc in call_type_accs]}")
    print(f"Mean Final Classification Accuracy: {np.mean(val_accs):.2f}% ± {np.std(val_accs):.2f}%")
    print(f"Mean Call Type Classification Accuracy: {np.mean(call_type_accs):.2f}% ± {np.std(call_type_accs):.2f}%")
    print(f"Mean Validation Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"95% Confidence Interval: [{np.mean(val_accs) - 1.96*np.std(val_accs)/np.sqrt(k_folds):.2f}%, "
          f"{np.mean(val_accs) + 1.96*np.std(val_accs)/np.sqrt(k_folds):.2f}%]")
    print(f"Mean Training Time per fold: {np.mean(training_times):.1f}s")
    
    # Save cross-validation summary
    summary_file = os.path.join(kfold_dir, "cv_summary.csv")
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold', 'best_val_acc', 'best_val_loss', 'final_train_acc', 
                        'final_call_type_acc', 'training_time'])
        for i in range(k_folds):
            writer.writerow([
                fold_results['fold'][i],
                fold_results['best_val_acc'][i],
                fold_results['best_val_loss'][i],
                fold_results['final_train_acc'][i],
                fold_results['final_call_type_acc'][i],
                fold_results['training_time'][i]
            ])
        
        writer.writerow([])
        writer.writerow(['SUMMARY STATISTICS'])
        writer.writerow(['Mean Val Acc', f"{np.mean(val_accs):.2f}%"])
        writer.writerow(['Std Val Acc', f"{np.std(val_accs):.2f}%"])
        writer.writerow(['Mean Call Type Acc', f"{np.mean(call_type_accs):.2f}%"])
        writer.writerow(['Std Call Type Acc', f"{np.std(call_type_accs):.2f}%"])
        writer.writerow(['Mean Val Loss', f"{np.mean(val_losses):.4f}"])
        writer.writerow(['Std Val Loss', f"{np.std(val_losses):.4f}"])
    
    print(f"\nDetailed results saved to: {kfold_dir}")
    return fold_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True, help="Path to spectrogram directory")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (REDUCED)")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per fold")
    parser.add_argument('--output_dir', default='results', help="Directory to save outputs")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping (REDUCED)")
    args = parser.parse_args()

    # Print memory info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Available GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print("Loading data...")
    file_paths = load_file_paths(args.data_dir)
    label_encoder = LabelEncoder()
    label_encoder.fit(encode_labels(file_paths))
    
    dataset = SpectrogramDataset(file_paths, label_encoder)
    print(f"Found {len(dataset)} samples")
    
    results = k_fold_cross_validation(dataset, args)
    
    print("\nK-fold cross-validation with Memory-Optimized Hierarchical Bayesian CNN completed!")

if __name__ == "__main__":
    main()