# Add these imports at the top of your simple_train.py (after your existing imports)
import os
import csv
from datetime import datetime
from simple_cnn import SimpleCNN
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from train_utils import save_checkpoint, calculate_accuracy, log_epoch_data
import torch.nn as nn
import collections
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Add these functions right after your imports in simple_train.py
def analyze_dataset_balance(data_dir):
    """Analyze class distribution in the dataset"""
    labels = []
    file_counts = {}
    
    for filename in os.listdir(data_dir):
        if filename.endswith('.pt'):
            # Extract call type from filename
            call_type = filename.split('_')[0]
            labels.append(call_type)
            file_counts[call_type] = file_counts.get(call_type, 0) + 1
    
    print("CLASS DISTRIBUTION:")
    print("=" * 40)
    total_samples = len(labels)
    
    for class_name, count in sorted(file_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"{class_name}: {count} samples ({percentage:.1f}%)")
    
    print(f"\nTotal samples: {total_samples}")
    
    # Check if dataset is imbalanced
    percentages = [count/total_samples for count in file_counts.values()]
    is_imbalanced = any(p < 0.2 or p > 0.5 for p in percentages)
    
    if is_imbalanced:
        print("\n⚠️  DATASET IS IMBALANCED!")
        print("Recommended solutions:")
        print("1. Use class weights in loss function")
        print("2. Apply data augmentation to minority classes")
        print("3. Use stratified sampling")
    else:
        print("\n✅ Dataset appears reasonably balanced")
    
    return file_counts, labels

def compute_class_weights_for_loss(labels):
    """Compute class weights for imbalanced dataset"""
    unique_labels = list(set(labels))
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    # Compute class weights
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(encoded_labels), 
        y=encoded_labels
    )
    
    weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("\nCLASS WEIGHTS (for loss function):")
    for i, label in enumerate(unique_labels):
        print(f"{label}: {weight_dict[i]:.3f}")
    
    return torch.FloatTensor(class_weights)
# Add these missing functions to your train.py file (after imports, before main())

def check_cuda_compatibility():
    """Check CUDA availability and compatibility"""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, using CPU")
        return False
    
    try:
        device_cap = torch.cuda.get_device_capability()
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: sm_{device_cap[0]}{device_cap[1]}")
        print(f"PyTorch CUDA version: {torch.version.cuda}")
        return True
    except Exception as e:
        print(f"CUDA check failed: {e}")
        return False

def setup_csv_logging(results_dir='results/10sec_chunks'):
    """Setup CSV file for logging training results."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f'training_log_{timestamp}.csv')
    
    # Create CSV with headers
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
            'learning_rate', 'early_stop_counter', 'timestamp'
        ])
    
    return csv_path

def log_to_csv(csv_path, epoch, train_loss, train_acc, val_loss, val_acc, 
               learning_rate, early_stop_counter):
    """Log epoch results to CSV file."""
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            epoch + 1, f"{train_loss:.6f}", f"{train_acc:.6f}", 
            f"{val_loss:.6f}", f"{val_acc:.6f}", f"{learning_rate:.8f}",
            early_stop_counter, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
        
        return self.early_stop
        
# REPLACE your entire main() function with this:
def main():
    # Configuration
    data_dir = '../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
    num_epochs = 100
    batch_size = 32
    initial_lr = 1e-3
    grad_clip = 1.0
    early_stopping_patience = 15  # Stop if no improvement for 15 epochs
    min_delta = 1e-4  # Minimum change to qualify as improvement

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup CSV logging
    csv_path = setup_csv_logging()
    print(f"Training results will be logged to: {csv_path}")
    
    # *** DATASET ANALYSIS SECTION ***
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Analyze dataset balance
    file_counts, all_labels = analyze_dataset_balance(data_dir)
    
    # Compute class weights
    class_weights = compute_class_weights_for_loss(all_labels)
    
    print("="*50)
    # *** END OF DATASET ANALYSIS ***
    
    # Load and process data
    file_paths = load_file_paths(data_dir)
    all_labels_original = encode_labels(file_paths)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_original)
    
    dataset = SpectrogramDataset(file_paths, label_encoder)
    sample_spectrogram = torch.load(file_paths[0])['spectrogram']
    input_shape = sample_spectrogram.shape
    num_classes = len(label_encoder.classes_)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {label_encoder.classes_}")
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    
    # Model setup
    model = SimpleCNN(input_shape, num_classes).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    # *** USE CLASS WEIGHTS IN LOSS FUNCTION ***
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience, 
        min_delta=min_delta, 
        restore_best_weights=True
    )
    
    # Setup model checkpoints directory
    checkpoint_dir = 'results/10sec_chunks/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    print(f"\nStarting training for up to {num_epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, y_train)
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                outputs = model(x_val.to(device))
                val_loss += criterion(outputs, y_val.to(device)).item()
                val_acc += calculate_accuracy(outputs, y_val.to(device))
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log to CSV
        log_to_csv(csv_path, epoch, avg_train_loss, avg_train_acc, 
                   avg_val_loss, avg_val_acc, current_lr, early_stopping.counter)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_path = os.path.join(checkpoint_dir, f'best_model_{timestamp}.pth')
            save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_train_acc, best_model_path)
            print(f"    → New best model saved! (Val Loss: {avg_val_loss:.4f}) -> {best_model_path}")
        
        # Check early stopping
        if early_stopping(avg_val_loss, model):
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            print(f"Best validation loss: {early_stopping.best_loss:.4f}")
            # Save early stopped model (with best weights restored by early stopping)
            early_stop_path = os.path.join(checkpoint_dir, f'early_stopped_model_{timestamp}.pth')
            save_checkpoint(model, optimizer, best_epoch, best_val_loss, 0, early_stop_path)
            print(f"Early stopped model saved: {early_stop_path}")
            break
        elif early_stopping.counter > 0:
            print(f"    Early stopping counter: {early_stopping.counter}/{early_stopping_patience}")
    else:
        # Training completed without early stopping - save final model
        final_model_path = os.path.join(checkpoint_dir, f'final_model_{timestamp}.pth')
        save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_train_acc, final_model_path)
        print(f"\nTraining completed! Final model saved: {final_model_path}")
    
    print("\n" + "="*70)
    print("MODEL CHECKPOINTS SUMMARY:")
    print("="*70)
    
    # List all saved models
    saved_models = []
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith(f'{timestamp}.pth'):
            saved_models.append(os.path.join(checkpoint_dir, filename))
    
    for model_path in sorted(saved_models):
        model_name = os.path.basename(model_path)
        print(f"✓ {model_name}")
    
    print(f"\nResults logged to: {csv_path}")
    print(f"Models saved to: {checkpoint_dir}/")
    
    return csv_path, checkpoint_dir

if __name__ == "__main__":
    csv_path, checkpoint_dir = main()
