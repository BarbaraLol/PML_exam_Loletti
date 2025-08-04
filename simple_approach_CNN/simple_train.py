# 299792458?
import os
from simple_cnn import SimpleCNN
from data_loading import SpectrogramDataset, load_file_paths, encode_labels
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, DataLoader
from train_utils import save_checkpoint, calculate_accuracy, log_epoch_data
import torch.nn as nn

def check_cuda_compatibility():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    device_cap = torch.cuda.get_device_capability()
    if device_cap[0] < 7:  # Minimum compute capability 7.0
        raise RuntimeError(f"GPU compute capability {device_cap} too low")
    
    print(f"Using device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA capability: sm_{device_cap[0]}{device_cap[1]}")
    print(f"PyTorch CUDA version: {torch.version.cuda}")

def main():
    # Configuration
    data_dir = '../Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
    num_epochs = 100
    batch_size = 32
    initial_lr = 1e-3
    grad_clip = 1.0

    # Verify CUDA compatibility
    check_cuda_compatibility()
    device = torch.device("cuda")
    
    # Load and process data
    file_paths = load_file_paths(data_dir)
    all_labels = encode_labels(file_paths)
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    dataset = SpectrogramDataset(file_paths, label_encoder)
    sample_spectrogram = torch.load(file_paths[0])['spectrogram']
    input_shape = sample_spectrogram.shape
    num_classes = len(label_encoder.classes_)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    
    # Model setup
    model = SimpleCNN(input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
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
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Train Acc: {train_acc/len(train_loader):.4f} | Val Acc: {val_acc/len(val_loader):.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, epoch, 
                          train_loss/len(train_loader), 
                          train_acc/len(train_loader),
                          'best_model.pth')

if __name__ == "__main__":
    main()