import os
from model import HybridCNN_BNN
from sklearn.preprocessing import LabelEncoder
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split, DataLoader
import pyro
from pyro.optim import Adam
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
data_dir = '../audio_segments'
num_epochs = 1000
batch_size = 16
learning_rate = 0.0001

def main():
    # Load and process data
    file_paths = load_file_paths(data_dir)
    all_labels = encode_labels(file_paths)

    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Create dataset
    dataset = SpectrogramDataset(file_paths, label_encoder)

    # Get input shape from first sample
    sample_spectrogram = torch.load(file_paths[0])['spectrogram']
    input_shape = sample_spectrogram.shape
    num_classes = len(label_encoder.classes_)

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
    optimizer = Adam({"lr": learning_rate})
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO())
    scaler = GradScaler()

    # Training setup
    log_file = ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs')
    print("Starting training")

    # Early stopping
    patience = 10
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        # Training phase
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                loss = svi.step(x_train, y_train)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                preds = model(x_train)
                train_acc += calculate_accuracy(preds, y_train)
            train_loss += loss

        # Validation phase
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        kl_divergence = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                
                # Calculate validation loss
                val_loss += svi.evaluate_loss(x_val, y_val)
                
                # Calculate KL divergence
                trace = poutine.trace(model.guide).get_trace(x_val)
                kl = sum(node["fn"].log_prob(node["value"]).sum() 
                       for name, node in trace.nodes.items() 
                       if "fn" in node and "value" in node)
                kl_divergence += kl.item()
                
                # Calculate accuracy
                preds = model(x_val)
                val_acc += calculate_accuracy(preds, y_val)
                
                # Store predictions and labels
                _, predicted = torch.max(preds, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(y_val.cpu().numpy())

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_kl = kl_divergence / len(val_loader)

        # Print metrics
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f} | Accuracy: {avg_train_acc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f} | Accuracy: {avg_val_acc:.4f}")
        print(f"KL Divergence: {avg_kl:.4f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_train_acc, filename='best_model.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered!")
                break

        # Log epoch data
        log_epoch_data(epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc, filename=log_file)

    # Final evaluation on test set
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            
            test_loss += svi.evaluate_loss(x_test, y_test)
            
            preds = model(x_test)
            test_acc += calculate_accuracy(preds, y_test)
            
            _, predicted = torch.max(preds, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(y_test.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_acc = test_acc / len(test_loader)

    print(f"\nFinal Test Results:")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {avg_test_acc:.4f}")

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_true, test_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', bbox_inches='tight')
    plt.close()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(test_true, test_preds, 
                              target_names=label_encoder.classes_))

    print("\nTraining completed!")

if __name__ == "__main__":
    main()