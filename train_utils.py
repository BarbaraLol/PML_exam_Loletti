import os
import torch
import pyro
import csv

# Function to save the model 
def save_checkpoint(model, optimizer, epoch, loss, accuracy, filename='bnn_checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'pyro_params': {name: pyro.param(name).detach().cpu() for name in pyro.get_param_store().get_all_param_names()},
        'loss': loss,
        'accuracy': accuracy,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch} with loss {loss:.4f} and accuracy {accuracy:.4f}")

# Function to load the model
def load_checkpoint(model, optimizer, filename='bnn_checkpoint.pth'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load Pyro parameters
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
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / labels.size(0)
    return accuracy

def ensuring_log_directory(log_dir='logs', log_filename_prefix='training_logs'):
    # Ensuring that the directory where the log files will be saved does exist
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # Get the current date and time
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    
    # Create a new log file with the timestamp
    log_filename = f"{log_filename_prefix}_{timestamp}.csv"
    
    return os.path.join(log_dir, log_file_name)

# Function to add the values to a text file after each epoch
def log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy, model, filename='training_logs.csv'):
    # Checking if the file actually exists
    file_exists = os.path.isfile(filename)        

    with open(filename, mode = 'a', newline = '') as f:
        writer = csv.writer(f)
        # If the file doesn't exist, write the header
        if not file_exists:
            writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy'])

    # Write the data for the current epoch
    writer.writerow([epoch+1, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy])

    # f.write(f"Epoch {epoch+1}:\n")
    # f.write(f"Train Loss: {avg_epoch_loss:.4f}, Train Accuracy: {avg_epoch_accuracy:.4f}\n")
    # f.write(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}\n")
    # f.write("Weight and Bias Distributions:\n")
        
    # # Log weight and bias distributions from Pyro parameters
    # pyro_param_store = pyro.get_param_store()
    # for name in pyro_param_store.get_all_param_names():
    #     param_value = pyro_param_store[name].detach().cpu().numpy()
    #     f.write(f"{name}: {param_value}\n")
    # f.write("\n")