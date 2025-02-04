# PML_exam_Loletti
- Bayesian_dataset_preprocessing.py: is the simplest version run on Orfeo and used to produce the dataset used by the BNN
- Bayesian_dataset_preprocessing_parallelized.py: Parallelized version of the Bayesian_dataset_preprocessing.py in which the number of processes can be set based on the particular features of the machine that will excute the code. Moreover the preprocessing works by deviding the whole starting dataset into small batches which size can be set manually. When the image of the spectrogram is saved, the GPU is involved in order to reduce the workload form the CPU and prevent any kind of possible bottleneck. A simple system is also implemented in order to save intermediate results so to have restart checkpoints







Implementing history - evolution of the code
1. Preprocessing
2. Parallelized preprocessing
3. BNN - training part
4. Splitting the BNN code for better maintenance
5. Problem resolutions before first real sbatch on Orfeo (dimention.py)
6. Results for Adam({"lr": 0.01}) quite poor so tried to change it to Adam({"lr": 0.001})
7. Results still quite poor so started again from scratch the training using Adam({"lr": 0.0001}) but the accuracy is poor and reducing at some point 
8. Trying with a 3 layers BNN, with 512 neurons, Adam({"lr": 0.0001}) 

Possible features to implement/add to the code
- Data augmentation 
- Eaerly stopping and learning rate schedulers
- Adding gradient clipping


# To do 
1. Potential model implemetations
    - Model pruning: pruning approach could be improved by experimenting with different sparsity methods (e.g., L1 norm regularization) to reduce model complexity without losing predictive power​
    - Data Augmentation: exploring augmentation techniques (such as noise addition or pitch shifting) might improve model robustness, especially given the potential for overfitting in smaller datasets.
2. Hyperparameter tuning
    - About the learning rate:
        - Learning Rate Scheduling: Gradually decrease the learning rate as the model trains. Techniques like ReduceLROnPlateau or Cosine Annealing can be useful.
        - Learning Rate Search: Use a grid search or random search to explore different learning rates, potentially trying values like 0.001, 0.0005, 0.0003.
    - Batch size:
        - Smaller batch size (e.g., 64 or 32) might help with a more stable gradient and improve generalization (64 WAS CHOSEN)
        - Larger batch size can speed up training but may require tuning of the learning rate to prevent divergence.
    - Number of epochs:
        - close monitoration of the validation loss to detect if it's overfitting (i.e., when the validation loss stops improving while training loss decreases) 
        - early stopping based on validation loss to prevent overfitting.
    - Model architecture:
        - Hidden layers
            - More hidden layers (8/10) in case the model is underfitting (UPGRATED TO 10 LAYERS)
            - different units per layer (from 256 to 516 or 128) to see how the performance is affected (DONE)
        - Dropout: Although the Bayesian approach used, which should inherently reduce overfitting, adding dropout to the fully connected layers could still help. Experiment with dropout rates between 0.2 and 0.5
    - Regularization: using L2 Regularization (Weight Decay) can help prevent overfitting by adding a penalty to the weights. Used by including weight_decay in the Adam optimizer, which can help your model generalize better. The typical values for weight decay range from 0.0001 to 0.01.
    - Model parameters: Since the usage of variational inference, the number of samples (T) for estimating the posterior is an important hyperparameter: experiment with different values for T (e.g., 10, 20, 50) to see if more samples help improve performance.
    - Data augmentation: data transformation by:  
        a. Time masking or frequency masking on the spectrograms to simulate real-world variations in sound.  
        b. Add random noise or slightly modify the pitch of the audio to make the model more robust.
    - Model evaluation: using  
        a. confusion matrix: After training, evaluate your model using a confusion matrix to see if it's misclassifying any specific classes. This can help you identify if certain classes need more training data or better feature representations.  
        b. cross validation: might help to ensure that your model is generalizing well across different subsets of the data
    - Optimizers: try and experiment with other optimizers such as:
        - SGD with momentum: While it can be slower to converge, it sometimes helps with generalization in deep networks
        - RMSprop: This optimizer works well when you have noisy gradients or highly variant data.
    - Hyperparameters search methods
        - Grid Search: Systematically search through a hyperparameter space.
        - Random Search: Randomly sample hyperparameters to find the best combination.
        - Bayesian Optimization: Uses probabilistic models to explore the hyperparameter space more efficiently.

# Possibilities for learning rate
## Learning rate scheduling
### ReduceLROnPlateau
It's a scheduler that adjusts the learning rate when the model’s validation loss plateaus. It reduces the learning rate by a factor (like 0.1) if the validation loss doesn't improve after a set number of epochs.

```
import torch
import torch.optim as optim

# Defining the optimizer
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Defining the function to optimize the learning rate
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop
for epoch in range(num_epoch):
    model.train()
    
    # Training loop goes here (compute loss, optimizer step, etc.)
    
    # After each epoch, step the scheduler based on the validation loss
    val_loss = compute_validation_loss()  # Replace with your actual validation loss function
    scheduler.step(val_loss)
```
What the parameters in the ReduceLROnPlateau do:
- factor: redices the learning rate by a factor of 0.1 when there's no improvement in validation loss.
- patience: if the validation's loss isn't improved after 5 epochs, the learning rate is reduced
- verbose=True: prints a message whenever the learning rate is reduced.

# Cosine Annealing
It decreases the learning rate following a cosine curve, which can help with smoother convergence.
```
import torch
import torch.optim as optim

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Implement Cosine Annealing scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

# Training loop (simplified)
for epoch in range(num_epochs):
    model.train()
    
    # Training loop goes here (compute loss, optimizer step, etc.)
    
    # Step the scheduler at the end of each epoch
    scheduler.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {optimizer.param_groups[0]['lr']}")
```
What the parameters in the CosineAnnealingLR do:
- T_max=50: the learning rate will decrease to the minimum over 50 epochs, after which the cycle repeats
- eta_min=0: the minimum learning rate that it can decay to

## Learning rate search
# Grid search
A a grid of values for the learning rate is specified and the model is systematically trained with each one
```
import torch.optim as optim
import torch.nn as nn

# Define the model and loss function
model = YourModel()
criterion = nn.CrossEntropyLoss()

# Define possible learning rates for grid search
learning_rates = [0.001, 0.0005, 0.0003]

# Loop through different learning rates
for lr in learning_rates:
    print(f"Training with learning rate: {lr}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop (simplified)
    for epoch in range(num_epochs):
        model.train()
        # Perform your training steps here...
        
    # Optionally, validate your model here and keep track of the best learning rate.
```
# Random search
We randomly sample values from a range
```
import torch.optim as optim
import random

# Define the model and loss function
model = YourModel()
criterion = nn.CrossEntropyLoss()

# Define random search range for learning rates
min_lr = 0.0001
max_lr = 0.01
num_trials = 10  # Number of trials to run

# Random search loop
for trial in range(num_trials):
    lr = random.uniform(min_lr, max_lr)
    print(f"Trial {trial+1}: Training with learning rate: {lr}")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop (simplified)
    for epoch in range(num_epochs):
        model.train()
        # Perform your training steps here...
        
    # Optionally, validate your model here and keep track of the best learning rate.
```

# Early stop implemetnation to prevent underfitting or overfitting

```
# Importing stuff
# Load checkpoint if available
start_epoch, _, _ = load_checkpoint(BNN_model, optimizer)

# Early stopping parameters
patience = 10  # Number of epochs with no improvement after which training will stop
best_val_loss = float('inf')  # Initialize best validation loss to infinity
epochs_without_improvement = 0  # Track how many epochs have passed without improvement

for epoch in range(num_epch):
    # trainig and validation step
    
    # Check if validation loss improved
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0  # Reset the counter if improvement is found
    else:
        epochs_without_improvement += 1

    # Early stopping check
    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered. Stopping training after {epoch + 1} epochs.")
        break
    # Log the epoch results
    log_epoch_data(epoch, avg_epoch_loss, avg_epoch_accuracy, avg_val_loss, avg_val_accuracy, filename=log_file)

    # Save checkpoint at the end of each epoch
    save_checkpoint(BNN_model, optimizer, epoch, avg_epoch_loss, avg_epoch_accuracy)
```
This approach should help you prevent both overfitting (by stopping early) and underfitting (by ensuring sufficient epochs for convergence).

#