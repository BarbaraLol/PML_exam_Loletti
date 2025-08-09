import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('results/5sec_chunks/training_log.csv')

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Training Results Analysis', fontsize=16, fontweight='bold')

# 1. Loss curves (train vs validation)
axes[0, 0].plot(df['epoch'], df['train_loss'], 'b-', label='Train Loss', linewidth=2)
axes[0, 0].plot(df['epoch'], df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training vs Validation Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Accuracy curves (train vs validation)
axes[0, 1].plot(df['epoch'], df['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
axes[0, 1].plot(df['epoch'], df['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].set_title('Training vs Validation Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Learning rate schedule
axes[0, 2].plot(df['epoch'], df['lr'], 'g-', linewidth=2, marker='o', markersize=4)
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Learning Rate')
axes[0, 2].set_title('Learning Rate Schedule')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].set_yscale('log')

# 4. KL Loss over time
axes[1, 0].plot(df['epoch'], df['kl_loss'], 'purple', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('KL Loss')
axes[1, 0].set_title('KL Divergence Loss')
axes[1, 0].grid(True, alpha=0.3)

# 5. Training time per epoch
axes[1, 1].plot(df['epoch'], df['time_elapsed'], 'orange', linewidth=2, marker='s', markersize=4)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Time Elapsed (seconds)')
axes[1, 1].set_title('Cumulative Training Time')
axes[1, 1].grid(True, alpha=0.3)

# 6. Overfitting analysis (difference between train and val accuracy)
accuracy_gap = df['train_acc'] - df['val_acc']
axes[1, 2].plot(df['epoch'], accuracy_gap, 'red', linewidth=2, marker='D', markersize=4)
axes[1, 2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Train Acc - Val Acc (%)')
axes[1, 2].set_title('Overfitting Analysis')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
plt.savefig('results/5sec_chunks/training_results.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('results/5sec_chunks/training_results.pdf', bbox_inches='tight', facecolor='white')
print("Plots saved as 'training_results.png' and 'training_results.pdf'")

plt.show()

# Print summary statistics
print("=" * 50)
print("TRAINING SUMMARY STATISTICS")
print("=" * 50)
print(f"Best Validation Accuracy: {df['val_acc'].max():.2f}% at epoch {df.loc[df['val_acc'].idxmax(), 'epoch']}")
print(f"Final Validation Accuracy: {df['val_acc'].iloc[-1]:.2f}%")
print(f"Best Validation Loss: {df['val_loss'].min():.4f} at epoch {df.loc[df['val_loss'].idxmin(), 'epoch']}")
print(f"Final Validation Loss: {df['val_loss'].iloc[-1]:.4f}")
print(f"Total Training Time: {df['time_elapsed'].iloc[-1]:.1f} seconds ({df['time_elapsed'].iloc[-1]/60:.1f} minutes)")
print(f"Average Time per Epoch: {df['time_elapsed'].iloc[-1]/len(df):.1f} seconds")
print(f"Learning Rate Reductions: {len(df['lr'].unique())} different rates used")

# Check for overfitting
final_gap = df['train_acc'].iloc[-1] - df['val_acc'].iloc[-1]
print(f"Final Accuracy Gap (Train - Val): {final_gap:.2f}%")
if final_gap > 5:
    print("Warning: Potential overfitting detected (train accuracy >> val accuracy)")
elif final_gap < 0:
    print("Good: Validation accuracy higher than training accuracy")
else:
    print("Good: Minimal overfitting observed")
