import pandas as pd
import matplotlib.pyplot as plt

# Load the training log data
df = pd.read_csv('training_log_20250809_173533.csv')

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot Accuracy
ax1.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='o', linewidth=2)
ax1.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s', linewidth=2)
ax1.set_title('Training vs Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot Loss
ax2.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
ax2.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
ax2.set_title('Training vs Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Adjust layout and save combined plot
plt.tight_layout()
plt.savefig('training_metrics_combined.png', dpi=300, bbox_inches='tight')
plt.show()

# Save individual plots
# Accuracy plot
plt.figure(figsize=(8, 6))
plt.plot(df['epoch'], df['train_acc'], label='Training Accuracy', marker='o', linewidth=2)
plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='s', linewidth=2)
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('accuracy_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Loss plot
plt.figure(figsize=(8, 6))
plt.plot(df['epoch'], df['train_loss'], label='Training Loss', marker='o', linewidth=2)
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='s', linewidth=2)
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('loss_plot.png', dpi=300, bbox_inches='tight')
plt.show()