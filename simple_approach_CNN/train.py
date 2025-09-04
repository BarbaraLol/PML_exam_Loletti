# Add these imports at the top of your simple_train.py
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

# Then modify your main() function like this:
def main():
    # Configuration
    data_dir = '../Chicks_Automatic_Detection_dataset/Processed_Data_10sec/audio_segments/'
    num_epochs = 100
    batch_size = 32
    initial_lr = 1e-3
    grad_clip = 1.0
    early_stopping_patience = 15
    min_delta = 1e-4

    # Verify CUDA compatibility
    check_cuda_compatibility()
    device = torch.device("cuda")
    
    # Setup CSV logging
    csv_path = setup_csv_logging()
    print(f"Training results will be logged to: {csv_path}")
    
    # *** ADD THIS SECTION HERE ***
    print("\n" + "="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    
    # Analyze dataset balance
    file_counts, all_labels = analyze_dataset_balance(data_dir)
    
    # Compute class weights
    class_weights = compute_class_weights_for_loss(all_labels)
    
    print("="*50)
    # *** END OF NEW SECTION ***
    
    # Load and process data (your existing code)
    file_paths = load_file_paths(data_dir)
    all_labels_original = encode_labels(file_paths)  # Your original function
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels_original)
    
    dataset = SpectrogramDataset(file_paths, label_encoder)
    sample_spectrogram = torch.load(file_paths[0])['spectrogram']
    input_shape = sample_spectrogram.shape
    num_classes = len(label_encoder.classes_)
    
    # Split dataset (your existing code)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Data loaders (your existing code)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    
    # Model setup
    model = SimpleCNN(input_shape, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    
    # *** MODIFY THIS LINE TO USE CLASS WEIGHTS ***
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    # *** INSTEAD OF: criterion = nn.CrossEntropyLoss() ***
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # ... rest of your training code stays the same ...