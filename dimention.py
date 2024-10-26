import torch
import os

# Directory where your .pt files are located
directory = './Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'

# Expected dimensions
expected_dimensions = torch.Size([1025, 938])

# List to store files with non-matching dimensions
mismatch_files = []

# Iterate through each file in the directory
for filename in os.listdir(directory):
    if filename.endswith('.pt'):
        # Load the .pt file with weights_only=True to avoid the warning
        file_path = os.path.join(directory, filename)
        data = torch.load(file_path, weights_only=True)
        
        # Check if data is a dictionary, tuple, or tensor and extract spectrogram
        if isinstance(data, dict):
            # Assuming the spectrogram is stored under a key, e.g., 'spectrogram'
            spectrogram = data.get('spectrogram', None)
            if spectrogram is None:
                print(f"Warning: No 'spectrogram' key found in {filename}")
                continue
        elif isinstance(data, tuple) and len(data) == 2:
            # If it's a tuple, assuming the first item is the spectrogram
            spectrogram, label = data
        else:
            # If it's directly a tensor, assign it to spectrogram
            spectrogram = data
        
        # Check if spectrogram is a tensor, then get its size and compare
        if isinstance(spectrogram, torch.Tensor):
            dimensions = spectrogram.size()
            
            # Log files that do not match the expected dimensions
            if dimensions != expected_dimensions:
                mismatch_files.append((filename, dimensions))
        else:
            print(f"Warning: No valid tensor found in {filename}")

# Print the results
if mismatch_files:
    print("Files with dimensions different from [1025, 938]:")
    for filename, dimensions in mismatch_files:
        print(f"  - {filename}: {dimensions}")
else:
    print("All spectrograms have the dimensions [1025, 938]")
