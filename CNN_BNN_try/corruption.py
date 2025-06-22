# data_dir = 'Chicks_Automatic_Detection_dataset/Registrazioni/audio_segments/'
import torch
import os

data_dir = '../audio_segments'
for f in os.listdir(data_dir):
    if f.endswith('.pt'):
        try:
            data = torch.load(os.path.join(data_dir, f))
            if 'spectrogram' not in data or 'label' not in data:
                print(f"File {f} has unexpected contents")
        except:
            print(f"Corrupted file: {f}")