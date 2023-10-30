import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from torchaudio.transforms import MelSpectrogram, Resample, Spectrogram, AmplitudeToDB


# Create a dataset class
class VocalsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = (
            transform  # In case we want to apply a transform to the waveform
        )
        self.file_names = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_name = self.file_names[idx]
        file_path = os.path.join(self.root_dir, file_name)
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate


# Preprocessing
class MelSpectrogramTransform:
    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128):
        self.transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.amplitude_to_db = AmplitudeToDB()

    def __call__(self, waveform):
        spectrogram = self.transform(waveform)
        return self.amplitude_to_db(spectrogram)


def create_data_loader(root_dir, batch_size=8, shuffle=True, num_workers=0):
    dataset = VocalsDataset(root_dir, transform=MelSpectrogramTransform())
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
