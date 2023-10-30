import unittest
from src.data_loader import VocalsDataset, MelSpectrogramTransform, create_data_loader

ROOT_DIR = "data/murga"

# attempt at creating a test for the data loader
class DataLoaderTest(unittest.TestCase):
    def test_vocals_dataset_loading(self):
        dataset = VocalsDataset(root_dir=ROOT_DIR)
        self.assertTrue(len(dataset) > 0) # at least one file
        waveform = dataset[0]
        self.assertEqual(waveform.shape[0], 1) # mono
        self.assertEqual(waveform.shape[1], 44100*10) # 10 seconds

    def test_mel_spectrogram_transform(self):
        transform = MelSpectrogramTransform()
        dataset = VocalsDataset(root_dir=ROOT_DIR)
        waveform, sample_rate = dataset[0]
        mel_spectrogram = transform(waveform)
        # self.assertEqual(mel_spectrogram, VocalsDataset(root_dir=ROOT_DIR, transform=transform)[0][0])
        self.assertEqual(len(mel_spectrogram.shape), 3)
        self.assertEqual(mel_spectrogram.shape[0], 1)
        self.assertEqual(mel_spectrogram.shape[1], 128) # 128 mel bands
        self.assertEqual(mel_spectrogram.shape[2], round(44100*10/512 + 0.5)) # 10 seconds in mel frames is 862
