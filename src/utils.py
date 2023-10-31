import torch
import torchaudio


# Function to inverse the mel spectrogram
def inverse_mel(mel_spectrogram, sample_rate=44100, n_fft=2048, hop_length=512):
    mel_spectrogram = mel_spectrogram.squeeze(0)
    audio = torchaudio.transforms.InverseMelScale(
        n_stft=n_fft, n_mels=mel_spectrogram.shape[0], sample_rate=sample_rate
    )(mel_spectrogram)
    audio = torchaudio.functional.griffinlim(audio, n_fft=n_fft, hop_length=hop_length)
    return audio
