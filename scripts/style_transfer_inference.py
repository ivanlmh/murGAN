import torch
import torchaudio
from src.models import Generator
from src.data_loader import MelSpectrogramTransform
from src.utils import inverse_mel


def style_transfer_inference(
    input_audio_path, output_audio_path, generator_model_path, device="cuda"
):
    # Load the pretrained generator
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_model_path))
    generator.eval()

    # Load the input audio
    waveform, sample_rate = torchaudio.load(input_audio_path)
    assert sample_rate == 44100, "Sample rate must be 44100"
    waveform = waveform.to(device)

    # Preprocess the input audio
    mel_spectrogram = MelSpectrogramTransform()(waveform)
    mel_spectrogram = mel_spectrogram.to(device)

    # Iterate on 10-second chunks of the input audio
    output = []
    for i in range(0, mel_spectrogram.shape[2], 862):
        mel_spectrogram_chunk = mel_spectrogram[:, :, i : i + 862]
        output_chunk = generator(mel_spectrogram_chunk)
        # Inverse mel
        output_chunk = inverse_mel(output_chunk)
        output.append(output_chunk)

    # Save the output audio
    output = torch.cat(output, dim=1)
    torchaudio.save(output_audio_path, output, sample_rate)


if __name__ == "__main__":
    style_transfer_inference(
        input_audio_path="data/classic/CSD_ER_tenor_1.wav",
        output_audio_path="out/output.mp3",
        generator_model_path="saved_models/generator.pth",
    )
