# murGAN: Singing Voice to Murga singing Style Transfer

This repository contains a simple approach to transfer the singing style of classical or pop singers to the unique style of Uruguayan Murga (of course, it can actually be trained to transfer to any style).

*Murga is a modern musical expression native to Uruguay. The singing has distinctive characteristics, especially when compared to pop singers or classical European choral singing. There are significant differences in spectral centroid and spectral flatness. There are significant differences in spectral centroid and spectral flatness. Murga singing often presents deviation from the fundamental frequency, and not much vibrato, among other particularities in terms of intonation and vocal expression.*

I use a Generative Adversarial Network architecture, copying the idea from the StarGAN architecture, that the discriminator can not only learn to distinguish real from fake audio, but also of the output of the generator is from a desired domain.


## Getting Started

1. Clone the repository:
    ```bash
    git clone git@github.com:ivanlmh/murGAN.git
    cd murGAN
    ```

2. Set up the environment:
    ```bash
    conda env create -f environment.yml
    conda activate murGAN
    ```

3. Install project for development
    ```bash
    pip install -e .
    ```

## Dataset Preparation

The datasets I use consist of two folders: one with "Murga" style singing and another with non-murga (classical, pop, etc) singing. Each track in the dataset should be at least 10 seconds long to ensure consistent input size for the model. For non-murga I use the vocals from MUSDB18 and the stems from the Choral Singing Dataset (see referances).

In the ```scripts``` folder you will find a script that creates symlinks of audios, so that you can add files from other projects to a local ```data``` folder, and keep things clean while not taking up more space.

## Model Architecture

The model leverages the StarGAN paradigm. For more details on the architecture, refer to the original [StarGAN paper](https://arxiv.org/abs/1711.09020).

## Training

To train the model, run:

```bash
python src/train.py
```

Ensure that both murga and classic datasets are prepared and placed in their respective directories.

## Inference

Once the model is trained, convert classical singing to the "Murga" style using:

```bash
python src/inference.py --input "path_to_input_classical_audio.wav" --output "path_to_output_murga_audio.wav"
```

## Tests
To run unit tests

```bash
python -m unittest discover tests
```

## Acknowledgments

- ChoralSingingDataset https://zenodo.org/records/2649950
- MUSDB18 https://sigsep.github.io/datasets/musdb.html#musdb18-compressed-stems
- StarGAN https://arxiv.org/abs/1711.09020