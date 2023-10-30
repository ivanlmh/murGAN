import torch.nn as nn
import torch.nn.functional as F


# Define ResNet-like blocks
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        x = self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))
        x += residual
        return self.relu(x)


# Define Generator
# Converts an input from one domain to another
# Consists of down-sampling layers, bottleneck layers, and up-sampling layers
class Generator(nn.Module):
    def __init__(self, input_dim=128, num_resnet_blocks=6):
        super(Generator, self).__init__()

        # Down-sampling layers
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=input_dim,
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU(inplace=True)

        # Residual blocks
        self.resnet_blocks = []
        for _ in range(num_resnet_blocks):
            self.resnet_blocks += [
                ResidualBlock(
                    in_channels=input_dim,
                    out_channels=input_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            ]

        # Up-sampling Layers
        self.conv_transpose1 = nn.ConvTranspose2d(
            in_channels=input_dim,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding=3,
            output_padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(1)
        self.tanh = nn.Tanh()

        model = (
            [self.conv, self.bn1, self.relu]
            + self.resnet_blocks
            + [
                self.conv_transpose1,
                self.bn2,
                self.tanh,
            ]
        )
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    

# Define Discriminator
# Classifies whether an image is real or fake
# AND classifies whether an image is from murga or non-murga
class Discriminator(nn.Module):
    def __init__(self, input_dim=128):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=input_dim,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=input_dim,
            out_channels=input_dim * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(input_dim * 2)

        model = [self.conv1, self.leaky_relu, self.conv2, self.bn2, self.leaky_relu]
        self.model = nn.Sequential(*model)

        # Resulting size
        # Input size is [128 (frequency bins) x 862 (time frames for 10 seconds segment)]
        # After two 4x4 convolutions with stride 2, height/width is halved twice
        # If more layers are added, this needs to be updated.
        freq_dim = 128 // 4  # Reduced by two conv layers with stride 2
        time_dim = 862 // 4   # T should be the number of time frames in the 10 seconds segment after MelSpectrogram
        
        flattened_dim = (input_dim * 2) * freq_dim * time_dim
        
        # Apply linear layer to get final outputs
        self.fc_real_fake = nn.Linear(flattened_dim, 1) # Real or fake
        self.fc_domain = nn.Linear(flattened_dim, 1) # Murga or non-murga

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        return self.fc_real_fake(x), self.fc_domain(x)

