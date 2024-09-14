# Import necessary PyTorch modules for building neural networks
import torch
import torch.nn as nn

# Define the Discriminator class that inherits from nn.Module
class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        # Call the __init__ method of nn.Module (the parent class)
        super(Discriminator, self).__init__()

        # Define the sequential model (a linear stack of layers) for the discriminator network
        self.disc = nn.Sequential(
            # First convolutional layer
            # Input shape: (N, channels_img, 64, 64), where N is the batch size, channels_img is the number of input channels
            nn.Conv2d(
                channels_img,  # Number of input channels (e.g., 3 for RGB images)
                features_d,     # Number of output channels (depth) for this layer (controls the feature maps)
                kernel_size=4,  # Size of the convolutional filter (4x4)
                stride=2,       # How much the filter moves across the image (2 pixels at a time)
                padding=1       # Padding added around the image to maintain size after convolution
            ),
            # Apply LeakyReLU activation to add non-linearity
            # LeakyReLU allows a small, non-zero gradient when the unit is not active
            nn.LeakyReLU(0.2),  # Negative slope of 0.2 to avoid vanishing gradients for negative values

            # Second block: Convolution + BatchNorm + LeakyReLU
            # This block will downsample the image and learn more complex features
            self._block(features_d, features_d * 2, 4, 2, 1),  # Increases depth (number of feature maps)
            
            # Third block: Same structure with more filters (features_d * 2 to features_d * 4)
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            
            # Fourth block: Same structure with even more filters (features_d * 4 to features_d * 8)
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            
            # Final block: Convolution to reduce the output to a single value (the "real" or "fake" prediction)
            # Output shape will be (N, 1, 1, 1), where N is the batch size, 1 represents the final output
            nn.Conv2d(
                features_d * 8,  # Input depth (number of feature maps)
                1,               # Output depth is 1 (single prediction for real or fake)
                kernel_size=4,    # 4x4 filter
                stride=2,         # Stride of 2 for downsampling
                padding=1         # Padding to ensure proper dimensionality
            ),
            
            # Sigmoid activation function: It squashes the output between 0 and 1
            # This is useful for binary classification (real or fake)
            nn.Sigmoid(),
        )

    # A helper method to create a block of Conv2D -> BatchNorm -> LeakyReLU layers
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            # Convolutional layer
            nn.Conv2d(
                in_channels,   # Input depth (from the previous layer)
                out_channels,  # Output depth (number of feature maps this layer produces)
                kernel_size,   # Filter size (e.g., 4x4)
                stride,        # Stride for downsampling
                padding,       # Padding to keep the image size consistent
                bias=False     # No bias term as BatchNorm takes care of it
            ),
            # Batch normalization helps stabilize training by normalizing the output of the Conv layer
            nn.BatchNorm2d(out_channels),
            # Apply LeakyReLU activation function
            nn.LeakyReLU(0.2),
        )
    
    # Forward method: Defines how the input tensor (image) flows through the network layers
    def forward(self, x):
        # Pass the input 'x' (image) through the 'disc' network (the sequential layers)
        return self.disc(x)
    
    


# ______________________________________________________________________________________________________________________________________





class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("Success, tests passed!")


    test()