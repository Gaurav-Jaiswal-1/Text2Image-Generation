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
    
    

