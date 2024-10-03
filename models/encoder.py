# encoder.py
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),  # Output: [64, 16, 16]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [128, 8, 8]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [256, 4, 4]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Output: [512, 2, 2]
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Flatten and map to latent vector
        self.fc = nn.Linear(512 * 2 * 2, latent_dim)
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.encoder(x)
        
        # Flatten the tensor for the fully connected layer
        x = x.view(x.size(0), -1)
        
        # Map to the latent space
        latent_vector = self.fc(x)
        
        return latent_vector


# Example usage:
if __name__ == "__main__":
    # Example input: batch of 3 RGB images of size 32x32
    input_data = torch.randn(3, 3, 32, 32)
    
    # Instantiate the encoder
    encoder = Encoder(input_channels=3, latent_dim=128)
    
    # Forward pass through the encoder
    latent_vector = encoder(input_data)
    
    # Print the shape of the latent vector
    print("Latent vector shape:", latent_vector.shape)
