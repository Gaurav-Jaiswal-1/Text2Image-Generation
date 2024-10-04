import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # For mean of the latent variable z
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # For log variance of z
        
        # Decoder
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        """Encodes the input into the latent space."""
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Samples from the latent space using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)    # Random noise
        return mu + eps * std
    
    def decode(self, z):
        """Decodes the latent variable z back into the input space."""
        h2 = F.relu(self.fc2(z))
        return torch.sigmoid(self.fc3(h2))  # Sigmoid for binary data like images (between 0 and 1)
    
    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x.view(-1, 784))  # Flatten input (for images like MNIST)
        z = self.reparameterize(mu, logvar)        # Sample z
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        """Computes the VAE loss (reconstruction loss + KL divergence)."""
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        
        # KL Divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD
