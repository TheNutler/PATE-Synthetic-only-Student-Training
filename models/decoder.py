"""VAE Decoder model for generating synthetic MNIST images."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VAEDecoder(nn.Module):
    """
    Convolutional transpose decoder that maps latent_dim → 28×28 image.
    
    Architecture:
    - Linear layer: latent_dim → 7*7*128
    - Reshape to: (128, 7, 7)
    - ConvTranspose2d: 128 → 64 channels, upscale to 14×14
    - ConvTranspose2d: 64 → 32 channels, upscale to 28×28
    - Conv2d: 32 → 1 channel (final image)
    
    Args:
        latent_dim: Dimension of the latent space (default: 32)
    """
    
    def __init__(self, latent_dim: int = 32):
        super(VAEDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Project latent vector to 7×7×128 feature map
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        
        # Convolutional transpose layers to upsample
        self.conv_t1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # 7×7 → 14×14
        self.conv_t2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 14×14 → 28×28
        
        # Final convolution to single channel
        self.conv_final = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector of shape (batch_size, latent_dim)
            
        Returns:
            Decoded image of shape (batch_size, 1, 28, 28) in range [0, 1]
        """
        batch_size = z.size(0)
        
        # Project to feature map
        x = self.fc(z)
        x = x.view(batch_size, 128, 7, 7)
        
        # Upsample through transpose convolutions
        x = F.relu(self.conv_t1(x))
        x = F.relu(self.conv_t2(x))
        
        # Final convolution and sigmoid to [0, 1]
        x = torch.sigmoid(self.conv_final(x))
        
        return x


class VAEEncoder(nn.Module):
    """
    Encoder for VAE training (only used during pretraining).
    
    Maps 28×28 image → (mu, logvar) for latent space.
    """
    
    def __init__(self, latent_dim: int = 32):
        super(VAEEncoder, self).__init__()
        self.latent_dim = latent_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)  # 28×28 → 14×14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14×14 → 7×7
        
        # Flatten and project to latent space
        self.fc_mu = nn.Linear(7 * 7 * 64, latent_dim)
        self.fc_logvar = nn.Linear(7 * 7 * 64, latent_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode image to latent distribution parameters.
        
        Args:
            x: Image of shape (batch_size, 1, 28, 28)
            
        Returns:
            Tuple of (mu, logvar) both of shape (batch_size, latent_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class VAE(nn.Module):
    """
    Full VAE model (encoder + decoder) for pretraining.
    Only the decoder is saved and used for generation.
    """
    
    def __init__(self, latent_dim: int = 32):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input image of shape (batch_size, 1, 28, 28)
            
        Returns:
            Tuple of (reconstructed_image, mu, logvar)
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

