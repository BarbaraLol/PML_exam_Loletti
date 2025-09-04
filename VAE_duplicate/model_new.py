import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class Encoder(nn.Module):
    """Simplified, stable encoder for VAE"""
    def __init__(self, input_shape, latent_dim=128):
        super(Encoder, self).__init__()
        
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # FIXED: Much simpler architecture - only 3 conv layers
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # /2
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),
            
            # Second conv block  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # /4
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # /8
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            encoder_output = self.encoder(dummy)
            self.encoder_shape = encoder_output.shape[1:]  # [C, H, W]
            self.encoder_flatten = encoder_output.numel() // encoder_output.shape[0]
            print(f"Encoder output shape: {self.encoder_shape}")
            print(f"Flatten size: {self.encoder_flatten}")

        # FIXED: Smaller latent dimension for stability
        self.fc_mu = nn.Linear(self.encoder_flatten, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_flatten, latent_dim)
        
        # FIXED: Initialize logvar to small negative values for stability
        nn.init.normal_(self.fc_logvar.weight, 0, 0.001)
        nn.init.constant_(self.fc_logvar.bias, -3.0)  # Start with small variance

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # FIXED: Clamp logvar to prevent explosion
        logvar = torch.clamp(logvar, min=-10, max=2)
        
        return mu, logvar


class Decoder(nn.Module):
    """Simplified, stable decoder for VAE"""
    def __init__(self, output_shape, encoder_shape, latent_dim=128):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        if len(encoder_shape) == 3:  # [C, H, W]
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:  # [B, C, H, W]
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # FC layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.encoder_flatten)

        # FIXED: Simpler decoder architecture - 3 layers to match encoder
        self.decoder = nn.Sequential(
            # First deconv block
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            # Second deconv block
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            # Final deconv block
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Ensure output in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.decoder(x)
        return x


class VariationalAutoEncoder(nn.Module):
    """Fixed VAE with stability improvements"""
    def __init__(self, input_shape, latent_dim=128, beta=1e-4):
        super(VariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = Encoder(input_shape, latent_dim)
        self.decoder = Decoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,
            latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with numerical stability"""
        std = torch.exp(0.5 * logvar)
        # FIXED: Clamp std to prevent numerical issues
        std = torch.clamp(std, min=1e-6, max=10)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """Stable VAE loss with proper scaling"""
        if beta is None:
            beta = self.beta
        
        # FIXED: Use MSE loss with proper reduction
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # FIXED: Stable KL divergence calculation
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # FIXED: Clamp KL loss to prevent explosion
        kl_loss = torch.clamp(kl_loss, min=0, max=100)
        
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """Generate new samples from the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # FIXED: Ensure output matches input dimensions
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return recon_x, mu, logvar


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    """Fixed Conditional VAE"""
    def __init__(self, input_shape, latent_dim=128, num_classes=3, embed_dim=32):
        # FIXED: Smaller embedding dimension
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # FIXED: Modify encoder first layer to accept additional channels
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,  # Original + embedded label channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # FIXED: Modify decoder FC layer
        original_fc = self.decoder.fc
        self.decoder.fc = nn.Linear(
            latent_dim + embed_dim,
            original_fc.out_features
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate samples for a specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            return samples

    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Embed labels
        label_embed = self.label_embedding(labels)
        
        # Expand label embedding to match spatial dimensions
        label_embed_expanded = label_embed.view(batch_size, self.embed_dim, 1, 1)
        label_embed_expanded = label_embed_expanded.expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate input with label embedding
        x_conditioned = torch.cat([x, label_embed_expanded], dim=1)
        
        # Encode
        mu, logvar = self.encoder(x_conditioned)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Concatenate latent with label embedding for decoder
        z_conditioned = torch.cat([z, label_embed], dim=1)
        
        # Decode
        recon_x = self.decoder(z_conditioned)
        
        # Ensure correct output shape
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return recon_x, mu, logvar