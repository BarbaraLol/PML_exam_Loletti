import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return self.relu(out)


class Encoder(nn.Module):
    """Encoder part of the VAE"""
    def __init__(self, input_shape, latent_dim=1024):
        super(Encoder, self).__init__()
        # Ensure input_shape is (channels, height, width)
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)  # Add channel dim
        else:
            self.input_shape = input_shape

        # Encoder with consistent dropout
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  
            nn.Dropout2d(0.1),

            nn.Conv2d(128, 256, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.15),

            nn.Conv2d(256, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.15),

            nn.Conv2d(512, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2), 
            nn.Dropout2d(0.2)
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            self.encoder_output = self.encoder(dummy)
            self.encoder_shape = self.encoder_output.shape[1:]
            self.encoder_flatten = self.encoder_output.numel() // self.encoder_output.shape[0]
            print(f"Encoder output shape: {self.encoder_shape}")
            print(f"Flatten size: {self.encoder_flatten}")

        # FC layers with proper dropout
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, latent_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, latent_dim)
        )

    def forward(self, x):
        # Ensure proper 4D input
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dim
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nn.Module):
    """Decoder part of the VAE"""
    def __init__(self, output_shape, encoder_shape, latent_dim=1024):
        super(Decoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Handle both 3D and 4D encoder_shape
        if len(encoder_shape) == 3:  # [C, H, W]
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:  # [B, C, H, W]
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        # Flattened size for FC layer
        self.encoder_flatten = self.channels * self.height * self.width

        # FC layer to expand latent vector with dropout
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim, self.encoder_flatten)
        )

        # Decoder architecture
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size = 2, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 1, kernel_size = 2, stride = 2, padding = 1),
            nn.Sigmoid()  # Removed BatchNorm before Sigmoid
        )

    def forward(self, z):
        # Expand latent vector
        x = self.fc(z)

        # Reshape for conv layers
        x = x.view(x.size(0), self.channels, self.height, self.width)

        # Decode
        x = self.decoder(x)

        return x


class VariationalAutoEncoder(nn.Module):
    """Variational Autoencoder for spectrograms"""
    def __init__(self, input_shape, latent_dim = 1024, beta = 1.0):
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
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """VAE Loss = reconstruction + KL divergence"""
        # Reconstruction loss with MSE
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        # Use beta if provided, else use class default
        effective_beta = self.beta if beta is None else beta
        total_loss = reconstruction_loss + (effective_beta * kl_loss)

        return total_loss, reconstruction_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        """Generate new samples from the latent space"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)

        # Prevent logvar explosion with stronger clamping
        logvar = torch.clamp(logvar, min=-10, max=5)

        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # Reshape reconstruction to match input
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return recon_x, mu, logvar


class ConditionalVariationalAutoEncoder(VariationalAutoEncoder):
    """VAE that can generate spectrograms for specific labels"""
    def __init__(self, input_shape, latent_dim=1024, num_classes=3, embed_dim=100):
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Modify encoder first conv layer to accept additional channels
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,  # Original + embedded label channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer to accept latent + label embedding
        original_fc_layer = self.decoder.fc[-1]  # Get the Linear layer from Sequential
        self.decoder.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(latent_dim + embed_dim, original_fc_layer.out_features)
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate samples for a specific class"""
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            
            # Interpolate to target size if needed
            if samples.shape[-2:] != self.input_shape[-2:]:
                samples = F.interpolate(
                    samples,
                    size=self.input_shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            return samples

    def forward(self, x, labels):
        batch_size = x.size(0)
        
        # Embed labels
        label_embed = self.label_embedding(labels)  # [batch_size, embed_dim]
        
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

        # Add interpolation to match input size
        if recon_x.shape[-2:] != x.shape[-2:]:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return recon_x, mu, logvar