import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class SimpleEncoder(nn.Module):
    """Much simpler encoder with only 3 conv layers"""
    def __init__(self, input_shape, latent_dim=32):
        super(SimpleEncoder, self).__init__()
        
        # Ensure input_shape is (channels, height, width)
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # Much simpler encoder - only 3 layers
        self.encoder = nn.Sequential(
            # Layer 1: 1025x938 -> 512x469
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 2: 512x469 -> 256x234  
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 3: 256x234 -> 128x117
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            encoder_output = self.encoder(dummy)
            self.encoder_shape = encoder_output.shape[1:]
            self.encoder_flatten = encoder_output.numel() // encoder_output.shape[0]
            print(f"Simple Encoder output shape: {self.encoder_shape}")
            print(f"Simple Encoder flatten size: {self.encoder_flatten}")

        # Much smaller FC layers
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        return self.fc_mu(x), self.fc_logvar(x)


class SimpleDecoder(nn.Module):
    """Much simpler decoder with only 3 layers"""
    def __init__(self, output_shape, encoder_shape, latent_dim=32):
        super(SimpleDecoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        # Handle encoder shape
        if len(encoder_shape) == 3:
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # Much smaller FC expansion
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.encoder_flatten)
        )

        # Simple decoder - only 3 layers (mirror of encoder)
        self.decoder = nn.Sequential(
            # Layer 1: 128x117 -> 256x234
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Layer 2: 256x234 -> 512x469
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Layer 3: 512x469 -> 1025x938 (approximately)
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.decoder(x)
        return x


class SimpleVariationalAutoEncoder(nn.Module):
    """Much simpler VAE with fewer parameters"""
    def __init__(self, input_shape, latent_dim=32, beta=1.0):
        super(SimpleVariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = SimpleEncoder(input_shape, latent_dim)
        self.decoder = SimpleDecoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,
            latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        effective_beta = self.beta if beta is None else beta
        total_loss = reconstruction_loss + (effective_beta * kl_loss)

        return total_loss, reconstruction_loss, kl_loss

    def sample(self, num_samples, device='cpu'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)

        # Stronger logvar clamping for stability
        logvar = torch.clamp(logvar, min=-5, max=2)

        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)

        # Interpolate to match input size
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        return recon_x, mu, logvar


class SimpleConditionalVAE(SimpleVariationalAutoEncoder):
    """Simple conditional VAE"""
    def __init__(self, input_shape, latent_dim=32, num_classes=3, embed_dim=16):
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Small label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Modify encoder first layer
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer
        self.decoder.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.decoder.encoder_flatten)
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            
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
        label_embed = self.label_embedding(labels)
        
        # Expand label embedding spatially
        label_embed_expanded = label_embed.view(batch_size, self.embed_dim, 1, 1)
        label_embed_expanded = label_embed_expanded.expand(-1, -1, x.size(2), x.size(3))
        
        # Concatenate input with label embedding
        x_conditioned = torch.cat([x, label_embed_expanded], dim=1)
        
        # Encode
        mu, logvar = self.encoder(x_conditioned)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode with label conditioning
        z_conditioned = torch.cat([z, label_embed], dim=1)
        recon_x = self.decoder(z_conditioned)

        # Interpolate to match input
        if recon_x.shape[-2:] != x.shape[-2:]:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return recon_x, mu, logvar