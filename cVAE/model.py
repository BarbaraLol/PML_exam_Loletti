import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class FixedSimpleEncoder(nn.Module):
    """Properly sized encoder with gradual dimension reduction"""
    def __init__(self, input_shape, latent_dim=128):  # Increased latent_dim
        super(FixedSimpleEncoder, self).__init__()
        
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # More gradual downsampling with proper channel progression
        self.encoder = nn.Sequential(
            # Layer 1: 1025x938 -> 513x469 (roughly half)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 2: 513x469 -> 257x235
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 3: 257x235 -> 129x118
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            
            # Layer 4: 129x118 -> 65x59
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.2),
        )

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            encoder_output = self.encoder(dummy)
            self.encoder_shape = encoder_output.shape[1:]
            self.encoder_flatten = encoder_output.numel() // encoder_output.shape[0]
            print(f"Fixed Encoder output shape: {self.encoder_shape}")
            print(f"Fixed Encoder flatten size: {self.encoder_flatten}")

        # Gradual FC reduction instead of direct jump
        self.fc_mu = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, 2048),  # First reduction
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),  # Second reduction
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)  # Final to latent
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.encoder_flatten, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        return self.fc_mu(x), self.fc_logvar(x)


class FixedSimpleDecoder(nn.Module):
    """Properly sized decoder that mirrors the encoder"""
    def __init__(self, output_shape, encoder_shape, latent_dim=128):
        super(FixedSimpleDecoder, self).__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim

        if len(encoder_shape) == 3:
            self.channels, self.height, self.width = encoder_shape
        elif len(encoder_shape) == 4:
            _, self.channels, self.height, self.width = encoder_shape
        else:
            raise ValueError(f"Invalid encoder_shape: {encoder_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # Gradual FC expansion (mirror of encoder)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, self.encoder_flatten)
        )

        # Mirror the encoder architecture exactly
        self.decoder = nn.Sequential(
            # Reverse of Layer 4: 65x59 -> 129x118
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # Reverse of Layer 3: 129x118 -> 257x235
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # Reverse of Layer 2: 257x235 -> 513x469
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Reverse of Layer 1: 513x469 -> 1025x938 (approximately)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), self.channels, self.height, self.width)
        x = self.decoder(x)
        return x


class FixedSimpleVariationalAutoEncoder(nn.Module):
    """Fixed VAE with proper architecture balance"""
    def __init__(self, input_shape, latent_dim=128, beta=1.0):  # Increased latent_dim
        super(FixedSimpleVariationalAutoEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = FixedSimpleEncoder(input_shape, latent_dim)
        self.decoder = FixedSimpleDecoder(
            output_shape=input_shape,
            encoder_shape=self.encoder.encoder_shape,
            latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        # Reconstruction loss - use MSE but normalize by number of elements
        reconstruction_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL divergence loss with proper normalization
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

        # More conservative logvar clamping
        logvar = torch.clamp(logvar, min=-10, max=5)

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


class FixedSimpleConditionalVAE(FixedSimpleVariationalAutoEncoder):
    """Fixed conditional VAE with proper architecture"""
    def __init__(self, input_shape, latent_dim=128, num_classes=3, embed_dim=64):  # Increased embed_dim
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Larger label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Modify encoder first layer to accept concatenated input
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,  # Input channels + embedded channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer to accept latent + embedding
        self.decoder.fc = nn.Sequential(
            nn.Linear(latent_dim + embed_dim, 512),  # Now properly sized
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(2048, self.decoder.encoder_flatten)
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
        
        # Expand label embedding spatially to match input
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