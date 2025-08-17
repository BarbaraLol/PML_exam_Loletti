import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np 

class AttentionBlock(nn.Module):
    """Self-attention for capturing long-range dependencies in spectrograms"""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Generate query, key, value
        query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, width * height)
        value = self.value(x).view(batch_size, -1, width * height)
        
        # Attention weights
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ResidualBlock(nn.Module):
    """Improved residual block with attention and better normalization"""
    def __init__(self, in_channels, use_attention=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # Use Group Normalization instead of BatchNorm for better small batch performance
        self.norm1 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        self.norm2 = nn.GroupNorm(min(32, in_channels // 4), in_channels)
        
        self.activation = nn.GELU()  # Better than LeakyReLU for VAEs
        self.dropout = nn.Dropout2d(0.1)
        
        self.attention = AttentionBlock(in_channels) if use_attention else None
        
    def forward(self, x):
        residual = x
        
        out = self.activation(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        
        if self.attention:
            out = self.attention(out)
        
        out += residual
        return self.activation(out)

class ImprovedEncoder(nn.Module):
    """Enhanced encoder with progressive resolution reduction and attention"""
    def __init__(self, input_shape, latent_dim=1024):
        super().__init__()
        
        if isinstance(input_shape, torch.Size):
            input_shape = tuple(input_shape)
        if len(input_shape) == 2:
            self.input_shape = (1, *input_shape)
        else:
            self.input_shape = input_shape

        # Progressive encoding with residual connections
        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        
        # Encoder blocks with progressive channel increase
        self.encoder_blocks = nn.ModuleList([
            # Block 1: 64 -> 128, reduce by 2
            nn.Sequential(
                ResidualBlock(64),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            # Block 2: 128 -> 256, reduce by 2
            nn.Sequential(
                ResidualBlock(128),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU(),
            ),
            # Block 3: 256 -> 512, reduce by 2
            nn.Sequential(
                ResidualBlock(256, use_attention=True),  # Add attention in deeper layers
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 512),
                nn.GELU(),
            ),
            # Block 4: 512 -> 512, reduce by 2
            nn.Sequential(
                ResidualBlock(512, use_attention=True),
                nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 512),
                nn.GELU(),
            ),
        ])
        
        # Calculate output dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, *self.input_shape)
            x = self.initial_conv(dummy)
            for block in self.encoder_blocks:
                x = block(x)
            self.encoder_output_shape = x.shape[1:]
            self.encoder_flatten = x.numel() // x.shape[0]
            print(f"Improved encoder output shape: {self.encoder_output_shape}")
            print(f"Flatten size: {self.encoder_flatten}")

        # Latent space projections with dropout
        self.fc_mu = nn.Sequential(
            nn.Linear(self.encoder_flatten, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.encoder_flatten, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.initial_conv(x)
        for block in self.encoder_blocks:
            x = block(x)
        
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class ImprovedDecoder(nn.Module):
    """Enhanced decoder with skip connections and progressive upsampling"""
    def __init__(self, output_shape, encoder_output_shape, latent_dim=1024):
        super().__init__()
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        
        if len(encoder_output_shape) == 3:
            self.channels, self.height, self.width = encoder_output_shape
        elif len(encoder_output_shape) == 4:
            _, self.channels, self.height, self.width = encoder_output_shape
        else:
            raise ValueError(f"Invalid encoder_output_shape: {encoder_output_shape}")
        
        self.encoder_flatten = self.channels * self.height * self.width

        # Enhanced FC layers
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, self.encoder_flatten),
            nn.GELU()
        )
        
        # Progressive decoder blocks
        self.decoder_blocks = nn.ModuleList([
            # Block 1: 512 -> 256, upsample by 2
            nn.Sequential(
                ResidualBlock(512, use_attention=True),
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(32, 256),
                nn.GELU(),
            ),
            # Block 2: 256 -> 128, upsample by 2
            nn.Sequential(
                ResidualBlock(256, use_attention=True),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(16, 128),
                nn.GELU(),
            ),
            # Block 3: 128 -> 64, upsample by 2
            nn.Sequential(
                ResidualBlock(128),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            ),
            # Block 4: 64 -> 32, upsample by 2
            nn.Sequential(
                ResidualBlock(64),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.GroupNorm(4, 32),
                nn.GELU(),
            ),
        ])
        
        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.GroupNorm(2, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        
        # Expand latent vector
        x = self.fc(z)
        x = x.view(batch_size, self.channels, self.height, self.width)
        
        # Progressive upsampling
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x

class ImprovedVariationalAutoEncoder(nn.Module):
    """Enhanced VAE with better architecture and training stability"""
    def __init__(self, input_shape, latent_dim=1024, beta=1.0):
        super().__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta

        self.encoder = ImprovedEncoder(input_shape, latent_dim)
        self.decoder = ImprovedDecoder(
            output_shape=input_shape,
            encoder_output_shape=self.encoder.encoder_output_shape,
            latent_dim=latent_dim
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def loss_function(self, recon_x, x, mu, logvar, beta=None):
        """Enhanced loss with perceptual components"""
        effective_beta = self.beta if beta is None else beta
        
        # Reconstruction loss (MSE + L1 for better detail preservation)
        mse_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        l1_loss = F.l1_loss(recon_x, x, reduction='sum') / x.size(0)
        reconstruction_loss = 0.8 * mse_loss + 0.2 * l1_loss
        
        # KL divergence with better numerical stability
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # Add spectral loss for frequency domain preservation
        spectral_loss = self._spectral_loss(recon_x, x)
        
        total_loss = reconstruction_loss + (effective_beta * kl_loss) + (0.1 * spectral_loss)
        
        return total_loss, reconstruction_loss, kl_loss
    
    def _spectral_loss(self, recon_x, x):
        """Additional loss to preserve spectral characteristics"""
        # Compute frequency domain difference
        x_freq = torch.fft.fft2(x)
        recon_freq = torch.fft.fft2(recon_x)
        
        # Compare magnitude spectra
        x_mag = torch.abs(x_freq)
        recon_mag = torch.abs(recon_freq)
        
        return F.mse_loss(recon_mag, x_mag, reduction='mean')

    def sample(self, num_samples, device='cpu'):
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(device)
            samples = self.decoder(z)
            return samples

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        # Ensure output matches input size
        if recon_x.shape != x.shape:
            recon_x = F.interpolate(
                recon_x,
                size=x.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        return recon_x, mu, logvar

class ConditionalVariationalAutoEncoder(ImprovedVariationalAutoEncoder):
    """VAE that is able to generate spectrograms for specific lables"""
    def __init__(self, input_shape, latent_dim = 512, num_classes = 3, embed_dim=100):
        super().__init__(input_shape, latent_dim)
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Label embedding
        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Encoder modification to accept concatenated input
        # Adjusting the first convolutional layer so to accept additional informations/channels
        original_conv = self.encoder.encoder[0]
        self.encoder.encoder[0] = nn.Conv2d(
            1 + embed_dim,  # Original + embedded label channels
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding
        )
        
        # Modify decoder FC layer to accept latent + label embedding
        original_fc = self.decoder.fc
        self.decoder.fc = nn.Linear(
            latent_dim + embed_dim,
            original_fc.out_features
        )

    def sample_class(self, class_label, num_samples, device='cpu'):
        """Generate samples for a specific class"""
        with torch.no_grad():
            # Create latent samples
            z = torch.randn(num_samples, self.latent_dim).to(device)
            
            # Create class labels
            labels = torch.full((num_samples,), class_label, dtype=torch.long).to(device)
            label_embed = self.label_embedding(labels)
            
            # Concatenate and decode
            z_conditioned = torch.cat([z, label_embed], dim=1)
            samples = self.decoder(z_conditioned)
            
            return samples 

    def encode(self, x, class_labels):
        pass

    def decode(self, z, class_labels):
        pass

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
        
        return recon_x, mu, logvar
