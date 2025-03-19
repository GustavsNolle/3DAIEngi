import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    """
    A single dense layer with BN-ReLU-Conv structure.
    The output is concatenated with the input.
    """
    def __init__(self, in_channels, growth_rate, kernel_size=3, padding=1):
        super(DenseLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size,
                              padding=padding, bias=False)
        
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        # Concatenate input and output along the channel dimension
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    """
    A dense block that stacks several DenseLayers.
    Each layer increases the number of channels by the growth rate.
    """
    def __init__(self, num_layers, in_channels, growth_rate, kernel_size=3, padding=1):
        super(DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_channels + i * growth_rate,
                                     growth_rate,
                                     kernel_size=kernel_size,
                                     padding=padding))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class TransitionLayer(nn.Module):
    """
    A transition layer that compresses channels and downsamples the spatial dimensions.
    """
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 convolution for channel compression
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # Average pooling for downsampling
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        return self.pool(x)

class PhysicsEncoder(nn.Module):
    """
    A convolutional encoder with dense connections that takes a multi-channel
    input (e.g. [T_g, n_g, P, metal_mask, T_m]) and outputs a latent vector.
    
    The network uses an initial convolution, two dense blocks with transition layers,
    a final convolution, global average pooling, and a fully connected layer.
    """
    def __init__(self, input_channels=5, latent_dim=256):
        super(PhysicsEncoder, self).__init__()
        # Initial convolution to get a feature map.
        self.conv0 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(32)
        self.relu0 = nn.ReLU(inplace=True)
        
        # Dense Block 1: 4 layers, growth rate 16
        self.denseblock1 = DenseBlock(num_layers=4, in_channels=32, growth_rate=16)
        # After DenseBlock1, channels = 32 + 4*16 = 96.
        self.trans1 = TransitionLayer(in_channels=96, out_channels=64)
        
        # Dense Block 2: 4 layers, growth rate 32
        self.denseblock2 = DenseBlock(num_layers=4, in_channels=64, growth_rate=32)
        # After DenseBlock2, channels = 64 + 4*32 = 192.
        self.trans2 = TransitionLayer(in_channels=192, out_channels=128)
        
        # Final convolution layer to further aggregate features.
        self.conv_final = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_final = nn.BatchNorm2d(256)
        self.relu_final = nn.ReLU(inplace=True)
        
        # Global average pooling to collapse spatial dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer to produce the latent vector.
        self.fc = nn.Linear(256, latent_dim)
    
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        x = self.relu0(self.bn0(self.conv0(x)))  # [batch, 32, H, W]
        x = self.denseblock1(x)                  # [batch, 96, H, W]
        x = self.trans1(x)                       # [batch, 64, H/2, W/2]
        x = self.denseblock2(x)                  # [batch, 192, H/2, W/2]
        x = self.trans2(x)                       # [batch, 128, H/4, W/4]
        x = self.relu_final(self.bn_final(self.conv_final(x)))  # [batch, 256, H/4, W/4]
        x = self.global_pool(x)                  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [batch, 256]
        latent = self.fc(x)                      # [batch, latent_dim]
        return latent

# Example usage:
if __name__ == "__main__":
    # Suppose the environment state is represented by 5 channels:
    # Gas Temperature (T_g), Gas Density (n_g), Pressure (P), Metal Mask, Metal Temperature (T_m)
    # For demonstration, create a dummy batch of two samples.
    batch_size = 2
    height = 200
    width = 100
    # Create random example data for each channel.
    T_g = torch.rand(batch_size, height, width) * 20 + 290   # Temperatures around 290-310 K
    n_g = torch.rand(batch_size, height, width) * 0.2 + 0.9    # Densities around 0.9-1.1
    P   = torch.rand(batch_size, height, width) * 10 + 295     # Pressure around 295-305 (arbitrary units)
    metal_mask = torch.randint(0, 2, (batch_size, height, width)).float()
    T_m = torch.rand(batch_size, height, width) * 20 + 290

    # Stack the channels into a single tensor with shape [batch, channels, height, width]
    state = torch.stack([T_g, n_g, P, metal_mask, T_m], dim=1)
    
    # Instantiate the encoder and compute the latent vector.
    encoder = PhysicsEncoder(input_channels=5, latent_dim=256)
    latent_vector = encoder(state)
    print("Latent vector shape:", latent_vector.shape)  # Expected: [batch_size, 256]
