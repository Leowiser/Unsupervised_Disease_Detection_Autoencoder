import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HyperspectralCNN(nn.Module):
    def __init__(self, in_channels):
        super(HyperspectralCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # Downsample by 2
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample again
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),  # Further downsample
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.encoder(x)

class Autoencoder(nn.Module):
    def __init__(self, in_channels):
        super(Autoencoder, self).__init__()
        self.encoder = HyperspectralCNN(in_channels)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Upsample
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),  # Final output
            nn.Sigmoid()  # or nn.Identity() if you're not normalizing
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Example usage
    

'''
Model without fully connected layer, batch norm or max pool layer.
'''

class CNN3DAE(nn.Module):
    def __init__(self, layers_list=[32, 64, 128], input_dim=1, kernel_sizes=3, strides=(1, 2, 2), paddings=1):
        super(CNN3DAE, self).__init__()

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(
                nn.Conv3d(
                    in_channels, out_channels,
                    kernel_size=kernel_sizes,
                    stride=strides,
                    padding=paddings
                )
            )
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Decoder (reverse the channels i gave in encoder)
        decoder = []
        rev_layers = layers_list[::-1]
        for i, out_channels in enumerate(rev_layers[:-1]):
            decoder.append(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=kernel_sizes,
                    stride=strides,
                    padding=paddings,
                    output_padding=(0, 1, 1)  # output padding to match dims due to stride (1,2,2)
                )
            )
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer 
        decoder.append(
            nn.ConvTranspose3d(
                in_channels, input_dim,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=paddings,
                output_padding=(0, 1, 1)
            )
        )
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        out = x
        #print(f"Input: {out.shape}")
        for i, layer in enumerate(self.encoder):
            out = layer(out)
        for i, layer in enumerate(self.decoder):
            out = layer(out)
            #if isinstance(layer, nn.ConvTranspose3d):
                #print(f"After Decoder Layer {i//2}: {out.shape}")
        #print(f"Output: {out.shape}")
        return out
    def _encode(self, x):
        z = self.encoder(x)
        features = 0
        return features, z


class CNN3DAE_PC(nn.Module):
    def __init__(self, input_dim=1, dropout_p=0.2):
        super(CNN3DAE_PC, self).__init__()

        self.dropout_p = dropout_p
        self.stride = (1, 2, 2)
        self.kernel_size = (1, 2, 2)
        self.padding = (0, 0, 0)

        # === Encoder ===
        self.encoder_blocks = nn.ModuleList([
            self._conv_block(input_dim, 8),
            self._conv_block(8, 16),
            self._conv_block(16, 32)
        ])
        self.encoder_downsamples = nn.ModuleList([
            nn.Conv3d(8, 8, kernel_size=self.kernel_size, stride=self.stride),
            nn.Conv3d(16, 16, kernel_size=self.kernel_size, stride=self.stride),
            nn.Conv3d(32, 32, kernel_size=self.kernel_size, stride=self.stride)
        ])

        # === Decoder ===
        self.decoder_upsamples = nn.ModuleList([
            nn.ConvTranspose3d(32, 16, kernel_size=self.kernel_size, stride=self.stride),
            nn.ConvTranspose3d(16, 8, kernel_size=self.kernel_size, stride=self.stride),
            nn.ConvTranspose3d(8, 4, kernel_size=self.kernel_size, stride=self.stride)
        ])
        self.decoder_blocks = nn.ModuleList([
            self._conv_block(16, 16),
            self._conv_block(8, 8)
        ])
        self.final_conv = nn.Conv3d(4, input_dim, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(self.dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def encoder(self, x):
        for enc_block, down in zip(self.encoder_blocks, self.encoder_downsamples):
            x = enc_block(x)
            x = down(x)
        return x

    def decoder(self, x):
        x = self.decoder_upsamples[0](x)
        x = self.decoder_blocks[0](x)
        x = self.decoder_upsamples[1](x)
        x = self.decoder_blocks[1](x)
        x = self.decoder_upsamples[2](x)
        x = self.final_conv(x)
        return x

    def forward(self, x):
        x_rec = self.decoder(self.encoder(x))
        assert x.shape == x_rec.shape, f"Mismatch: input {x.shape}, output {x_rec.shape}"
        return x_rec

    def _encode(self, x):
        features = 0
        return features, self.encoder(x)
    
class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class Encoder3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc1 = ConvBlock3D(in_channels, 8)
        self.down1 = nn.Conv3d(8, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 256 → 128

        self.enc2 = ConvBlock3D(8, 16)
        self.down2 = nn.Conv3d(16, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 128 → 64

        self.enc3 = ConvBlock3D(16, 32)
        self.down3 = nn.Conv3d(32, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 64 → 32

    def forward(self, x):
        x = self.enc1(x)
        x = self.down1(x)
        x = self.enc2(x)
        x = self.down2(x)
        x = self.enc3(x)
        x = self.down3(x)
        return x

class Decoder3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 32 → 64
        self.dec1 = ConvBlock3D(16, 16)

        self.up2 = nn.ConvTranspose3d(16, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 64 → 128
        self.dec2 = ConvBlock3D(8, 8)

        self.up3 = nn.ConvTranspose3d(8, 4, kernel_size=(1, 2, 2), stride=(1, 2, 2))  # 128 → 256
        self.out = nn.Conv3d(4, 1, kernel_size=1)  # back to 1 channel

    def forward(self, x):
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

class Autoencoder3D(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = Encoder3D(in_channels)
        self.decoder = Decoder3D()

    def forward(self, x):
        x_rec = self.decoder(self.encoder(x))
        assert x.shape == x_rec.shape, f"Mismatch: input {x.shape}, output {x_rec.shape}"
        return x_rec


'''
Model with 3D convolutions, fully connected layers, max pool layers
'''

class CNN3DAEFCMP(nn.Module):
    def __init__(
        self,
        layers_list=[32, 64, 64, 128],
        input_dim=1,
        kernel_sizes=3,
        strides=(1, 2, 2),
        paddings=1,
        z_dim=128
    ):
        super(CNN3DAEFCMP, self).__init__()

        self.z_dim = z_dim
        self.layers_list = layers_list

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings))
            encoder.append(nn.MaxPool3d(kernel_sizes, stride=strides))
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Determine flattened feature shape after encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 17, 256, 256)
            dummy_output = self.encoder(dummy_input)
            self.feature_shape = dummy_output.shape[1:]  # (C, D, H, W)
            self.feature_dim = int(np.prod(self.feature_shape))

        # Fully Connected layers (bottleneck)
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_dim, z_dim),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(z_dim, self.feature_dim),
            nn.ReLU(inplace=True),
        )

        decoder = []
        rev_layers = layers_list[::-1]
        in_channels = rev_layers[0]
        for out_channels in rev_layers[1:]:
            decoder.append(nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=False))
            decoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer
        decoder.append(nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=False))
        decoder.append(nn.Conv3d(in_channels, input_dim, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*decoder)
        #self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        bs = x.size(0)
        out = self.encoder(x)
        out = out.view(bs, -1)  # Flatten
        z = self.fc1(out)
        out = self.fc2(z)
        out = out.view(bs, *self.feature_shape)
        out = self.decoder(out)
        return out

'''
Model with a maxpooling layer
'''
class CNN3DAEMAX_try(nn.Module):
    def __init__(
        self,
        layers_list=[32, 64, 64, 128],
        input_dim=1,
        kernel_sizes=3,
        strides=(1, 2, 2),
        paddings=1
    ):
        super(CNN3DAEMAX_try, self).__init__()
        self.layers_list = layers_list

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings))
            encoder.append(nn.MaxPool3d(kernel_sizes, stride=strides))
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Get output shape of encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 17, 256, 256)
            dummy_output = self.encoder(dummy_input)
            self.feature_shape = dummy_output.shape[1:]  # (C, D, H, W)

        # Decoder (reversed)
        decoder = []
        rev_layers = layers_list[::-1]
        in_channels = rev_layers[0]

        for out_channels in rev_layers[1:]:
            decoder.append(nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=paddings,
                output_padding=(0, 1, 1)  # helps recover 256 from downsampling
            ))
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to bring channels back to input_dim (e.g., 1)
        decoder.append(nn.ConvTranspose3d(
            in_channels,
            input_dim,
            kernel_size=kernel_sizes,
            stride=strides,
            padding=paddings,
            output_padding=(0, 1, 1)
        ))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = F.interpolate(out, size=(17, 256, 256), mode="trilinear", align_corners=False)
        return out
    
    def _encode(self, x):
        features = 0
        return features, self.encoder(x)



class CNN3DAEMAX_upsample(nn.Module):
    def __init__(
        self,
        layers_list=[32, 64, 64, 128],
        input_dim=1,
        kernel_sizes=4,
        strides=(1, 2, 2),
        paddings=1
    ):
        super(CNN3DAEMAX_upsample, self).__init__()
        self.layers_list = layers_list

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings))
            encoder.append(nn.MaxPool3d(kernel_size=kernel_sizes, stride=strides))
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Get output shape of encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 17, 256, 256)
            dummy_output = self.encoder(dummy_input)
            self.feature_shape = dummy_output.shape[1:]  # (C, D, H, W)

        # Decoder using upsampling instead of transposed convolutions
        decoder = []
        rev_layers = layers_list[::-1]
        in_channels = rev_layers[0]

        for out_channels in rev_layers[1:]:
            decoder.append(nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=False))
            decoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, padding=paddings))
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to bring back to input_dim (e.g., 1 channel)
        decoder.append(nn.Upsample(scale_factor=strides, mode='trilinear', align_corners=False))
        decoder.append(nn.Conv3d(in_channels, input_dim, kernel_size=kernel_sizes, padding=paddings))

        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        out = F.interpolate(out, size=(17, 256, 256), mode="trilinear", align_corners=False)
        return out

    def _encode(self, x):
        features = 0
        return features, self.encoder(x)
    

'''
Model with Fully connected layer and batch norm
'''

class CNN3DAEFC(nn.Module):
    def __init__(
        self,
        layers_list=[32, 64, 64, 128],
        input_dim=1,
        kernel_sizes=3,
        strides=(1, 2, 2),
        paddings=1,
        z_dim=128
    ):
        super(CNN3DAEFC, self).__init__()

        self.z_dim = z_dim
        self.layers_list = layers_list

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=strides, padding=paddings))
            encoder.append(nn.BatchNorm3d(out_channels))
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Determine flattened feature shape after encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 17, 256, 256)
            dummy_output = self.encoder(dummy_input)
            self.feature_shape = dummy_output.shape[1:]  # (C, D, H, W)
            self.feature_dim = int(np.prod(self.feature_shape))

        # Fully Connected layers (bottleneck)
        self.fc1 = nn.Sequential(
            nn.Linear(self.feature_dim, z_dim),
            nn.ReLU(inplace=True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(z_dim, self.feature_dim),
            nn.ReLU(inplace=True),
        )

        # Decoder
        decoder = []
        rev_layers = layers_list[::-1]
        in_channels = rev_layers[0]
        for out_channels in rev_layers[1:]:
            decoder.append(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=kernel_sizes,
                    stride=strides,
                    padding=paddings,
                    output_padding=(0, 1, 1)  # Note: hardcoded for (1,2,2) stride
                )
            )
            decoder.append(nn.BatchNorm3d(out_channels))
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final output layer
        decoder.append(
            nn.ConvTranspose3d(
                in_channels, input_dim,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=paddings,
                output_padding=(0, 1, 1)
            )
        )
        # Optional activation for final output
        # decoder.append(nn.Sigmoid())  # Uncomment if input is normalized
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        bs = x.size(0)
        out = self.encoder(x)
        out = out.view(bs, -1)  # Flatten
        z = self.fc1(out)
        out = self.fc2(z)
        out = out.view(bs, *self.feature_shape)
        out = self.decoder(out)
        return out
    
    def _encode(self, x):
        z = self.encoder(x)
        features = 0
        return features, z
    

'''
Class for 3D convolutions with MaxPool layers but without fully connected layers
'''

class CNN3DAEMAX(nn.Module):
    def __init__(
        self,
        layers_list=[32, 64, 64, 128],
        input_dim=1,
        kernel_sizes=3,
        strides=(1, 2, 2),
        paddings=1
    ):
        super(CNN3DAEMAX, self).__init__()
        self.layers_list = layers_list

        # Encoder
        encoder = []
        in_channels = input_dim
        for out_channels in layers_list:
            encoder.append(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_sizes, stride=1, padding=paddings))
            encoder.append(nn.MaxPool3d(kernel_sizes, stride=strides))
            encoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.encoder = nn.Sequential(*encoder)

        # Get output shape of encoder
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_dim, 17, 256, 256)
            dummy_output = self.encoder(dummy_input)
            self.feature_shape = dummy_output.shape[1:]  # (C, D, H, W)

        # Decoder (reversed)
        decoder = []
        rev_layers = layers_list[::-1]
        in_channels = rev_layers[0]

        for out_channels in rev_layers[1:]:
            decoder.append(nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=kernel_sizes,
                stride=strides,
                padding=paddings,
                output_padding=(0, 1, 1)  # helps recover 256 from downsampling
            ))
            decoder.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        # Final layer to bring channels back to input_dim (e.g., 1)
        decoder.append(nn.ConvTranspose3d(
            in_channels,
            input_dim,
            kernel_size=kernel_sizes,
            stride=strides,
            padding=paddings,
            output_padding=(0, 1, 1)
        ))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out
    
    def _encode(self, x):
        return self.encoder(x)

'''
Kernel size changes inbetween
'''
class CNN3DAE_VarKernels(nn.Module):
    def __init__(
        self,
        layers_list=[18, 32, 32, 64],
        input_dim=1,
        strides=(1, 2, 2),
        paddings=(0, 1, 1)
    ):
        super(CNN3DAE_VarKernels, self).__init__()
        self.layers_list = layers_list
        self.strides = strides
        self.paddings = paddings

        # Define varying kernel sizes for encoder
        self.encoder_kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)]
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]

        # --- Encoder ---
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.encoder_kernel_sizes):
            block = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=strides, padding=self.paddings),
                nn.ReLU(inplace=True)
            )
            self.enc_blocks.append(block)
            in_channels = out_channels

        # --- Decoder ---
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for idx, (in_channels, out_channels, ks) in enumerate(zip(rev_layers, rev_layers[1:], self.decoder_kernel_sizes[1:])):
            block = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels, out_channels,
                    kernel_size=ks, stride=strides,
                    padding=self.paddings, output_padding=(0, 1, 1)
                ),
                nn.ReLU(inplace=True)
            )
            self.dec_blocks.append(block)

        # Final output layer
        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=self.decoder_kernel_sizes[-1], stride=strides,
            padding=self.paddings, output_padding=(0, 1, 1)
        )

    def encoder(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        return features, x

    def forward(self, x):
        enc_feats,_ = self.encoder(x)
        x = enc_feats[-1]

        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:  # last decoder block before final layer
                x = x + F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)

        x = self.final_layer(x)

        # Resize to original input shape
        x = F.interpolate(x, size=(17, 256, 256), mode='trilinear', align_corners=False)
        return x

    def _encode(self, x):
        features, z = self.encoder(x)
        return features, z

'''
Model with dropout in the latent space.
Variational kernel size
'''
class CNN3DAE_TightDropout(nn.Module):
    def __init__(
        self,
        layers_list=[18, 32, 32, 64],
        input_dim=1,
        strides=(1, 2, 2),
        paddings=(0, 1, 1),
        dropout_p=0.2
    ):
        super(CNN3DAE_TightDropout, self).__init__()

        self.strides = strides
        self.paddings = paddings
        self.dropout_p = dropout_p

        # Kernel sizes per encoder layer
        self.encoder_kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)]
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.encoder_kernel_sizes):
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=strides, padding=paddings),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        # Extra encoder bottleneck block for tighter latent space
        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=strides, padding=paddings),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # === Decoder ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, ks in zip(rev_layers, rev_layers[1:], self.decoder_kernel_sizes[1:]):
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=ks, stride=strides,
                        padding=paddings, output_padding=(0, 1, 1)
                    ),
                    nn.ReLU(inplace=True)
                )
            )

        # Extra decoder block to match bottleneck layer
        self.bottleneck_decoder = nn.Sequential(
            nn.ConvTranspose3d(
                rev_layers[0], rev_layers[0],
                kernel_size=(3, 3, 3), stride=strides,
                padding=paddings, output_padding=(0, 1, 1)
            ),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=self.decoder_kernel_sizes[-1], stride=strides,
            padding=paddings, output_padding=(0, 1, 1)
        )

    def encoder(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x

    def forward(self, x):
        enc_feats, x = self.encoder(x)
        x = self.bottleneck_decoder(x)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:
                # Final skip connection (after full spatial upsampling)
                skip = F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)
                x = x + skip

        x = self.final_layer(x)
        x = F.interpolate(x, size=(17, 256, 256), mode='trilinear', align_corners=False)
        return x
    
    def _encode(self, x):
        features, z = self.encoder(x)
        return features, z

'''Strict Model with two dropout layers'''
class CNN3DAE_Strict(nn.Module):
    def __init__(
        self,
        layers_list=[18, 32, 32, 64],
        input_dim=1,
        strides=(1, 2, 2),
        paddings=(0, 1, 1),
        dropout_p=0.2
    ):
        super(CNN3DAE_Strict, self).__init__()

        self.strides = strides
        self.paddings = paddings
        self.dropout_p = dropout_p

        self.encoder_kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3)]
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.encoder_kernel_sizes):
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=strides, padding=paddings),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        # === Bottleneck Encoder ===
        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=strides, padding=paddings),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=strides, padding=paddings),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # === Decoder ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, ks in zip(rev_layers, rev_layers[1:], self.decoder_kernel_sizes[1:]):
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=ks, stride=strides,
                        padding=paddings, output_padding=(0, 1, 1)
                    ),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # === Bottleneck Decoder ===
        self.bottleneck_decoder = nn.Sequential(
            nn.ConvTranspose3d(
                rev_layers[0], rev_layers[0],
                kernel_size=(3, 3, 3), stride=strides,
                padding=paddings, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(rev_layers[0]),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(
                rev_layers[0], rev_layers[0],
                kernel_size=(3, 3, 3), stride=strides,
                padding=paddings, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(rev_layers[0]),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=self.decoder_kernel_sizes[-1], stride=strides,
            padding=paddings, output_padding=(0, 1, 1)
        )

    def _encode(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x

    def forward(self, x):
        enc_feats, x = self._encode(x)
        x = self.bottleneck_decoder(x)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:
                skip = F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)
                x = x + skip

        x = self.final_layer(x)
        x = F.interpolate(x, size=(17, 256, 256), mode='trilinear', align_corners=False)
        return x
    

'''
4 Layers and Maxpool no sigmoid
'''
class AE3DCNN_4Layer_128_3Channel(nn.Module):
    def __init__(self):
        super(AE3DCNN_4Layer_128_3Channel, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),   # <- changed from 1 to 3 channels
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),

            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2))
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),

            nn.ConvTranspose3d(16, 3, kernel_size=(4, 2, 2), stride=(1, 2, 2), padding=(2, 0, 0)),  # <- output 3 channels
            #nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = F.interpolate(decoded, size=(x.shape[2], x.shape[3], x.shape[4]), mode='trilinear', align_corners=False)
        return decoded
    
    def _encode(self, x):
        x = self.encoder(x)
        return 0, x


'''
5 layers and downsampling to a smaller latent space.
Using BatchNorm.
'''
class CNN3DAE_5Layer_Depth2(nn.Module):
    def __init__(
        self,
        layers_list=[8, 16, 32, 64, 128],
        input_dim=1,
        latent_dim=4,
        dropout_p=0.2
    ):
        super(CNN3DAE_5Layer_Depth2, self).__init__()

        self.strides = (2, 2, 2)
        self.paddings = (1, 1, 1)
        self.kernel_sizes = [(3, 3, 3)] * 5

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.kernel_sizes):
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=self.strides, padding=self.paddings),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        # === Bottleneck ===
        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_channels, latent_dim, kernel_size=1),
            nn.BatchNorm3d(latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        self.bottleneck_decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_dim, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # === Decoder ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(rev_layers, rev_layers[1:]):
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=(3, 3, 3), stride=self.strides,
                        padding=self.paddings, output_padding=1
                    ),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                rev_layers[-1], input_dim,
                kernel_size=(3, 3, 3), stride=self.strides,
                padding=self.paddings, output_padding=1
            )
            # No BatchNorm or ReLU here — final output layer
        )

    def _encode(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x

    def forward(self, x):
        enc_feats, x = self._encode(x)
        x = self.bottleneck_decoder(x)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:
                skip = F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)
                x = x + skip

        x = self.final_layer(x)
        x = F.interpolate(x, size=(18, 256, 256), mode='trilinear', align_corners=False)
        return x
    



'''
Model with dropout downsampling strides and all bands snv transformed
'''
class CNN3DAE_TightDropout_snv(nn.Module):
    def __init__(
        self,
        layers_list=[16, 32, 64, 128],
        input_dim=1,
        strides=(2, 2, 2),
        paddings=(1, 1, 1),
        dropout_p=0.3,
        latent_channels=16
    ):
        super(CNN3DAE_TightDropout_snv, self).__init__()

        self.strides = strides
        self.paddings = paddings
        self.dropout_p = dropout_p

        # Uniform kernel sizes
        self.encoder_kernel_sizes = [(3, 3, 3)] * len(layers_list)
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.encoder_kernel_sizes):
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=strides, padding=paddings),
                    nn.BatchNorm3d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        # Bottleneck encoder → latent
        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_channels, latent_channels, kernel_size=1),
            nn.BatchNorm3d(latent_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # Bottleneck decoder → back to feature depth
        self.bottleneck_decoder = nn.Sequential(
            nn.ConvTranspose3d(latent_channels, in_channels, kernel_size=1),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )

        # === Decoder ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, ks in zip(rev_layers, rev_layers[1:], self.decoder_kernel_sizes[1:]):
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=ks, stride=strides,
                        padding=paddings, output_padding=1
                    ),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )
            )

        # Final output layer
        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=self.decoder_kernel_sizes[-1], stride=strides,
            padding=paddings, output_padding=1
        )

    def _encode(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x  

    def forward(self, x):
        _, x = self._encode(x)
        x = self.bottleneck_decoder(x)

        for block in self.dec_blocks:
            x = block(x)

        x = self.final_layer(x)
        x = F.interpolate(x, size=(210, 256, 256), mode='trilinear', align_corners=False)
        return x


'''
Denoises and zooms into the images.
zooms by 80%
noises with  Gaussian Noise Injection
'''
class CNN3DAE_Denoising(nn.Module):
    def __init__(
        self,
        input_dim=1,
        layers_list=[16, 32, 32, 64],
        strides=(1, 2, 2),
        paddings=(0, 1, 1),
        dropout_p=0.3,
        use_batchnorm=True,
        residual=True
    ):
        super(CNN3DAE_Denoising, self).__init__()
        self.use_batchnorm = use_batchnorm
        self.dropout_p = dropout_p
        self.residual = residual

        self.encoder_kernel_sizes = [(1, 3, 3)] * 2 + [(3, 3, 3)] * 2
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for out_channels, ks in zip(layers_list, self.encoder_kernel_sizes):
            block = [
                nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=strides, padding=paddings)
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm3d(out_channels))
            block.append(nn.ReLU(inplace=True))
            self.enc_blocks.append(nn.Sequential(*block))
            in_channels = out_channels

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=strides, padding=paddings),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # Latent space projection (tight bottleneck)
        self.latent_proj = nn.Sequential(
            nn.Conv3d(64, 16, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # First decoder layer after latent projection
        self.latent_decoder = nn.Sequential(
            nn.ConvTranspose3d(
                16, 64,
                kernel_size=self.decoder_kernel_sizes[0], stride=strides,
                padding=paddings, output_padding=(0, 1, 1)
            ),
            nn.BatchNorm3d(64) if use_batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

        # === Remaining Decoder ===
        rev_layers = layers_list[::-1]
        decoder_in_channels = rev_layers[:-1]  # because first decode is now separate
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, ks in zip(decoder_in_channels, rev_layers[1:], self.decoder_kernel_sizes[1:]):
            block = [
                nn.ConvTranspose3d(
                    in_ch, out_ch,
                    kernel_size=ks, stride=strides,
                    padding=paddings, output_padding=(0, 1, 1)
                )
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm3d(out_ch))
            block.append(nn.ReLU(inplace=True))
            self.dec_blocks.append(nn.Sequential(*block))

        # Final decoder layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose3d(
                rev_layers[-1], input_dim,
                kernel_size=self.decoder_kernel_sizes[-1], stride=strides,
                padding=paddings, output_padding=(0, 1, 1)
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc_feats = []
        for block in self.enc_blocks:
            x = block(x)
            enc_feats.append(x)

        x = self.bottleneck(x)
        x = self.latent_proj(x)
        x = self.latent_decoder(x)

        for i, block in enumerate(self.dec_blocks):
            x = block(x)
            if i == len(self.dec_blocks) - 1 and self.residual:
                x = x + F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)

        x = self.final_layer(x)
        x = F.interpolate(x, size=(17, 256, 256), mode='trilinear', align_corners=False)
        return x

    def _encode(self, x):
        for block in self.enc_blocks:
            x = block(x)
        x = self.bottleneck(x)
        x = self.latent_proj(x)
        features = 0
        return features, x


'''
Model used for the fine tuning with the tight dropout basic
'''
class CNN3DAE_TightDropout_finetuning(nn.Module):
    def __init__(
        self,
        layers_list=[18, 32, 32, 64],
        input_dim=1,
        strides=[(1, 2, 2)] * 4,
        dropout_p=0.2
    ):
        super(CNN3DAE_TightDropout_finetuning, self).__init__()
        self.dropout_p = dropout_p

        def get_padding(kernel_size, stride):
            return tuple((k - s) // 2 for k, s in zip(kernel_size, stride))

        self.encoder_kernel_sizes = [(3, 3, 3)] * len(layers_list)
        self.decoder_kernel_sizes = self.encoder_kernel_sizes[::-1]
        self.encoder_strides = strides
        self.decoder_strides = strides[::-1]

        # === Encoder ===
        self.enc_blocks = nn.ModuleList()
        in_channels = input_dim
        for i, out_channels in enumerate(layers_list):
            ks = self.encoder_kernel_sizes[i]
            stride = self.encoder_strides[i]
            padding = get_padding(ks, stride)
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=ks, stride=stride, padding=padding),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 3, 3), stride=self.encoder_strides[-1], padding=get_padding((3,3,3), self.encoder_strides[-1])),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # === Decoder ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(rev_layers, rev_layers[1:])):
            ks = self.decoder_kernel_sizes[i+1]
            stride = self.decoder_strides[i]
            padding = get_padding(ks, stride)
            output_padding = tuple(s - 1 for s in stride)
            self.dec_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose3d(
                        in_ch, out_ch,
                        kernel_size=ks, stride=stride,
                        padding=padding, output_padding=output_padding
                    ),
                    nn.ReLU(inplace=True)
                )
            )

        self.bottleneck_decoder = nn.Sequential(
            nn.ConvTranspose3d(
                rev_layers[0], rev_layers[0],
                kernel_size=(3, 3, 3), stride=self.decoder_strides[0],
                padding=get_padding((3,3,3), self.decoder_strides[0]), output_padding=tuple(s - 1 for s in self.decoder_strides[0])
            ),
            nn.ReLU(inplace=True)
        )

        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=self.decoder_kernel_sizes[-1], stride=self.decoder_strides[-1],
            padding=get_padding(self.decoder_kernel_sizes[-1], self.decoder_strides[-1]),
            output_padding=tuple(s - 1 for s in self.decoder_strides[-1])
        )

    def _encode(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x

    def forward(self, x):
        enc_feats, x = self._encode(x)
        x = self.bottleneck_decoder(x)

        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:
                skip = F.interpolate(enc_feats[0], size=x.shape[2:], mode="trilinear", align_corners=False)
                x = x + skip

        x = self.final_layer(x)
        x = F.interpolate(x, size=(18, 256, 256), mode='trilinear', align_corners=False)
        return x
    

class CNN3DAE_finetuning(nn.Module):
    def __init__(
        self,
        layers_list,
        input_dim=1,
        strides=None,
        dropout_p=0.2,
        pool_type="none",
        use_batchnorm=None,
        activation="relu"
    ):
        super().__init__()
        self.dropout_p = dropout_p
        self.pool_type = pool_type

        # select activation class
        if activation == "relu":
            Act = nn.ReLU
        elif activation == "gelu":
            Act = nn.GELU
        else:
            Act = nn.ELU

        n_layers = len(layers_list)
        strides = strides or [(1,2,2)] * n_layers
        use_batchnorm = use_batchnorm or [False] * n_layers

        # === Encoder blocks (conv -> [BN] -> Act) ===
        self.enc_blocks = nn.ModuleList()
        in_ch = input_dim
        for i, out_ch in enumerate(layers_list):
            ks, stride = (3,3,3), strides[i]
            pad = tuple((k - s)//2 for k, s in zip(ks, stride))
            block = [
                nn.Conv3d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad)
            ]
            if use_batchnorm[i]:
                block.append(nn.BatchNorm3d(out_ch))
            block.append(Act(inplace=True) if activation == "relu" else Act())
            self.enc_blocks.append(nn.Sequential(*block))
            in_ch = out_ch

        # === Bottleneck Encoder (no further downsampling) ===
        ks, stride = (3,3,3), (1,1,1)
        pad = tuple((k - 1)//2 for k in ks)
        self.bottleneck_encoder = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=ks, stride=stride, padding=pad),
            Act(inplace=True) if activation == "relu" else Act(),
            nn.Dropout3d(p=dropout_p)
        )

        # === Decoder blocks (ConvTranspose -> Act -> [Upsample]) ===
        rev_layers = layers_list[::-1]
        self.dec_blocks = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(rev_layers, rev_layers[1:])):
            ks, stride = (3,3,3), strides[::-1][i]
            pad = tuple((k - s)//2 for k, s in zip(ks, stride))
            out_pad = tuple(s - 1 for s in stride)
            block = [
                nn.ConvTranspose3d(
                    in_ch, out_ch,
                    kernel_size=ks, stride=stride,
                    padding=pad, output_padding=out_pad
                ),
                Act(inplace=True) if activation == "relu" else Act()
            ]
            if pool_type == "max":
                block.append(nn.Upsample(scale_factor=(1,2,2), mode='nearest'))
            elif pool_type == "avg":
                block.append(nn.Upsample(
                    scale_factor=(1,2,2), mode='trilinear', align_corners=False
                ))
            self.dec_blocks.append(nn.Sequential(*block))

        # === Bottleneck Decoder (mirror encoder, no upsample for conv-stride) ===
        ks, stride = (3,3,3), (1,1,1)
        pad = tuple((k - 1)//2 for k in ks)
        bottleneck = [
            nn.ConvTranspose3d(
                rev_layers[0], rev_layers[0],
                kernel_size=ks, stride=stride,
                padding=pad, output_padding=(0,0,0)
            ),
            Act(inplace=True) if activation == "relu" else Act()
        ]
        # no upsample here since stride is 1
        self.bottleneck_decoder = nn.Sequential(*bottleneck)

        # === Final reconstruction layer ===
        ks, stride = (3,3,3), strides[-1]
        pad = tuple((k - s)//2 for k, s in zip(ks, stride))
        out_pad = tuple(s - 1 for s in stride)
        self.final_layer = nn.ConvTranspose3d(
            rev_layers[-1], input_dim,
            kernel_size=ks, stride=stride,
            padding=pad, output_padding=out_pad
        )

    def _encode(self, x):
        features = []
        for block in self.enc_blocks:
            x = block(x)
            if self.pool_type != "none":
                N, C, T, H, W = x.shape
                if H >= 2 and W >= 2:
                    if self.pool_type == "max":
                        x = F.max_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
                    else:
                        x = F.avg_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
            features.append(x)
        x = self.bottleneck_encoder(x)
        return features, x

    def forward(self, x):
        enc_feats, x = self._encode(x)
        x = self.bottleneck_decoder(x)
        for idx, block in enumerate(self.dec_blocks):
            x = block(x)
            if idx == len(self.dec_blocks) - 1:
                skip = F.interpolate(
                    enc_feats[0], size=x.shape[2:], mode='trilinear', align_corners=False
                )
                x = x + skip
        x = self.final_layer(x)
        x = F.interpolate(x, size=(17, 256, 256), mode='trilinear', align_corners=False)
        return x



class CNN3DAE_Exact_Configurable(nn.Module):
    def __init__(
        self,
        input_dim=1,
        base_channels=[32, 64, 128],
        dropout_p=0.2,
        pool_type="none",       # "none", "max", or "avg"
        use_batchnorm=False     # bool or list of bools
    ):
        super().__init__()
        self.pool_type = pool_type
        self.dropout_p = dropout_p

        # decide downsample stride for convs
        #  - if no pooling, convs do (1,2,2) downsample
        #  - if pooling, convs keep stride=1 and pooling does the downsample
        if pool_type == "none":
            conv_stride = (1, 2, 2)
        else:
            conv_stride = (1, 1, 1)

        conv_kernel = 3
        conv_padding = 1

        # normalize batchnorm flag to list
        if isinstance(use_batchnorm, bool):
            use_batchnorm = [use_batchnorm] * len(base_channels)

        # === Encoder ===
        self.encoder = nn.ModuleList()
        in_ch = input_dim
        for i, out_ch in enumerate(base_channels):
            layers = [
                nn.Conv3d(in_ch, out_ch,
                          kernel_size=conv_kernel,
                          stride=conv_stride,
                          padding=conv_padding)
            ]
            if use_batchnorm[i]:
                layers.append(nn.BatchNorm3d(out_ch))
            layers += [
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p)
            ]
            self.encoder.append(nn.Sequential(*layers))
            in_ch = out_ch

        # === Bottleneck ===
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_p)
        )

        # decide upsample scale
        #  - if convs downsample, we upsample same way
        #  - if pooling downsampled, upsample by (1,2,2)
        if pool_type == "none":
            up_scale = conv_stride
        else:
            up_scale = (1, 2, 2)

        # === Decoder ===
        self.decoder = nn.ModuleList()
        rev_channels = base_channels[::-1]
        in_ch = rev_channels[0]
        for out_ch in rev_channels[1:]:
            self.decoder.append(nn.Sequential(
                nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch,
                          kernel_size=conv_kernel,
                          stride=1,
                          padding=conv_padding),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout_p)
            ))
            in_ch = out_ch

        # === Final reconstruction layer ===
        self.final_block = nn.Sequential(
            nn.Upsample(scale_factor=up_scale, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, input_dim,
                      kernel_size=conv_kernel,
                      stride=1,
                      padding=conv_padding)
        )

    def _maybe_pool(self, x):
        if self.pool_type == "max":
            return F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        elif self.pool_type == "avg":
            return F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return x

    def forward(self, x):
        orig_shape = x.shape[2:]  # (18, 256, 256)

        # Encoder
        for block in self.encoder:
            x = block(x)
            x = self._maybe_pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for block in self.decoder:
            x = block(x)

        # Final
        x = self.final_block(x)

        assert x.shape[2:] == orig_shape, (
            f"Output shape {x.shape[2:]} != input shape {orig_shape}"
        )
        return x