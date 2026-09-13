import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions,
    each followed by BatchNorm and ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """
    Basic U-Net.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.

    out_channels : int
        Number of output classes/maps.
    """

    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec1 = DoubleConv(128, 64)

        # Final prediction
        self.output = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):

        # Encoder
        e1 = self.enc1(x)

        e2 = self.enc2(
            self.pool(e1)
        )

        e3 = self.enc3(
            self.pool(e2)
        )

        e4 = self.enc4(
            self.pool(e3)
        )

        # Bottleneck
        b = self.bottleneck(
            self.pool(e4)
        )

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.output(d1)

class UNetNoSkips(nn.Module):
    """
    U-Net-like encoder-decoder without skip connections.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConv(
            in_channels, 64
        )
        self.enc2 = DoubleConv(
            64, 128
        )
        self.enc3 = DoubleConv(
            128, 256
        )
        self.enc4 = DoubleConv(
            256, 512
        )

        # Bottleneck
        self.bottleneck = DoubleConv(
            512, 1024
        )

        # Decoder -- NO concatenation with encoder
        self.up4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2
        )
        self.dec4 = DoubleConv(
            512, 512
        )

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )
        self.dec3 = DoubleConv(
            256, 256
        )

        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )
        self.dec2 = DoubleConv(
            128, 128
        )

        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )
        self.dec1 = DoubleConv(
            64, 64
        )

        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(
            self.pool(x1)
        )
        x3 = self.enc3(
            self.pool(x2)
        )
        x4 = self.enc4(
            self.pool(x3)
        )

        x5 = self.bottleneck(
            self.pool(x4)
        )

        x = self.up4(x5)
        x = self.dec4(x)

        x = self.up3(x)
        x = self.dec3(x)

        x = self.up2(x)
        x = self.dec2(x)

        x = self.up1(x)
        x = self.dec1(x)

        return self.final(x)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling.
    Preserves spatial resolution and channel count.
    """

    def __init__(
        self,
        in_channels=1024,
        out_channels=1024
    ):
        super().__init__()

        branch_channels = (
            out_channels // 4
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=2,
                dilation=2
            ),
            nn.BatchNorm2d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=4,
                dilation=4
            ),
            nn.BatchNorm2d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                branch_channels,
                kernel_size=3,
                padding=8,
                dilation=8
            ),
            nn.BatchNorm2d(
                branch_channels
            ),
            nn.ReLU(inplace=True)
        )

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1
            ),
            nn.BatchNorm2d(
                out_channels
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat(
            [x1, x2, x3, x4],
            dim=1
        )

        return self.project(x)


class UNetASPP(nn.Module):
    """
    Standard U-Net with ASPP after the bottleneck.
    Skip connections are preserved.
    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.enc1 = DoubleConv(
            in_channels, 64
        )
        self.enc2 = DoubleConv(
            64, 128
        )
        self.enc3 = DoubleConv(
            128, 256
        )
        self.enc4 = DoubleConv(
            256, 512
        )

        self.bottleneck = DoubleConv(
            512, 1024
        )

        self.aspp = ASPP(
            in_channels=1024,
            out_channels=1024
        )

        # Standard U-Net decoder with skips
        self.up4 = nn.ConvTranspose2d(
            1024, 512, 2, 2
        )
        self.dec4 = DoubleConv(
            1024, 512
        )

        self.up3 = nn.ConvTranspose2d(
            512, 256, 2, 2
        )
        self.dec3 = DoubleConv(
            512, 256
        )

        self.up2 = nn.ConvTranspose2d(
            256, 128, 2, 2
        )
        self.dec2 = DoubleConv(
            256, 128
        )

        self.up1 = nn.ConvTranspose2d(
            128, 64, 2, 2
        )
        self.dec1 = DoubleConv(
            128, 64
        )

        self.final = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.enc1(x)
        x2 = self.enc2(
            self.pool(x1)
        )
        x3 = self.enc3(
            self.pool(x2)
        )
        x4 = self.enc4(
            self.pool(x3)
        )

        x5 = self.bottleneck(
            self.pool(x4)
        )

        x5 = self.aspp(x5)

        x = self.up4(x5)
        x = torch.cat(
            [x, x4],
            dim=1
        )
        x = self.dec4(x)

        x = self.up3(x)
        x = torch.cat(
            [x, x3],
            dim=1
        )
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat(
            [x, x2],
            dim=1
        )
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat(
            [x, x1],
            dim=1
        )
        x = self.dec1(x)

        return self.final(x)


# ------------------------------------------------------------------
# Architecture registry and checkpoint loading
# ------------------------------------------------------------------
#
# Every script that needs a trained model goes through
# load_model_from_checkpoint, so the architecture is always rebuilt
# from what the checkpoint itself recorded. That keeps the ablation
# honest: there is no place left where a checkpoint can be loaded
# into the wrong decoder.

ARCHITECTURES = {
    "unet": UNet,
    "no_skips": UNetNoSkips,
    "aspp": UNetASPP,
}


def build_architecture(
    architecture,
    in_channels=3,
    out_channels=3,
):
    """
    Instantiate one of the ablation architectures by name.
    """

    if architecture not in ARCHITECTURES:
        raise ValueError(
            f"Unknown architecture: {architecture!r}. "
            f"Available: {sorted(ARCHITECTURES)}"
        )

    return ARCHITECTURES[architecture](
        in_channels=in_channels,
        out_channels=out_channels,
    )


def load_model_from_checkpoint(
    checkpoint_path,
    device,
    in_channels=3,
    out_channels=3,
):
    """
    Rebuild the correct architecture from a checkpoint and load its
    weights, already moved to `device` and in eval mode.

    Returns
    -------
    model, checkpoint
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    architecture = checkpoint.get(
        "architecture",
        "unet",
    )

    model = build_architecture(
        architecture,
        in_channels=in_channels,
        out_channels=out_channels,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model, checkpoint
