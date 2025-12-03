import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

from escnn import gspaces
from escnn import nn as enn
from escnn import group
from escnn.gspaces.r2 import GSpace2D

from fvf.model.modules.layers import ResNetBlock
from fvf.model.modules.equiv_layers import CyclicResNetBlock, SO2ResNetBlock
from fvf.model.modules.fourier import Fourier

class ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout):
        super().__init__()

        self.conv = nn.Sequential(
            # 84x84
            nn.Conv2d(in_channels, z_dim // 8, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            # 80x80
            ResNetBlock(z_dim // 8, z_dim // 8),
            ResNetBlock(z_dim // 8, z_dim // 8),
            nn.MaxPool2d(2),
            # 40x40
            ResNetBlock(z_dim // 8, z_dim // 4),
            ResNetBlock(z_dim // 4, z_dim // 4),
            nn.MaxPool2d(2),
            # 20x20
            ResNetBlock(z_dim // 4, z_dim // 2),
            ResNetBlock(z_dim // 2, z_dim // 2),
            nn.MaxPool2d(2),
            # 10x10
            ResNetBlock(z_dim // 2, z_dim),
            ResNetBlock(z_dim, z_dim),
            nn.MaxPool2d(2),
            # 5x5
            nn.Conv2d(z_dim, z_dim, kernel_size=5, padding=0),
            nn.ReLU(inplace=True)
            # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class ImageEncoder2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        z_dim: int,
        dropout: float,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        norm_input: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm_input = norm_input

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.channel_adapter = None
        if in_channels != 3:
            self.channel_adapter = nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)

        if self.norm_input:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("imgnet_mean", mean)
            self.register_buffer("imgnet_std", std)

        self.head = nn.Sequential(
            nn.Linear(512, z_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def _maybe_adapt_and_norm(self, x: torch.Tensor) -> torch.Tensor:
       
        if self.channel_adapter is not None:
            x = self.channel_adapter(x)

        if self.norm_input:

            x = torch.clamp(x, 0.0, 1.0)
            x = (x - self.imgnet_mean) / self.imgnet_std

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self._maybe_adapt_and_norm(x)
        feats_512 = self.backbone(x)      # [B, 512]（ResNet avgpool + flatten）
        z = self.head(feats_512)          # [B, z_dim]
        return z.view(z.size(0), z.size(1), 1, 1)



class CyclicImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout, lmax=3, N=16, initialize=True):
        super().__init__()
        self.cyclic = gspaces.rot2dOnR2(N)

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.z_dim = z_dim

        self.in_type = enn.FieldType(
            self.cyclic,
            [self.cyclic.trivial_repr] * in_channels
        )
        self.out_type = enn.FieldType(
            self.cyclic,
            z_dim * [self.cyclic.regular_repr]
        )

        layers = list()
        # 84x84
        layers.append(
            enn.R2Conv(
                self.in_type,
                enn.FieldType(self.cyclic, z_dim // 8 * [self.cyclic.regular_repr]),
                kernel_size=5,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(
            enn.ReLU(
                enn.FieldType(self.cyclic, z_dim // 8 * [self.cyclic.regular_repr]),
                inplace=True
            )
        )
        # 80x80
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 8, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 8, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 40x40
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 4, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 4, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 20x20
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 2, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim // 2, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 10x10
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N, initialize=initialize))
        layers.append(CyclicResNetBlock(layers[-1].out_type, z_dim, N=N, initialize=initialize))
        layers.append(enn.PointwiseMaxPool(layers[-1].out_type, 2))
        # 5x5
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                self.out_type,
                kernel_size=5,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(enn.ReLU(self.out_type, inplace=True))
        # 1x1

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv(x).tensor

        return out.view(B, self.z_dim, -1)

class SO2ImageEncoder(nn.Module):
    def __init__(self, in_channels, z_dim, dropout, lmax=3, N=16, initialize=True):
        super().__init__()
        self.G = group.so2_group()
        self.gspace = GSpace2D((None, -1), lmax)

        self.in_type = enn.FieldType(
            self.gspace,
            [self.gspace.trivial_repr] * in_channels
        )

        layers = list()
        # 96x96
        layers.append(SO2ResNetBlock(self.in_type, z_dim // 8, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 48x48
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim // 4, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 24x24
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim // 2, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 12x12
        layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim, lmax=lmax, N=N, initialize=initialize))
        layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # # 6x6
        # layers.append(SO2ResNetBlock(layers[-1].out_type, z_dim, lmax=lmax, N=N, initialize=initialize))
        # layers.append(enn.NormMaxPool(layers[-1].out_type, 2))
        # 3x3
        act = enn.FourierELU(
            self.gspace,
            channels=z_dim,
            irreps=self.G.bl_irreps(L=lmax),
            inplace=True,
            type="regular",
            N=N,
        )
        layers.append(
            enn.R2Conv(
                layers[-1].out_type,
                act.in_type,
                # kernel_size=3,
                kernel_size=5,
                padding=0,
                initialize=initialize
            )
        )
        layers.append(act)
        # 1x1

        self.out_type = layers[-1].out_type
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B = x.size(0)
        x = enn.GeometricTensor(x, self.in_type)
        out = self.conv(x)

        return out.tensor

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet