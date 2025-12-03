import torch
import torch.nn as nn

from escnn import gspaces
from escnn import nn as enn
from escnn import group
from escnn.gspaces.r2 import GSpace2D

from fvf.model.modules.vision_encoder import ImageEncoder, CyclicImageEncoder, SO2ImageEncoder, ImageEncoder2
from fvf.model.modules.fourier import Fourier
from fvf.model.modules.equiv_layers import SO2MLP, SO3MLP
from fvf.model.modules.layers import MLP


class KeypointEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        in_feat: int,
        z_dim: int,
        num_layers: int = 4,
        dropout: float = 0.0,
        spec_norm: bool = False,
        initialize: bool = True,
    ):
        super().__init__()

        self.lin = MLP(
            [num_obs * in_feat] + [z_dim] * num_layers,
            dropout=dropout,
            act_out=True,
            spec_norm=spec_norm,
        )

    # def forward(self, obs) -> torch.Tensor:
    #     B, _, _ = obs["keypoints"].shape

    #     x = obs["keypoints"].reshape(B, -1)
    #     obs_feat = self.lin(x)

    #     return obs_feat

    def forward(self, obs) -> torch.Tensor:
        keypoints = obs["keypoints"]
        B = keypoints.shape[0]
        x = keypoints.reshape(B, -1)
        obs_feat = self.lin(x)
        return obs_feat


class ObsEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        img_channels: int,
        z_dim: int,
        dropout: float = 0.0,
        spec_norm: bool = False,
        initialize: bool = True,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder(img_channels, z_dim, dropout)
        self.lin = MLP(
            [num_obs * z_dim + num_obs * 2] + [z_dim],
            dropout=dropout,
            act_out=True,
            spec_norm=False,
        )

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs["image"].shape

        img_feat = self.image_encoder(obs["image"].view(B * T, C, H, W)).view(B, -1)
        state = obs["agent_pos"].view(B, -1)

        img_feat_state = torch.cat([img_feat, state], dim=-1)
        obs_feat = self.lin(img_feat_state)

        return obs_feat

class ObsEncoder2(nn.Module):
    def __init__(
        self,
        num_obs: int,
        img_channels: int,
        z_dim: int,
        dropout: float = 0.0,
        spec_norm: bool = False,
        initialize: bool = True,
    ):
        super().__init__()

        self.image_encoder = ImageEncoder2(img_channels, z_dim, dropout)
        self.lin = MLP(
            [num_obs * z_dim + num_obs * 2] + [z_dim],
            dropout=dropout,
            act_out=True,
            spec_norm=False,
        )

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs["image"].shape

        img_feat = self.image_encoder(obs["image"].view(B * T, C, H, W)).view(B, -1)
        state = obs["agent_pos"].view(B, -1)

        img_feat_state = torch.cat([img_feat, state], dim=-1)
        obs_feat = self.lin(img_feat_state)

        return obs_feat


class SO3KeypointEncoder(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        num_obs: int,
        in_feat: int,
        z_dim: int,
        num_layers: int,
        lmax: int = 3,
        N: int = 16,
        dropout: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()

        self.G = group.so3_group(lmax)
        self.gspace = gspaces.no_base_space(self.G)
        self.num_layers = num_layers
        self.z_dim = z_dim

        keypoint_type = [self.gspace.irrep(1)]
        obs_type = num_keypoints * keypoint_type
        self.in_type = enn.FieldType(self.gspace, num_obs * obs_type)
        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))
        self.out_type = enn.FieldType(self.gspace, z_dim * [rho])
        self.keypoint_enc = SO3MLP(
            self.in_type,
            channels=[z_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            out_type=self.out_type,
            N=N,
            dropout=dropout,
            act_out=False,
            initialize=initialize,
        )

    def forward(self, obs) -> torch.Tensor:
        B, T, Do = obs["keypoints"].shape
        keypoints = obs["keypoints"]

        x = self.in_type(keypoints.reshape(B, -1))
        obs_feat = self.keypoint_enc(x)

        return obs_feat.tensor


class SO2KeypointEncoder2(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        num_obs: int,
        in_feat: int,
        z_dim: int,
        num_layers: int,
        lmax: int = 3,
        N: int = 16,
        dropout: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()

        self.G = group.so2_group(lmax)
        self.gspace = gspaces.no_base_space(self.G)
        self.num_layers = num_layers
        self.z_dim = z_dim

        obs_type = num_keypoints * [self.gspace.irrep(1), self.gspace.irrep(0)] + [
            self.gspace.irrep(0)
        ]
        self.in_type = enn.FieldType(self.gspace, num_obs * obs_type)
        self.keypoint_enc = SO2MLP(
            self.in_type,
            channels=[z_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            N=N,
            dropout=dropout,
            act_out=True,
            initialize=initialize,
        )
        self.out_type = self.keypoint_enc.out_type

    def forward(self, obs) -> torch.Tensor:
        B, T, Do = obs["keypoints"].shape

        x = self.in_type(obs["keypoints"].reshape(B, -1))
        obs_feat = self.keypoint_enc(x)

        return obs_feat.tensor


class SO2KeypointEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        in_feat: int,
        z_dim: int,
        num_layers: int,
        lmax: int = 3,
        N: int = 16,
        dropout: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()

        self.G = group.so2_group()
        self.gspace = gspaces.no_base_space(self.G)
        self.num_layers = num_layers
        self.z_dim = z_dim

        self.in_type = enn.FieldType(
            self.gspace, num_obs * in_feat * [self.gspace.irrep(1)]
        )
        self.keypoint_enc = SO2MLP(
            self.in_type,
            channels=[z_dim] * num_layers,
            lmaxs=[lmax] * num_layers,
            N=N,
            dropout=dropout,
            act_out=True,
            initialize=initialize,
        )
        self.out_type = self.keypoint_enc.out_type

    def forward(self, obs) -> torch.Tensor:
        B, T, Do = obs["keypoints"].shape

        x = self.in_type(obs["keypoints"].view(B, -1))
        obs_feat = self.keypoint_enc(x)

        return obs_feat.tensor


class SO2ObsEncoder(nn.Module):
    """SO2 image & agent position encoder."""

    def __init__(
        self,
        num_obs: int,
        img_channels: int,
        z_dim: int,
        lmax: int = 3,
        N: int = 16,
        dropout: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()

        self.G = group.so2_group(lmax)
        self.gspace = gspaces.no_base_space(self.G)
        self.z_dim = z_dim

        self.image_encoder = CyclicImageEncoder(
            img_channels, z_dim, dropout, lmax=lmax, N=N, initialize=initialize
        )
        self.fourier = Fourier(self.gspace, z_dim, self.G.bl_irreps(L=lmax), N=N)

        rho = self.G.spectral_regular_representation(*self.G.bl_irreps(L=lmax))
        self.lin_in_type = enn.FieldType(
            self.gspace,
            num_obs * z_dim * [rho] + num_obs
            # * ([self.gspace.irrep(1), self.gspace.irrep(0)] + [self.gspace.irrep(0)]),
            * [self.gspace.irrep(1)],
        )
        self.lin = SO2MLP(
            self.lin_in_type,
            channels=[z_dim],
            lmaxs=[lmax],
            N=N,
            dropout=dropout,
            act_out=True,
            initialize=initialize,
        )
        self.out_type = self.lin.out_type

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs["image"].shape

        c8_img_feat = self.image_encoder(obs["image"].view(B * T, C, H, W))
        so2_img_feat = self.fourier(c8_img_feat).tensor.view(B, -1)
        state = obs["agent_pos"].view(B, -1)

        img_feat_state = self.lin_in_type(torch.cat([so2_img_feat, state], dim=-1))
        obs_feat = self.lin(img_feat_state)

        return obs_feat.tensor


class SO2ObsEncoder2(nn.Module):
    """SO2 image & agent position encoder."""

    def __init__(
        self,
        num_obs: int,
        img_channels: int,
        z_dim: int,
        lmax: int = 3,
        N: int = 16,
        dropout: float = 0.0,
        initialize: bool = True,
    ):
        super().__init__()

        self.G = group.so2_group(lmax)
        self.gspace = gspaces.no_base_space(self.G)
        self.z_dim = z_dim

        self.image_encoder = CyclicImageEncoder(
            img_channels, z_dim, dropout, lmax=lmax, N=N, initialize=initialize
        )
        self.fourier = Fourier(self.gspace, z_dim, self.G.bl_irreps(L=lmax), N=N)

        img_feat_type = self.G.spectral_regular_representation(
            *self.G.bl_irreps(L=lmax)
        )
        # ee_pos_type = [self.gspace.irrep(1), self.gspace.irrep(0)]
        ee_pos_type = [self.gspace.irrep(1)]
        self.lin_in_type = enn.FieldType(
            self.gspace,
            num_obs * z_dim * [img_feat_type]
            + num_obs * ee_pos_type
            # + num_obs * [self.gspace.irrep(0)],
        )
        self.lin = SO2MLP(
            self.lin_in_type,
            channels=[z_dim],
            lmaxs=[lmax],
            N=N,
            dropout=dropout,
            act_out=True,
            initialize=initialize,
        )
        self.out_type = self.lin.out_type

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs["image"].shape

        c8_img_feat = self.image_encoder(obs["image"].view(B * T, C, H, W))
        so2_img_feat = self.fourier(c8_img_feat).tensor.view(B, -1)
        # ee_state = obs["keypoints"].view(B, -1)
        ee_state = obs["agent_pos"].view(B, -1)

        img_feat_state = self.lin_in_type(torch.cat([so2_img_feat, ee_state], dim=-1))
        obs_feat = self.lin(img_feat_state)

        return obs_feat.tensor

class SO2ObsEncoder3(nn.Module):
    def __init__(
        self,
        num_obs: int = 2,
        img_channels: int = 3,
        z_dim: int = 256,
        dropout: float = 0.0,
        spec_norm: bool = False,
        initialize: bool = True,
        lmax: int = 3,
        N: int = 16,
    ):
        super().__init__()
        
        self.num_obs = num_obs
        self.z_dim = z_dim

        self.image_encoder = SO2ImageEncoder(img_channels, z_dim, dropout, lmax=lmax, N=N, initialize=initialize)
        
        # We'll initialize the projection layer lazily in the first forward pass
        self.so2_proj = None
        self.dropout = dropout
        
        self.lin = MLP(
            [num_obs * z_dim + num_obs * 2] + [z_dim],
            dropout=dropout,
            act_out=True,
            spec_norm=False,
        )

    def forward(self, obs) -> torch.Tensor:
        B, T, C, H, W = obs["image"].shape
        
        # Process through SO2ImageEncoder
        so2_features = self.image_encoder(obs["image"].view(B * T, C, H, W))
        
        # Flatten the SO2 features
        so2_features = so2_features.view(B * T, -1)
        
        # Initialize projection layer on first forward pass
        if self.so2_proj is None:
            so2_dim = so2_features.shape[-1]
            self.so2_proj = nn.Sequential(
                nn.Linear(so2_dim, self.z_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            ).to(so2_features.device)
        
        # Project SO2 features to z_dim for each image
        so2_features = self.so2_proj(so2_features)  # [B*T, z_dim]
        
        # Reshape and concatenate
        img_feat = so2_features.view(B, T * self.z_dim)  # [B, T * z_dim]
        state = obs["agent_pos"].view(B, -1)
        
        img_feat_state = torch.cat([img_feat, state], dim=-1)
        obs_feat = self.lin(img_feat_state)
        
        return obs_feat
    
class DroneObsEncoder(nn.Module):
    """
    Observation encoder for drone tasks with image observations.
    
    Input format (obs dict):
        - "image": (B, T, C, H, W) or (B, T, H, W, C) - RGB images
        - "keypoint": (B, T, 3, 3) - 3D keypoints [current_pos, initial_pos, target_pos]
    """
    def __init__(
        self,
        num_obs: int,
        img_channels: int = 3,
        z_dim: int = 256,
        dropout: float = 0.0,
        pos_dim: int = 3,
        use_keypoints: bool = True,
        spec_norm: bool = False,
        initialize: bool = True,
    ):
        """
        Args:
            num_obs: number of observation frames (temporal window)
            img_channels: number of image channels (3 for RGB)
            z_dim: latent dimension
            dropout: dropout rate
            pos_dim: dimension of position (3 for 3D drone, 2 for 2D PushT)
            use_keypoint: if True, use full keypoint (9 dim), else use agent_pos (3 dim)
        """
        super().__init__()
        self.num_obs = num_obs
        self.z_dim = z_dim
        self.pos_dim = pos_dim
        self.use_keypoints = use_keypoints
        
        self.image_encoder = ImageEncoder(img_channels, z_dim, dropout)
        self.img_norm = nn.LayerNorm(num_obs * z_dim)
        
        if use_keypoints:
            state_dim = num_obs * 9
        else:
            state_dim = num_obs * pos_dim
        
        self.lin = MLP(
            [num_obs * z_dim + state_dim, z_dim],
            dropout=dropout,
            act_out=True,
            spec_norm=spec_norm,
        )
    
    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: dict with keys:
                - "image": (B, T, C, H, W) - RGB images
                - "keypoints": (B, T, 3, 3) or (B, T, 9) - keypoints
        
        Returns:
            obs_feat: (B, z_dim) - encoded observation features
        """
        image = obs["image"]
        
        # (B, T, H, W, C) -> (B, T, C, H, W)
        if image.shape[-1] == 3 and image.dim() == 5:
            image = image.permute(0, 1, 4, 2, 3)

        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        B, T, C, H, W = image.shape
        
        # Encode images: (B*T, C, H, W) -> (B*T, z_dim)
        img_feat = self.image_encoder(image.view(B * T, C, H, W))
        img_feat = img_feat.view(B, -1)  # (B, T * z_dim)
        img_feat = self.img_norm(img_feat)

        keypoints = obs.get("keypoints", obs.get("keypoint"))
        if keypoints is None:
            raise KeyError(f"No keypoints found. Keys: {obs.keys()}")
        state = keypoints.view(B, -1)  # (B, T * 9)

        img_feat_state = torch.cat([img_feat, state], dim=-1)
        obs_feat = self.lin(img_feat_state)
        
        return obs_feat

class SO2DroneObsEncoder(nn.Module):
    def __init__(
        self,
        num_obs: int,
        img_channels: int = 3,
        z_dim: int = 256,
        dropout: float = 0.0,
        pos_dim: int = 3,
        use_keypoints: bool = True,
        spec_norm: bool = False,
        initialize: bool = True,
        lmax: int = 3,
        N: int = 16,
    ):
        super().__init__()
        self.num_obs = num_obs
        self.z_dim = z_dim
        self.pos_dim = pos_dim
        self.use_keypoints = use_keypoints
        
        self.image_encoder = SO2ImageEncoder(img_channels, z_dim, dropout, lmax=lmax, N=N, initialize=initialize)
        
        # Get actual output dimension from SO2ImageEncoder
        img_feat_dim = self.image_encoder.out_type.size
        self.img_norm = nn.LayerNorm(num_obs * img_feat_dim)
        
        if use_keypoints:
            state_dim = num_obs * 9
        else:
            state_dim = num_obs * pos_dim
        
        self.lin = MLP(
            [num_obs * img_feat_dim + state_dim, z_dim],
            dropout=dropout,
            act_out=True,
            spec_norm=spec_norm,
        )
    
    def forward(self, obs: dict) -> torch.Tensor:
        image = obs["image"]
        
        if image.shape[-1] == 3 and image.dim() == 5:
            image = image.permute(0, 1, 4, 2, 3)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        
        B, T, C, H, W = image.shape
        
        img_feat = self.image_encoder(image.view(B * T, C, H, W))
        img_feat = img_feat.view(B, -1)
        img_feat = self.img_norm(img_feat)
        
        keypoints = obs.get("keypoints", obs.get("keypoint"))
        if keypoints is None:
            raise KeyError(f"No keypoints found. Keys: {obs.keys()}")
        state = keypoints.view(B, -1)
        
        img_feat_state = torch.cat([img_feat, state], dim=-1)
        obs_feat = self.lin(img_feat_state)
        
        return obs_feat