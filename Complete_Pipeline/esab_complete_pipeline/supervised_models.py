from __future__ import annotations

import copy

import torch
from torch import nn


class SimpleCNNPredictor(nn.Module):
    """Lightweight CNN for CWT classification or regression."""

    def __init__(self, in_channels: int, output_dim: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class PatchViTPredictor(nn.Module):
    """Small ViT that supports non-square CWT scalograms."""

    def __init__(
        self,
        image_height: int,
        image_width: int,
        in_channels: int,
        output_dim: int,
        patch_size: tuple[int, int] = (79, 20),
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        patch_h, patch_w = patch_size
        if image_height % patch_h != 0 or image_width % patch_w != 0:
            raise ValueError(
                f"Image size ({image_height}, {image_width}) must be divisible by "
                f"patch_size {patch_size}."
            )

        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        num_patches = (image_height // patch_h) * (image_width // patch_w)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        batch_size = x.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.dropout(x + self.pos_embed)
        x = self.encoder(x)
        return self.head(self.norm(x[:, 0]))


def _load_torchvision_models():
    """Import torchvision models only when a torchvision model is built."""
    try:
        from torchvision import models
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "torchvision is required for supervised ResNet/ViT models. "
            "Install a torchvision version compatible with your PyTorch build."
        ) from exc
    return models


def _replace_resnet_first_conv(model: nn.Module, in_channels: int) -> None:
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        if in_channels == old_conv.in_channels:
            new_conv.weight.copy_(old_conv.weight)
        elif in_channels == 1:
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        else:
            repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_conv.weight.copy_(repeated)
        if old_conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv1 = new_conv


def _replace_vit_conv_projection(model: nn.Module, in_channels: int) -> None:
    old_proj = model.conv_proj
    new_proj = nn.Conv2d(
        in_channels,
        old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )
    with torch.no_grad():
        if in_channels == old_proj.in_channels:
            new_proj.weight.copy_(old_proj.weight)
        elif in_channels == 1:
            new_proj.weight.copy_(old_proj.weight.mean(dim=1, keepdim=True))
        else:
            repeated = old_proj.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
            new_proj.weight.copy_(repeated)
        if old_proj.bias is not None and new_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
    model.conv_proj = new_proj


def build_supervised_model(
    model_name: str,
    in_channels: int,
    output_dim: int,
    pretrained: bool = False,
    *,
    image_height: int | None = None,
    image_width: int | None = None,
    patch_size: tuple[int, int] = (79, 20),
    patch_vit_embed_dim: int = 192,
    patch_vit_depth: int = 4,
    patch_vit_num_heads: int = 6,
    patch_vit_mlp_ratio: float = 4.0,
    patch_vit_dropout: float = 0.1,
) -> nn.Module:
    """Build a supervised model for classification or regression."""
    name = str(model_name).strip().lower()

    if name == "cnn":
        return SimpleCNNPredictor(in_channels=in_channels, output_dim=output_dim)

    if name == "patch_vit":
        if image_height is None or image_width is None:
            raise ValueError("patch_vit requires image_height and image_width.")
        return PatchViTPredictor(
            image_height=int(image_height),
            image_width=int(image_width),
            in_channels=in_channels,
            output_dim=output_dim,
            patch_size=patch_size,
            embed_dim=patch_vit_embed_dim,
            depth=patch_vit_depth,
            num_heads=patch_vit_num_heads,
            mlp_ratio=patch_vit_mlp_ratio,
            dropout=patch_vit_dropout,
        )

    models = _load_torchvision_models()
    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        _replace_resnet_first_conv(model, in_channels)
        model.fc = nn.Linear(model.fc.in_features, output_dim)
        return model

    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        model = models.vit_b_16(weights=weights)
        _replace_vit_conv_projection(model, in_channels)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, output_dim)
        return model

    raise ValueError(
        "Unsupported model_name. Use 'cnn', 'resnet18', 'vit_b_16', or 'patch_vit'."
    )


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a CPU copy of a model state dict."""
    return {key: value.detach().cpu().clone() for key, value in copy.deepcopy(model.state_dict()).items()}
