"""
Vision Transformer (ViT) Architecture for Face Attribute Classification.
Implements patch embedding projection, learnable positional embeddings,
class token concatenation, multi-head self-attention transformer encoder blocks,
and classification head for CelebA face tensors (B, 3, 64, 64).
"""

import math

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and projects them to embedding dimension.
    For 64x64 image with patch_size=8: (64/8)^2 = 64 patches.
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Standard Vision Transformer Encoder block with Multi-Head Attention and MLP.
    """
    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.mha(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerClassifier(nn.Module):
    """
    Compact Vision Transformer for 64x64 face attribute classification.
    Input shape: (B, 3, 64, 64) -> Output: (B, num_classes) logits.
    """
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 1,
        embed_dim: int = 64,
        depth: int = 3,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # Take class token
        return self.head(cls_out)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities for binary classification."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
