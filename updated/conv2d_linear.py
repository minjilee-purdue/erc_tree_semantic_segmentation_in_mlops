# comparison
"""
Configuration file for SAM fine-tuning experiments
"""

import torch
import torch.nn as nn

class SimpleViTPatchEmbedConv(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SimpleViTPatchEmbedLinear(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * in_channels
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, -1, self.patch_dim)
        x = self.proj(x)
        return x


if __name__ == "__main__":
    img = torch.randn(2, 3, 224, 224)

    patch_conv = SimpleViTPatchEmbedConv()
    patch_linear = SimpleViTPatchEmbedLinear()

    out_conv = patch_conv(img)
    out_linear = patch_linear(img)

    print(f"Conv2D Patch Embedding Output Shape: {out_conv.shape}")
    print(f"Linear Patch Embedding Output Shape: {out_linear.shape}")
