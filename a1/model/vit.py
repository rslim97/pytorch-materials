import torch
import torch.nn as nn
from modules.transformer.transformer_layers import *
import copy


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, src_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask)

        return output


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.
    """

    def __init__(
        self,
        img_size=32,
        patch_size=8,
        in_channels=3,
        embed_dim=128,
        num_layers=6,
        num_heads=4,
        dim_feedforward=256,
        num_classes=10,
        dropout=0.1,
    ):
        """
        Inputs:
         - img_size: Size of input image (assumed square).
         - patch_size: Size of each patch (assumed square).
         - in_channels: Number of image channels.
         - embed_dim: Embedding dimension for each patch.
         - num_layers: Number of Transformer encoder layers.
         - num_heads: Number of attention heads.
         - dim_feedforward: Hidden size of feedforward network.
         - num_classes: Number of classification labels.
         - dropout: Dropout probability.
        """
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=dropout)

        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification layer to predict class scores from pooled token.
        self.head = nn.Linear(embed_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Forward pass of Vision Transformer.

        Inputs:
         - x: Input image tensor of shape (N, C, H, W)

        Returns:
         - logits: Output classification logits of shape (N, num_classes)
        """
        N = x.size(0)
        logits = torch.zeros(N, self.num_classes, device=x.device)

        projected_inputs = self.patch_embed(x)
        projected_inputs = self.positional_encoding(projected_inputs)
        out = self.transformer(projected_inputs)
        out = torch.mean(out, dim=1)
        logits = self.head(out)

        return logits
