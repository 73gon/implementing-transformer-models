import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention


class FeatureTransformation(nn.Module):
    """
    Position-wise feed-forward network for the encoder layer.

    Names (linear1, linear2) are chosen to match the test STATE_DICT keys:
    'feature_transformation.linear1.*' and 'feature_transformation.linear2.*'
    """

    def __init__(self, input_dim: int, feature_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, feature_dim)
        self.linear2 = nn.Linear(feature_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = linear2(ReLU(linear1(x)))
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


class BaseTransformerLayer(nn.Module):
    """
    Single encoder-style Transformer layer used in the tests.

    - self_attention (Multi-head self-attention)
    - residual + layer_norm_1
    - feature_transformation (position-wise FFN)
    - residual + layer_norm_2
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        feature_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=False,
            dropout=dropout,
        )

        self.feature_transformation = FeatureTransformation(input_dim=input_dim, feature_dim=feature_dim)

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_dim)
        attention_mask: (batch_size, seq_len) where 1 = keep, 0 = pad
        """
        # 1) Self-attention sublayer
        attn_out = self.self_attention(x, x, x, mask=attention_mask)
        x = self.layer_norm_1(x + self.dropout(attn_out))

        # 2) Feed-forward sublayer
        ffn_out = self.feature_transformation(x)
        x = self.layer_norm_2(x + self.dropout(ffn_out))

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Decoder layer matching the test expectations.

    Names must match STATE_DICT keys:
    - self_attention.*
    - encoder_attention.*
    - feature_transformation.linear1/linear2.*
    - layer_norm_1/2/3.*
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        feature_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Masked self-attention
        self.self_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=True,
            dropout=dropout,
        )

        # Encoder–decoder cross attention
        self.encoder_attention = MultiHeadAttention(
            d_model=input_dim,
            num_heads=num_heads,
            mask_future=False,
            dropout=dropout,
        )

        # Position-wise feed-forward
        self.feature_transformation = FeatureTransformation(
            input_dim=input_dim,
            feature_dim=feature_dim,
        )

        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Called as: layer(input, encoder, encoder_attention_mask, attention_mask)

        Args:
            hidden_states:         (B, tgt_len, input_dim)
            encoder_hidden_states: (B, src_len, input_dim)
            encoder_attention_mask:(B, src_len) or (B, tgt_len, src_len)
            attention_mask:        (B, tgt_len) or (B, tgt_len, tgt_len)
        """
        x = hidden_states

        # 1) Masked self-attention (decoder)
        sa_out = self.self_attention(x, x, x, mask=attention_mask)
        x = self.layer_norm_1(x + self.dropout(sa_out))

        # 2) Encoder–decoder cross-attention
        ca_out = self.encoder_attention(
            x,
            encoder_hidden_states,
            encoder_hidden_states,
            mask=encoder_attention_mask,
        )
        x = self.layer_norm_2(x + self.dropout(ca_out))

        # 3) Feed-forward
        ffn_out = self.feature_transformation(x)
        x = self.layer_norm_3(x + self.dropout(ffn_out))

        return x
