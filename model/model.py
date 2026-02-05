import torch
import torch.nn as nn

from model.embeddings import TokenEmbedding
from model.functional import BaseTransformerLayer, TransformerDecoderLayer
from model.positional_encoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder consisting of multiple encoder layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                BaseTransformerLayer(
                    input_dim=d_model,
                    num_heads=n_heads,
                    feature_dim=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_encoder_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) where 1 = keep, 0 = pad

        Returns:
            (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        return x


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder consisting of multiple decoder layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    input_dim=d_model,
                    num_heads=n_heads,
                    feature_dim=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_decoder_layers)
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch_size, tgt_len, d_model)
            encoder_hidden_states: (batch_size, src_len, d_model)
            encoder_attention_mask: (batch_size, src_len) where 1 = keep, 0 = pad
            attention_mask: (batch_size, tgt_len) where 1 = keep, 0 = pad

        Returns:
            (batch_size, tgt_len, d_model)
        """
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                attention_mask=attention_mask,
            )
        return hidden_states


class Transformer(nn.Module):
    """
    Complete Transformer model with encoder and decoder.

    This is a configurable transformer model suitable for sequence-to-sequence tasks
    like machine translation, summarization, etc.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        max_len: int = 512,
        pad_id: int = 0,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary
            d_model: Dimensionality of the embedding and hidden layers
            n_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Dimensionality of the feedforward network
            dropout: Dropout probability
            max_len: Maximum sequence length
            pad_id: Padding token ID (default: 0)
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_id = pad_id

        # Embeddings and positional encoding
        self.encoder_embedding = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)
        self.decoder_embedding = TokenEmbedding(vocab_size, d_model, pad_id=pad_id)

        self.encoder_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )
        self.decoder_positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        # Encoder and decoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.decoder = TransformerDecoder(
            d_model=d_model,
            n_heads=n_heads,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following best practices for transformers."""
        init_std = 0.02  # Standard for transformer models

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                # Keep padding embedding at zero
                if hasattr(module, "padding_idx") and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _create_attention_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Create attention mask from token IDs where padding tokens are masked.

        Args:
            token_ids: (batch_size, seq_len)

        Returns:
            attention_mask: (batch_size, seq_len) where 1 = keep, 0 = pad
        """
        return (token_ids != self.pad_id).float()

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input_ids: (batch_size, src_len) - source sequence token IDs
            decoder_input_ids: (batch_size, tgt_len) - target sequence token IDs
            encoder_attention_mask: (batch_size, src_len) where 1 = keep, 0 = pad
                                   If None, created from encoder_input_ids
            decoder_attention_mask: (batch_size, tgt_len) where 1 = keep, 0 = pad
                                   If None, created from decoder_input_ids

        Returns:
            logits: (batch_size, tgt_len, vocab_size)
        """
        # Create masks if not provided
        if encoder_attention_mask is None:
            encoder_attention_mask = self._create_attention_mask(encoder_input_ids)
        if decoder_attention_mask is None:
            decoder_attention_mask = self._create_attention_mask(decoder_input_ids)

        # Encoder
        encoder_embeddings = self.encoder_embedding(encoder_input_ids)
        encoder_embeddings = self.encoder_positional_encoding(encoder_embeddings)
        encoder_output = self.encoder(
            encoder_embeddings,
            attention_mask=encoder_attention_mask,
        )

        # Decoder
        decoder_embeddings = self.decoder_embedding(decoder_input_ids)
        decoder_embeddings = self.decoder_positional_encoding(decoder_embeddings)
        decoder_output = self.decoder(
            hidden_states=decoder_embeddings,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=decoder_attention_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(decoder_output)
        return logits
