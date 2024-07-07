import math

import torch
from torch import nn

from decoder_lm.utils import WordGPTConfig, scaled_dot_product


class WordGPTPositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model

        # In transformer models, each token is represented by a vector of size d_model. This means that for each token,
        # we have d_model features.
        # Position Representation:
        # The goal of positional encoding is to add information about the position to each token's representation.
        # Encoding application:
        # Each row in the pe matrix corresponds to a position in the sequence.
        # Each column in the pe matrix corresponds to a dimension of the token's feature vector.
        # Even/odd application:
        # The sine and cosine functions are applied to alternate dimensions (columns) of each position's encoding, not
        # to alternate positions (rows).

        # Unique Encoding: By applying different functions (sine and cosine) to different dimensions of the same position,
        # we ensure that each position has a unique encoding across all dimensions.
        # (Think of it as a way to create a unique "signature" for each position in the input sequence. The same pe is used for any token
        # that appear in that position)
        # Relative Position Information: The sine and cosine functions with different frequencies allow the model to easily
        # attend to relative positions. The model can learn linear combinations of these features to attend to both nearby and distant positions.
        # Dimension Independence: Each dimension of the positional encoding varies independently with position, which gives
        # the model flexibility in how it uses this information.
        pe = torch.zeros((seq_len, d_model))

        # position.shape = (seq_len, 1)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)

        # express the power of 10000 in log-space, a^b = exp(b log(a)) = exp(log(a^b))
        # thus 10000^{-2i/d_model} can be rewritten as exp(-2i/d_model log(10000))
        div_term = torch.exp(
            (torch.arange(0, d_model, 2).float() / d_model) * -math.log(10_000)
        )

        # sin for even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # cos for odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # going from (seq_len, d_model) => (1, seq_len, d_model)
        # Adding a third dimension to match the model's expected input shape
        self.pe = pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

    def forward(self, token_embeddings: torch.Tensor):
        token_embeddings = token_embeddings + (
            self.pe[:, token_embeddings.shape[1], :]
        ).requires_grad(False)

        return token_embeddings


class WordGPTEmbedding(nn.Module):
    def __init__(self, config: WordGPTConfig) -> None:
        super().__init__()

        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_encoding = WordGPTPositionalEncoding(
            seq_len=config.context_window,
            d_model=config.d_model,
        )

        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, x: torch.Tensor):
        hidden_state = self.word_embeddings(x)
        hidden_state = self.position_encoding(hidden_state)

        return self.dropout(hidden_state)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, head_dim) -> None:
        super().__init__()

        self.w_q = nn.Linear(d_model, head_dim)
        self.w_k = nn.Linear(d_model, head_dim)
        self.w_v = nn.Linear(d_model, head_dim)

    def forward(self, hidden_state: torch.Tensor):
        head_out = scaled_dot_product(
            q=self.w_q(hidden_state),
            k=self.w_k(hidden_state),
            v=self.w_v(hidden_state),
        )

        return head_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        assert (
            d_model % num_heads == 0
        ), f"d_model: {d_model} must be divisible by num_heads: {num_heads}"

        self.head_dim = d_model // num_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(d_model=d_model, head_dim=self.head_dim)
                for _ in range(num_heads)
            ]
        )
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, hidden_state: torch.Tensor):
        heads_out = torch.cat([head(hidden_state) for head in self.heads], dim=-1)
        heads_out = self.linear(heads_out)

        return heads_out
