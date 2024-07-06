import torch
from torch import nn

from decoder_lm.utils import WordGPTConfig, scaled_dot_product


class WordGPTEmbedding(nn.Module):
    def __init__(self, config: WordGPTConfig) -> None:
        super().__init__()

        self.word_embeddings = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
        )

        self.position_embeddings = nn.Embedding(config.context_window, config.d_model)

    def forward(self, x: torch.Tensor):
        pass


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
