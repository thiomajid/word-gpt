import math

import torch
import torch.nn.functional as F
from pydantic import BaseModel


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    d_k = q.size(-1)
    sims = torch.matmul(q, k)
    alphas = sims / math.sqrt(d_k)

    if mask is not None:
        alphas = alphas.masked_fill(mask == 0, -math.inf)

    alphas = F.softmax(alphas, dim=-1)

    attention_weights = torch.matmul(alphas, v)
    return attention_weights


class WordGPTConfig(BaseModel):
    vocab_size: int
    d_model: int
    hidden_dim: int
    dropout_prob: int
    context_window: int
