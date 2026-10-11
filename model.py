from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from data import Tokenizer


@dataclass
class GPTConfig:
    vocab_size: int = len(Tokenizer.tokens)
    max_seq_len: int = 151 #max_full_sequence_length(32)
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ff_hidden: int = 192
    dropout: float = 0.0

    def __post_init__(self):
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer for RoPE")


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        mean_square = x.float().pow(2).mean(dim=-1, keepdim=True)
        return (x * torch.rsqrt(mean_square + self.eps).to(x.dtype)) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim)))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # q, k: [batch, heads, tokens, head_dim]. Pair adjacent coordinates
        positions = torch.arange(start = 0, end = q.size(-2), device=q.device, dtype=torch.float32)
        angles = torch.outer(positions, self.inv_freq)#dont care about the red line, it is just a warning about the type of the tensor/
        cosine = angles.cos()[None, None, :, :]
        sine = angles.sin()[None, None, :, :]

        def rotate(x):
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            a, b = pairs.unbind(-1)
            return torch.stack((a * cosine - b * sine, a * sine + b * cosine), dim=-1).flatten(-2).to(x.dtype)

        return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)#you only mutiply once
        self.proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch, token, m_dim = x.shape
        q, k, v = self.qkv(x).split(m_dim, dim=-1)
        def heads(t):
            return t.view(batch, token, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        q, k = self.rope(q, k)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.ones(token, token, device=x.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(F.softmax(scores.float(), dim=-1).to(scores.dtype))
        output = (weights @ v).transpose(1, 2).contiguous().view(batch, token, m_dim)
        return self.dropout(self.proj(output))


class SwiGLU(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.ff_hidden, bias=False)
        self.up = nn.Linear(config.d_model, config.ff_hidden, bias=False)
        self.down = nn.Linear(config.ff_hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config)
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, ids, labels=None):
        if ids.size(1) > self.config.max_seq_len:
            raise ValueError(f"Sequence length {ids.size(1)} exceeds max_seq_len={self.config.max_seq_len}")
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return logits, loss

    @torch.inference_mode()
    def generate(self, prompt: list[int], eos_id: int, max_new_tokens: int, device: torch.device) -> list[int]:
        self.eval()
        ids = torch.tensor([prompt], dtype=torch.long, device=device)
        generated = []
        for _ in range(max_new_tokens):
            if ids.size(1) > self.config.max_seq_len:
                raise ValueError("Generation exceeded model context length")
            logits, _ = self(ids)
            token = int(logits[0, -1].argmax())
            generated.append(token)
            if token == eos_id:
                break
            ids = torch.cat((ids, torch.tensor([[token]], device=device)), dim=1)
        return generated

    def config_dict(self):
        return asdict(self.config)
