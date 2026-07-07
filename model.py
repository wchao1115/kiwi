from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import G


class Head(nn.Module):
    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(G.embedding_dim, head_size)
        self.query = nn.Linear(G.embedding_dim, head_size)
        self.value = nn.Linear(G.embedding_dim, head_size)
        self.dropout = nn.Dropout(G.dropout_rate)
        self.register_buffer("tril", torch.tril(torch.ones(G.block_size, G.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        w = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        mask = self.tril[: w.size(1), : w.size(1)]
        w = w.masked_fill(mask == 0, float("-inf"))
        w = F.softmax(w, dim=-1)
        w = self.dropout(w)
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, head_count: int, head_size: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(head_count)])
        self.proj = nn.Linear(G.embedding_dim, G.embedding_dim)
        self.dropout = nn.Dropout(G.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(G.embedding_dim, 4 * G.embedding_dim),
            nn.ReLU(),
            nn.Linear(4 * G.embedding_dim, G.embedding_dim),
            nn.Dropout(G.dropout_rate),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Block(nn.Module):
    def __init__(self, embedding_dim: int, head_count: int) -> None:
        super().__init__()
        self.sa_heads = MultiHeadAttention(head_count, embedding_dim // head_count)
        self.ffwd = FeedForward()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa_heads(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


class Model(nn.Module):
    def __init__(self, vocab_size: int, device: torch.device) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, G.embedding_dim)
        self.position_embedding = nn.Embedding(G.block_size, G.embedding_dim)
        self.blocks = nn.Sequential(
            *[Block(G.embedding_dim, G.head_count) for _ in range(G.layer_count)],
            nn.LayerNorm(G.embedding_dim),
        )
        self.lm_head = nn.Linear(G.embedding_dim, vocab_size)
        self.to(device)

    def forward(
        self, x: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        t = x.size(1)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(torch.arange(t, device=x.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, x: torch.Tensor, max_new_tokens: int = 48) -> torch.Tensor:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ctx = x[:, -G.block_size :]
                logits, _ = self.forward(ctx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                new_token = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, new_token], dim=1)
        if was_training:
            self.train()
        return x
