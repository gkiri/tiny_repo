"""Our tiny transformer: 9,417 learned numbers that predict the next character of Shakespeare.

It reads up to 32 characters of text and gives a probability to each of the 65 characters that could come next. Writing is
that prediction in a loop: pick a character, append it, slide the 32-character window, and predict again. (The class is
called Prompter for the course's file names; learners meet it as "our tiny transformer".)

The architecture is the "10k" configuration of maxpolaczuk/tiny-character-transformer (a 1-block, 2-head character GPT on Tiny
Shakespeare). The code here is written fresh for the course (that repository has no licence, so none of its code or weights
are copied): every line is one the course teaches.

    characters ─► token table (65 × 28) + position table (32 × 28)        lookup:      what, and where
               ─► one block: LayerNorm ─► 2-head causal attention ─► +    look back:   which earlier letters matter
                             LayerNorm ─► feed-forward 28 → 56 → 28 ─► +  think:       transform each row on its own
               ─► final LayerNorm ─► readout with the token table again   score all 65 characters
               ─► softmax                                                  probabilities for the next character

Parameters: token table 1,820 · positions 896 · block 6,580 (two LayerNorms 112, q|k|v 2,436, attention out 812,
feed-forward 1,624 + 1,596) · final LayerNorm 56 · readout bias 65 = 9,417.
"""

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 65    # every distinct character in Tiny Shakespeare
    block_size: int = 32    # how many characters the model can read at once (its context)
    d_model: int = 28       # how many numbers describe one character in one place (a row)
    n_head: int = 2         # attention heads, each looking with 28 / 2 = 14 numbers
    d_ff: int = 56          # the feed-forward network's hidden width (2 × d_model)
    n_layer: int = 1        # transformer blocks

    def to_dict(self):
        return asdict(self)


class Tokenizer:
    """One id per character, in sorted order: '\\n' is 0, ' ' is 1, … 'z' is 64."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)


class CausalSelfAttention(nn.Module):
    """Each row asks a question (q), every earlier row offers a label (k) and a message (v). A row reads the messages of the
    rows whose labels match its question best, never a row to its right."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)     # q, k and v for both heads in one matrix
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)        # mix the heads' results back into one row
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)

    def forward(self, x, keep=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # how well each question matches each label
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))   # no peeking at the future
        att = F.softmax(att, dim=-1)                                          # shares that add up to 1
        if keep is not None:
            keep["attn"] = att.detach()
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)            # each row: a weighted mix of messages
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))

    def forward(self, x, keep=None):
        x = x + self.attn(self.ln1(x), keep)    # look back, and add what was found to the row
        x = x + self.ff(self.ln2(x))            # think about each row on its own, and add that too
        return x


class Prompter(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)    # one learned row per character
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)    # one learned row per place 0 … 31
        self.blocks = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head_bias = nn.Parameter(torch.zeros(cfg.vocab_size))  # the readout reuses tok_emb (tied); only its bias is new
        self.apply(self._init)

    @staticmethod
    def _init(m):
        # small random numbers (σ = 0.02), so the untrained model's guesses are nearly even: its first loss is about
        # ln 65 = 4.17, the loss of a uniform guess over 65 characters
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, keep=None):
        T = idx.size(1)
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))   # what + where
        for b in self.blocks:
            x = b(x, keep)
        x = self.ln_f(x)
        logits = x @ self.tok_emb.weight.T + self.head_bias                         # a score for every character
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, generator=None):
        for _ in range(n):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
