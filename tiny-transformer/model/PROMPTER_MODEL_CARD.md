# Model card: our tiny transformer (`prompter`)

**What it is.** A character-level transformer with 9,417 numbers, trained on Tiny Shakespeare on a CPU in minutes. It reads up to
32 characters and gives a probability to each of the 65 characters for what comes next. It is the Tiny Transformer course's
model: every part the course teaches (tokenizer, token and position tables, attention with two heads, LayerNorm, the
feed-forward network, the readout) is in it exactly once.

**Architecture.** `model.py`: token table 65 × 28 + position table 32 × 28 → one pre-LayerNorm block (two causal attention
heads of 14 numbers, a GELU feed-forward network 28 → 56 → 28, residual adds) → final LayerNorm → readout that reuses the token
table, plus its own 65 biases. The "10k" configuration of maxpolaczuk/tiny-character-transformer, re-implemented (that
repository has no licence).

**Training.** `train.py`, recipe in `config.json`; one recorded run, `results/prompter-r1/` (log, `train.json`, samples).
Initial weights are small (σ 0.02), so the untrained model's loss is the even guess, ln 65 ≈ 4.17.

**How good is it.** Run `prompter-r1` (2026-10-03, 100,000 steps, 1,280 s on 4 CPU threads; best checkpoint at step 98,000):

| On the whole validation text (111,540 characters) | loss (nats/char) |
|---|---|
| an even guess over 65 characters (ln 65) | 4.1744 |
| single-character counts (unigram, +1 smoothing) | 3.3473 |
| pair counts (bigram, +1 smoothing; F1's model) | 2.4819 |
| **our tiny transformer** | **1.9317** (2.787 bits/char; on the training text 1.7651) |

The reference reports 1.9538 for this configuration on its own evaluation. After "You taught me language; and my p" (from the
validation text) it gives "r" 0.340 and "a" 0.210; always picking the most probable, it writes "rove the shall the shall" and
loops. It writes text with Shakespeare's shape (speaker names, line breaks, common short
words) and almost no sense. That gap is the course's point: one block with 9,417 numbers learns spelling and rhythm; meaning
needs much more.

**Not for.** Anything but teaching. It invents words, it knows no facts, and it has never seen modern English.
