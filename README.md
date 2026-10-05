# Tiny AI Labs

The runnable side of the **Tiny AI courses**: for each course, a notebook that trains the course's model yourself, with its
dataset; and the companion notebooks of its episodes, with the data and the trained model they load. Every notebook opens in
Google Colab from the links below, or runs on your own computer.

## Train it yourself

One notebook per course trains the course's own model with the course's own recipe, top to bottom, in short sections a
beginner can read: set up, get the data, the model, train, watch it learn, how good is it, make it write, and save the trained
model to an `outputs/` folder. Open it in Colab, sign in with your Google account, and choose **Runtime → Run all**.

| Course | Model | Data | Time | Open |
|---|---|---|---|---|
| Tiny Transformer | our tiny transformer, 9,417 numbers | Tiny Shakespeare, 1.1 MB | about 1 min (quick) · about 21 min (full), CPU, measured on the course's server | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T8-train-your-own.ipynb) |
| Tiny LLM (Phyll) | Phyll, the cactus chat model, 8,726,016 numbers | Phyll's 217,798 chats, 11 MB gzipped | about 25 min (quick, 2,000 steps), CPU, measured on the course's server · use a T4 GPU; the full 18,000 steps are not timed (Phyll's own two runs took about 3.3 h on that CPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-llm/notebooks/L14-train-your-own.ipynb) |
| Tiny ViT, Tiny VLM | later | | | |

## Tiny Transformer

Unit 1 is a tour of one real model: **our tiny transformer**, 9,417 learned numbers trained on *Tiny Shakespeare*. It reads up
to 32 characters and gives a probability to each of the 65 characters that could come next. The notebooks run it, open it part
by part with every tensor's shape, and train it. (Units 3 and later still use Sprout, the course's earlier model, until they
move over.)

| Episode | | Open |
|---|---|---|
| P0 · Python Fundamentals for Transformers | optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python.ipynb) |
| P1 · PyTorch Fundamentals for Transformers | optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch.ipynb) |
| P2.1 · Vectors and shapes | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-vectors.ipynb) |
| P2.2 · Dot products | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-dot-product.ipynb) |
| P2.3 · Matrices and learned linear layers | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-linear.ipynb) |
| P2.4 · Token and position embeddings | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-positions.ipynb) |
| P2.5 · Mean, variance and LayerNorm | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-layernorm.ipynb) |
| P2.6 · Exponentials, probability and softmax | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-softmax.ipynb) |
| P2.7 · Attention as a weighted mixture | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-attention.ipynb) |
| P2.8 · Nonlinearity and GELU | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-gelu.ipynb) |
| P2.9 · Residual additions | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-residuals.ipynb) |
| P2.10 · Logarithms and cross-entropy | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-loss.ipynb) |
| P2.11 · Derivatives and backpropagation | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-gradients.ipynb) |
| P2.12 · Learning updates and the complete calculation | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P2-math-updates.ipynb) |
| F0 · The big picture: text in, next character out | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/F0-whole-machine.ipynb) |
| M1 · The data and the goal | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M1-training-data.ipynb) |
| M2 · Inside the machine 1: characters become vectors | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M2-inside-embeddings.ipynb) |
| M3 · Inside the machine 2: self-attention and two heads | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M3-inside-attention.ipynb) |
| M4 · Inside the machine 3: feed-forward and the residual add | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M4-inside-feed-forward.ipynb) |
| M5 · Inside the machine 4: from vectors to the next character | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M5-inside-readout.ipynb) |
| M6 · How training works | unit 1 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/M6-training-loop.ipynb) |
| F1 · Optional: the baseline to beat, counting pairs | unit 1, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/F1-counting.ipynb) |
| T1 · From text to training batches | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T1-training-batches.ipynb) |
| T2 · From logits to one loss | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T2-training-objective.ipynb) |
| T3 · How backpropagation finds gradients | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T3-training-gradients.ipynb) |
| T4 · How AdamW changes parameters | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T4-training-optimizer.ipynb) |
| T5 · Assemble the training loop | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T5-training-step.ipynb) |
| T6 · Measure what the model learns | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T6-training-metrics.ipynb) |
| T7 · Watch it learn, then train it yourself | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T7-training-progress.ipynb) |
| T8 · Train your own tiny transformer | unit 2 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/T8-train-your-own.ipynb) |
| P0.1 · Read, run, inspect | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-runtime.ipynb) |
| P0.2 · Text, positions, and slices | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-text.ipynb) |
| P0.3 · Containers and shared references | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-containers.ipynb) |
| P0.4 · Build a vocabulary | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-vocabulary.ipynb) |
| P0.5 · Follow iteration and unpacking | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-iteration.ipynb) |
| P0.6 · Functions with clear contracts | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-functions.ipynb) |
| P0.7 · Imports, files, and configuration | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-files.ipynb) |
| P0.8 · Classes and instances | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-classes.ipynb) |
| P0.9 · Read real model Python | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-model-python.ipynb) |
| P0.10 · Text-to-batch capstone | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P0-python-pipeline.ipynb) |
| P1.1 · What a tensor contains | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-tensors.ipynb) |
| P1.2 · Read tensor axes | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-axes.ipynb) |
| P1.3 · Select the values you mean | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-indexing.ipynb) |
| P1.4 · Assemble sequences and batches | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-batches.ipynb) |
| P1.5 · Reshape without losing meaning | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-reshape.ipynb) |
| P1.6 · Elementwise operations and broadcasting | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-broadcasting.ipynb) |
| P1.7 · Reduce along the right axis | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-reductions.ipynb) |
| P1.8 · Matrix multiplication and linear layers | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-matmul.ipynb) |
| P1.9 · Embeddings and positions | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-embeddings.ipynb) |
| P1.10 · Modules, parameters, and model state | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-modules.ipynb) |
| P1.11 · Assemble attention tensors | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-attention.ipynb) |
| P1.12 · Feed-forward blocks and residual paths | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-blocks.ipynb) |
| P1.13 · Predictions, targets, and loss | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-loss.ipynb) |
| P1.14 · Autograd follows the computation | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-autograd.ipynb) |
| P1.15 · Perform one training step | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-step.ipynb) |
| P1.16 · Evaluation and repeatable experiments | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-modes.ipynb) |
| P1.17 · Generate one token at a time | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-generation.ipynb) |
| P1.18 · Read, run, save, and restore | unit 0, optional | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gkiri/tiny_repo/blob/main/tiny-transformer/notebooks/P1-pytorch-capstone.ipynb) |

Open a notebook, press **Setup** (the first cell), then run the cells from top to bottom. Each section ends with a **Your turn**
exercise and a hidden solution.

### On your own computer

```
git clone https://github.com/gkiri/tiny_repo
cd tiny_repo/tiny-transformer/notebooks
pip install -r requirements.txt
jupyter notebook
```

## What is here

- `tiny-transformer/notebooks/`: the notebooks, committed with their outputs. `T8-train-your-own.ipynb` is the
  train-it-yourself notebook.
- `tiny-transformer/requirements.txt`: what T8 installs (lower bounds; Colab's preinstalled versions satisfy them).
  `tiny-transformer/notebooks/requirements.txt` pins the exact versions the course ran.
- `tiny-transformer/model/`: our tiny transformer's trained numbers (`prompter.json`, run prompter-r1), its `prompter_model.py`, config,
  model card and the run's training log (`prompter-r1-train.json`, the curve T8 draws yours against); Sprout's trained numbers (`sprout.json`, run sprout-r1), its tokenizer, its config and model card,
  and `model.py`, the model code it shares with Phyll, the Advanced course's model.
- `tiny-transformer/data/tinyshakespeare.txt`: Tiny Shakespeare (from karpathy/char-rnn; the plays are in the public domain)
  and its datasheet.
- `tiny-transformer/data/garden-notes/`: the Garden Notes generator, its fixed splits and its datasheet.
- `tiny-llm/notebooks/L14-train-your-own.ipynb`: train your own Phyll, the Tiny LLM courses' cactus (Lesson 14 of the
  Advanced course), with `tiny-llm/requirements.txt`.
- `tiny-llm/data/`: Phyll's chats (`train.jsonl.gz`, `eval.jsonl.gz`), its tokenizer and their datasheet (`PHYLL_CHATS.md`).
- `tiny-llm/model/`: `model.py`, Phyll's model code, and `phyll-record.json`, Phyll's recorded training and the finished
  Phyll's scores, which the notebook draws your run against.
- `MANIFEST.json`: every file's sha256 and the course commit it was published from.

## Licences

Code (`*.py`, notebook code cells): MIT, see `LICENSE`. Data, model numbers and prose: CC BY 4.0, see `LICENSE-DATA.md`.
These files are published from the course repository by its own tooling; please report problems there rather than editing here.
