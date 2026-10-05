"""Sprout's tokenizer: one id per character, plus four markers. No merges, nothing learned.

Phyll reads bytes merged into 4,096 pieces (byte-level BPE, content/courses/tiny-llm/code/tokenizer.py). Sprout reads single
characters from a small declared alphabet, so its whole vocabulary fits on one screen. The transformer code is the same; only the
text encoding differs.

Ids 0-2 match Phyll's: 0 pads (the loss skips it, model.py ignore_index=0), 1 starts a chat, 2 ends it. Id 3 stands for a character
outside the alphabet. It never occurs in Sprout's data, as an input or as a target. Its row of the token table still moves a
little in training: the readout scores every id with that same table (tied), and each step pushes down the scores of the ids
that were not the target, ⟨unk⟩ among them; weight decay shrinks it too.
"""

PAD, START, END, UNK = 0, 1, 2, 3
MARKERS = ["⟨pad⟩", "⟨start⟩", "⟨end⟩", "⟨unk⟩"]
ALPHABET = "\n .:=?" + "0123456789" + "abcdefghijklmnopqrstuvwxyz"
VOCAB = MARKERS + list(ALPHABET)
VOCAB_SIZE = len(VOCAB)  # 46
ID = {ch: i for i, ch in enumerate(VOCAB)}


def encode(text, strict=True):
    """Characters to ids. strict: an unknown character is an error (data tools); otherwise it becomes ⟨unk⟩ (the live lab)."""
    ids = []
    for ch in text:
        i = ID.get(ch)
        if i is None or i < len(MARKERS):
            if strict:
                raise ValueError(f"character {ch!r} is not in Sprout's alphabet")
            i = UNK
        ids.append(i)
    return ids


def chat_ids(text, strict=True):
    """One chat as Sprout reads it: ⟨start⟩, the characters, ⟨end⟩."""
    return [START] + encode(text, strict) + [END]


def decode(ids, markers=False):
    """Ids to text. Markers are dropped unless asked for (then spelled ⟨start⟩, ⟨end⟩ …)."""
    out = []
    for i in ids:
        if i < len(MARKERS):
            if markers:
                out.append(VOCAB[i])
        else:
            out.append(VOCAB[i])
    return "".join(out)


def label(i):
    """How a token is printed on a page: markers by name, the newline as ↵, the space as ␣."""
    t = VOCAB[i]
    return {"\n": "↵", " ": "␣"}.get(t, t)
