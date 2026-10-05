# Phyll's chats (build m9)

The chats Phyll, the Tiny LLM courses' cactus, learned to talk from. Lesson 14's notebook (`L14-train-your-own.ipynb`)
downloads them to train a Phyll of your own.

| File | Chats | What it is for |
|---|---|---|
| `train.jsonl.gz` | 217,798 | training |
| `eval.jsonl.gz` | 9,113 | held back: never trained on, only for checks |
| `tokenizer.json` | | Phyll's tokenizer: 4,096 tokens (byte-level BPE) |

**Format.** One JSON object per line: `{"text": …, "category": …}`. `text` is a whole chat in Phyll's chat template, turns
joined by a new line:

```
<|im_start|>user
what can you do? i have to run<|im_end|>
<|im_start|>assistant
i can talk a little. that is enough before you go.<|im_end|>
```

A chat about the windowsill starts with a state line, `[days 9 | weather clear] should i water now`. `category` names the
topic: 23 families (safety, identity, water, hunger and thirst, light, feelings, greetings, …) and an intent.

**Tokens.** With `tokenizer.json`, the training chats hold 8,661,178 tokens; the longest chat is 119 tokens, so every chat
fits Phyll's 128 positions. Only two special tokens occur: id 1 (`<|im_start|>`) and id 2 (`<|im_end|>`). Id 0 is padding.

**How it was made.** The chats were written for the course with AI writing assistants, from the course's own character notes
for Phyll, then filtered and built by the course's tools. The build repeats some families (safety and identity twice, yes/no
questions three times) and adds copies of windowsill chats with the days changed. The held-back chats are 5% of the writing
batches, kept whole, so no batch is split between the two files. Some prompts try on purpose to make Phyll something it is not;
Phyll answers as itself. It is a toy character: its watering talk is not plant-care advice.

**Which Phyll.** The finished Phyll (`phyll-9M@71d251a`) was trained on these chats in its second run (m9), which continued
from its first run (m8) on an earlier build of the chats.

**Checksums** (sha256 of the uncompressed files): `train.jsonl` 8802d43a204ae59ac44c8e3aea145d8ee6247ee18755e084aa1130fdb5d7b659,
`eval.jsonl` 2de01f198296c93eeb5589d91e0fbb65625027f52c070875536a4ed1bd3cb43e, `tokenizer.json`
5816f3083451d11028d0861201853d5305c741ff863c111b6c5ca667c9db5b53.

**Licence.** CC BY 4.0, see `LICENSE-DATA.md` at the top of this repository.
