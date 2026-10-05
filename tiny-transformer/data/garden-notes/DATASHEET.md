# Datasheet: Garden Notes

Garden Notes is Sprout's training and test data: synthetic plant-care chats written by this course (after *Datasheets for
Datasets*, Gebru et al.). **Status: split revision 3**, fixed with `config.json` and `GATES.md` in the commit made before the
real run, `sprout-r1`.

`python -B training/tiny-transformer/garden_notes.py build` writes the seven fixed sets to `data/*.jsonl` (git-ignored) and their
manifest to `data/splits.json` (committed). The build is deterministic: it rebuilds the same bytes every time.

## Summary and motivation

Two kinds of question teach the two kinds of memory a transformer has, at a size a 20,672-parameter model learns on a CPU in
minutes:
- **Remembered facts (A):** a fixed world of 12 plants × 3 facts (how often to water, how much light, where it lives), asked 8
  ways. The answer is nowhere in the chat, so it must be stored in the weights (the feed-forward pages).
- **Note recall (B):** a note line pairs 3 different plants with 3 values, then asks about one plant. The answer is in the
  chat, so it must be read from the context (attention's job: the look-up head pages).

```
u:how often do i water aloe?          n:fern=wet aloe=hot rose=wet
s:every 14 days.                      u:aloe?
                                      s:hot.
```

## Composition

Records are text, never token ids. Each row is a JSON object with:
- its `text` (`n:` note, `u:` question, `s:` answer lines) and its `answer`;
- what generated it. A rows: `plant`, `kind`, `phrasing`, `held`. B rows: `plants`, `values`, `query`, `plant`, `family`,
  `split`. Paired rows add `group` and `edit`.

The alphabet has 42 characters. With four markers, Sprout's vocabulary is 46 ids (`sprout_tokenizer.py`). A chat with its
⟨start⟩ and ⟨end⟩ must fit in 64 ids; the longest any generator can make is 57. Type C (days until the next watering) exists in
the generator but is off.

| Set | Role | Rows | Drawn from | sha256 (split revision 3) |
|---|---|---:|---|---|
| `a_train` | train | 216 | the trained phrasings | `89fdefb59ac2c86447013571d06b2edd1c665164f232a7ae2ef0ea53aeab2d05` |
| `a_val` | val | 36 | 6 plants × 3 kinds × (new combination, new wording) | `4ffffe2ef95cab125f00f060bedef0b60eb13147bb7fa88a444e5f769c7c6b99` |
| `a_test` | test | 36 | the other 6 plants; 36 fact × phrasing ids | `8d2064d9eb2941a189414d90d91982017f07b5255cd6932daf6039263b360ea7` |
| `b_val` | val | 576 | 461 held-out families | `da9a369b5cfef5c9c3edc708da72225b82b0e5e270c0d43b6bb9f4a37886328f` |
| `b_test` | test | 576 | 453 held-out families | `22b199f8ba82221ab4536bd300838b9d6c0fda4b73653ebd7dd4a387996b75a4` |
| `b_paired` | test | 384 (96 groups × 4) | 95 held-out families | `61fa6bb475cbf2d9f42725149e347dad8f4b01cf88f991cf006389f5e0da0064` |
| `b_probe` | probe | 576 | 569 training families | `9ce58e024d83e33a29b54fc3e339eb4c4c69178bdb1b45c5f04e74aa79c53cf8` |

- **The hashes are `data/splits.json`'s.** It also holds the family counts, the salt, and the sha256 of the generator and the
  tokenizer. Every script loads a set only after its file's actual sha256 matches `splits.json`.
- **Family counts are not denominators.** They say how many families a set is drawn from. Each score is over the set's rows
  (and `b_paired`'s consistency over its 96 groups).
- **The training stream is generated, not stored.** Each update reads 64 chats. Each is a remembered fact with probability 0.25
  (from the 216 trained phrasings); otherwise it is a fresh note-recall chat from a training family. Measured over the first
  200 updates: 24.7% of chats, and 22.6% of target characters, are A.

## How it is made

`garden_notes.py` holds the world, the rules, the splits, the training stream and the build.
- Every choice is a `random.Random` seeded with `int(sha256("garden-notes:1:<key>"))` (`_h`). The key is `set:<name>` for a
  fixed set and `stream:<seed>:<step>` for a training batch.
- The batch for step n depends on the seed and n only, so the whole stream is reproducible without being stored.
- Splits hash their key directly, so they never depend on Python's hash randomisation or iteration order.
- Serialization: `json.dumps(row, ensure_ascii=False)` plus a newline, keys in the order they are built, UTF-8.

**The build fails unless:**
- held-out A phrasings are never trained texts, each new combination is trained for some other plant, and each new wording for
  none;
- no set repeats a text, and `b_val` and `b_test` share no family;
- every paired group keeps one family: *swap* and *requery* change the answer, *reorder* keeps it and moves the asked note;
- no held-out family was held out by revision 1 (the split rule itself retires the families of revisions 1 and 2);
- every note-recall chat in the first 200 training updates comes from a training family;
- every chat fits in 64 ids.

## Splits and what is held out

- **Remembered facts (A):** of each kind's 8 phrasings, one (the new wording) is never trained for any plant. Of the other 7,
  one more per fact (the new combination) is held out: trained for other plants, never for this one. The other 6 train. The 12
  plants split 6/6 between validation and test. These rules are unchanged since revision 1.
- **Note recall (B):** every one of the 16 × 6 plant=value pairs is trained. What is held out is whole **families**: a set of
  three plants with a multiset of three values, in any pairing, order or question. 31,360 families exist.
  - `b_val`, `b_test`: every (asked plant, its value, its slot) twice, unique texts, from held-out families only.
  - `b_paired`: 96 groups, each a base chat and three edits that keep its family. *Swap* changes the asked value, *requery*
    asks a plant with a different value, *reorder* moves the asked note to another slot (same answer). 34 of its 95 families
    also occur in `b_test`, by design: both are drawn from the test families.
  - `b_probe`: training families, on purpose (a probe, not a holdout). 121 of its 569 families were held out by an earlier
    revision (71 by revision 1, 50 by revision 2) and are training families now.
  - The training stream draws a note-recall chat again until its family is a training family. So held-out families never
    occur in training.

## Split revisions (why revision 3)

Each revision retires every family an earlier revision held out (it may only train), and draws validation and test afresh,
with its own salt, from the rest. The salt decides the allocation only. It is never part of an identity, so overlaps across
revisions stay visible.

| Revision | Allocation key | Train | Val | Test | Retired | What happened to its held-out families |
|---|---|---:|---:|---:|---:|---|
| 1 | `b:family:` | 28,215 | 1,563 | 1,582 | 0 | test families read by `provisional-sprout-s8k` (results seen) and a 30-step test model |
| 2 | `b:family:v2:` | 28,726 | 1,330 | 1,304 | 3,145 | every test family read by four 30-step test models before any freeze |
| **3** | **`b:family:v3:`** | **28,685** | **1,378** | **1,297** | **5,779** | fresh: no identity in the exposure history |

**Why revision 3 exists.** Four 30-step test models read every revision-2 test family: two end-to-end tests, a claims-tool test,
and one inside Astra's own round-3 review. They were trained on the production training stream. Nobody looked at their
results, but a model read the test rows, so revision 2 could no longer give a held-out result. Astra's exposure consultation
(`docs/reviews/course-standard/sprout-exposure-astra-r1.md`) chose a fresh holdout.

Revision 3 retires both earlier allocations: 3,145 revision-1 families and 2,634 more from revision 2. It draws from the rest
with salt `v3`, fixed once and never tried against model performance.
- Unchanged: the world, the seed, the remembered-fact rules, the identity definition, serialization and the RNG.
- The training stream changes only because it now skips a different set of families.
- Retiring all 5,779 families is a conservative rule, not a claim that all of them were evaluated.

**Checked** when revision 3 was drawn, and again for this datasheet: the families of `b_val`, `b_test` and `b_paired` share none
with any B identity in the exposure history (`provisional/test-exposure.json`), and none with any family revision 1 or 2 held
out, for validation or for test.

This folder builds revision 3 only. The allocations of revisions 1 and 2 can still be computed from their keys (`RETIRED_KEYS`
in `garden_notes.py`), and the identities earlier models read are listed in `provisional/test-exposure.json`.

## Holdout identity

- **B:** the family: the sorted plants joined by `,`, then `|`, then the sorted values joined by `,` (`aloe,elm,fern|cool,dry,dry`).
- **A:** fact × phrasing: `plant/kind/phrasing`, the phrasing's index in `garden_notes.PHRASINGS` (`aloe/water/3`).

Identities do not depend on serialization or on the salt, so the exposure history stays comparable across revisions.

## Purposes and exposure

- **`b_val`** chooses the checkpoint and the look-up head. **`a_val`** is reported, and is part of gate 4's validation loss.
- **`b_test`, `b_paired`: confirmatory.** Fresh family holdouts, read by one evaluation of the frozen checkpoint
  (`evaluate.py`): one frozen suite, with its planned overlap (34 families) and its planned ablations.
- **`a_test`: descriptive.** Its 36 identities are the same in every revision. Earlier prototypes read them
  (`provisional-sprout-s8k` on revision 1, and the 30-step test models), so they are **previously exposed**. A-test results are
  reported apart from the B results, never gated, and never called held-out.
- **`b_probe`** shows how Sprout does on training families. It is not a holdout.

Every earlier read of a test row is listed in `provisional/TEST-EXPOSURE.md`. No test or tool runs a model on `b_test`,
`b_paired` or `a_test` outside that one evaluation. What learners are told: "Earlier prototypes informed development. B results
use fresh family holdouts; the A wording examples were examined before and are descriptive."

## Intended uses

Training and testing Sprout for the Tiny Transformer course, and showing learners what the model reads. Not a benchmark. The
plant facts are simplified care advice for teaching, not horticultural guidance.

## Licence and sources

Generated by this course (seed 1). No third-party data, nothing downloaded. The licence is not chosen yet: the founder decides
before publication.
