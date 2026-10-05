# Model card: Sprout

After *Model Cards for Model Reporting* (Mitchell et al.). **Status: recorded.** The recipe, the gates and the selection rules were
frozen in `config.json` and `GATES.md` before the one confirmation run.

**The recorded run, `sprout-r1` (2026-10-02, commit `5fdf209`):** 165 s on 6 CPU threads; the checkpoint is step 8,000
(highest B-val exact match, 100%). Greedy exact match: B-test 99.7%, B-paired 99.5% (97.9% of groups all four right),
A-train 100%, A-val 94.4% (A-test 88.9%, descriptive only). Validation loss 0.3157 against the smoothed bigram's 1.6433.
**Gates 0–5 and 6a pass; gate 6b fails**, and is reported, not re-run: both heads of the second block share the look-up
(attention on the asked value 0.57 and 0.50). Zeroing the chosen head (L2H2) at the answer position drops B-test
first-character accuracy by 44.8 points, past 6a's 40; zeroing the other (L2H1) drops it by 59.0, so the 20-point margin
over the control is not met. Evidence: `results/sprout-r1/` (train.json, log, eval.json, GATES-report.md, the dry run,
`best.pt`, plots); the course data: `src/specimen/tiny-transformer/data/`.

- **Summary:** Sprout is the seedling the free Tiny Transformer course builds, trains, tests and opens up, one episode at a time.
  It reads a garden note and answers a question about it (attention's look-up), and answers questions about a small fixed world
  of plant facts (memory in the weights).
- **Architecture, config and tally:** Phyll's own `content/courses/tiny-llm/code/model.py`, unmodified. Each run records its
  sha256 (`model_py_sha256` in `train.json`). Preset `s`: vocab 46, context 64, d 32, 2 blocks, 2 heads × 16, FFN 64 (ReLU),
  pre-LN, tied readout, dropout 0. **20,672 parameters**, counted from the module (`common.tally`): token table 1,472 ·
  positions 2,048 · per block 8,544 (qkv 3,168 · out 1,056 · up 2,112 · down 2,080 · two LayerNorms 128) · final LayerNorm 64.
  The config moves Phyll's architecture across, not Phyll's training: Sprout trains from scratch and inherits nothing from Phyll.
- **Tokenizer:** one id per character (`sprout_tokenizer.py`): ⟨pad⟩ 0, ⟨start⟩ 1, ⟨end⟩ 2 (as Phyll), ⟨unk⟩ 3, then 42 characters.
- **Data:** Garden Notes, split revision 3 (`DATASHEET.md`). Synthetic, written by this course.
- **Recipe:** `config.json`: seed 1, 8,000 updates of 64 chats, AdamW (lr 3e-3, betas 0.9/0.99, weight decay 0.1), 100 warm-up
  steps then a cosine to 10%, clip 1.0, fp32, 6 CPU threads. The dry run projects 164 s for the whole run.
- **Protocol and gates:** `GATES.md` and `config.json`: one real run, no retries; a failed gate is reported, never re-run for a
  pass. Gates 0–2 come from the dry run of 2026-10-02 on revision 3 (`train.py --dry`): start loss 3.8197 against ln 46 =
  3.8286; one chat below 0.01 after 148 updates; eight chats within 0.02 of their floor after 145 updates.
- **Results:** `results/sprout-r1/GATES-report.md` (one evaluation of the frozen checkpoint, step 8,000). Greedy exact match:
  B-test 574 of 576, B-paired 382 of 384 (94 of 96 groups all right), A-train 216 of 216, A-val 34 of 36, A-test 32 of 36
  (read by earlier models: descriptive only). Gate 6b fails (see Status); every other gate passes.
- **The look-up, honestly:** L2H2 was chosen to inspect because it put the most attention on the asked value on validation.
  On B-test both second-block heads attend strongly to that value (L2H2 0.574, L2H1 0.496). Zeroing L2H2 only where the first
  answer character is predicted lowers accuracy by 44.8 points; the same cut on L2H1 lowers it by 59.0. Both heads contribute;
  these measurements do not establish one uniquely specialised look-up head (Astra's wording,
  `docs/reviews/course-standard/sprout-r1-astra-r1.md`).
- **Behaviours measured:** the look-up head (chosen on validation, cut on test at the answer-writing position, against a
  control), first-layer binding, feed-forward units for facts, the causal mask, position bands, and the plateau then the drop.
  Each is reported with its measured number, or "not found"; none is claimed without one.
- **Limitations:** a teaching model on a synthetic task. Remembered facts generalise to new wordings only partly (the provisional
  runs scored about 92% on A-val). A-test is previously exposed, so it is descriptive only. Cuts show what a part is needed for
  on this data, not that attention alone holds the look-up or that the FFN alone holds the facts; no page may claim either.
- **Committed weights:** the founder's decision (2026-10-01). The selected checkpoint's float32 weights are committed inside the
  course export, `src/specimen/tiny-transformer/data/sprout.json`, named by the sha256 of those weights. The export's
  `NOTICE.md` names the run, the checkpoint's step, and the sha256 of the run's checkpoint file (`best.pt`, committed in `results/sprout-r1/`).
- **Provenance:** the run's `train.json`. It records the arguments, `config.json`'s sha256, the git commit (a real run refuses
  uncommitted code), the sha256 of the data, the code and `model.py`, the environment, the wall clock, the best step, the
  checkpoints' sha256 and the loss at every step. The published copies (`train.json`, `eval.json`, `log.jsonl`,
  `GATES-report.md`, the dry run and the plots) go into `results/sprout-r1/`. Every number on a course page comes from
  `export.py`'s files, made from these records.
- **Tests:** `test_sprout.py` runs 15 fast tests of the pipeline's logic. They score models on validation and training rows only,
  never on a test set.
- **History:** first built in `tools/py/sprout/` (snapshot commit `7b76c2c`). The `ml/` framework of ADR 0008 was dropped for
  this standalone folder (ADR 0009).
- **Intended use:** the Tiny Transformer course's pages (traces, curves, one recorded update, samples), with the evidence label
  "Sprout recorded". Not for any other purpose.
