# Datasheet: Tiny Shakespeare

- **What:** 1,115,394 characters from Shakespeare's plays, as speeches with the speaker's name in capitals on its own line
  (`ROMEO:`). 65 distinct characters: the new line, space, `!$&',-.3:;?`, and the 26 letters in both cases.
- **Where from:** collected by Andrej Karpathy for char-rnn,
  <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt> (repository MIT; the plays are in
  the public domain). Committed as `data/tinyshakespeare.txt`, sha256
  `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`; `data.py` refuses any other bytes.
- **Split:** the first 90% (1,003,854 characters) trains; the last 10% (111,540) validates. The split is contiguous, so
  validation is a stretch of plays the model never read. There is no separate test set: the model is a teaching model and
  its one score is the validation loss, reported with the baselines (an even guess, single-character counts, pair counts).
- **Known quirks:** the text is old spelling and verse, full of names; some passages repeat between plays; `3` and `$` appear
  only a handful of times. The course's running example, "You taught me language; and my p" (Caliban, in The Tempest), is in the validation part:
  the model never trained on it.
- **Use in the course:** every character the model knows and every count F1 shows come from this file.
