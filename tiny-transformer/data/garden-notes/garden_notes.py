"""Sprout's Garden Notes: the small plant-care chat Sprout learns from. Written by us, generated from a seed, nothing downloaded.

Two kinds of question teach the two kinds of memory a transformer has:

  Type A, remembered facts. A fixed little world: 12 plants, 3 facts each (how often to water, how much light, where it lives),
  asked 8 ways. The answer is nowhere in the chat, so it must be stored in the weights.
      u:how often do i water aloe?
      s:every 14 days.
  Of the 8 phrasings of each kind of question, one is never trained for any plant (new wording). Of the other 7, one more per
  fact is held out: that phrasing was trained for other plants, never for this one (a new combination). The other 6 train.

  Type B, note recall. A note line pairs 3 different plants with 3 values (values may repeat), then asks about one plant. The
  answer is in the chat, so it must be read from the context: attention's job.
      n:fern=wet aloe=hot rose=wet
      u:aloe?
      s:hot.
  Every one of the 16 x 6 plant=value pairs is trained. What is held out is whole families: a set of three plants together with
  a set of three values (in any pairing, any order, any question). About 1 family in 10 never occurs in training; validation
  and test use those only, balanced over the asked plant, its value and its slot. A paired test then changes one thing at a time
  (swap the values, ask a plant with another value, move the asked note) and checks the answer moves with the notes. This is
  rev 3 of the family split: the families revisions 1 and 2 held out were read by earlier models, so they are retired to training.

  Type C (off unless asked): days until the next watering, from a note and a remembered fact. A stretch goal, not in rev 1.

Plants, values and the line tags (n, u, s) all start with different letters, so the first character of a name identifies it.

    python -B training/tiny-transformer/garden_notes.py build      # write the fixed sets to training/tiny-transformer/data/
    python -B training/tiny-transformer/garden_notes.py sample 8   # print a few training chats
"""

import hashlib
import itertools
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sprout_tokenizer as tok  # noqa: E402

SEED = 1
DATA = os.path.join(HERE, "data")
MAX_IDS = 64  # ⟨start⟩ + characters + ⟨end⟩; the model reads at most 64 positions (the input is every id but the last)

# ------------------------------------------------------------------------------------------------ the world

# Type A: 12 plants and their facts (water every n days, light, spot). Close to real care advice, simplified.
FACTS = {
    "aloe":   (14, "full sun", "by the window"),
    "basil":  (2,  "full sun", "in the kitchen"),
    "fern":   (3,  "shade",    "on a shelf"),
    "ivy":    (7,  "some sun", "on a shelf"),
    "jade":   (21, "full sun", "by the window"),
    "kale":   (2,  "full sun", "in the garden"),
    "orchid": (7,  "some sun", "by the window"),
    "palm":   (10, "some sun", "on the porch"),
    "rose":   (4,  "full sun", "in the garden"),
    "thyme":  (5,  "full sun", "in the kitchen"),
    "violet": (4,  "shade",    "on a desk"),
    "yucca":  (14, "full sun", "on the porch"),
}
KINDS = ("water", "light", "spot")
PHRASINGS = {
    "water": ["how often do i water {p}?", "when do i water {p}?", "how often does {p} need water?",
              "how often should {p} get water?", "how often to water {p}?", "when should i water {p}?",
              "water {p} how often?", "how often is {p} watered?"],
    "light": ["how much sun does {p} like?", "how much light does {p} need?", "does {p} like sun?",
              "how much sun for {p}?", "what light suits {p}?", "how much light for {p}?",
              "what light does {p} like?", "how sunny should {p} be?"],
    "spot":  ["where should {p} live?", "where do i put {p}?", "where does {p} grow best?", "where should i keep {p}?",
              "where is a good spot for {p}?", "where do i keep {p}?", "what spot suits {p}?", "where should {p} go?"],
}

# Type B: 16 plants and 6 values. Every first letter is different from every other, and from the tags n, u, s.
PLANTS = ["aloe", "basil", "elm", "fern", "ginger", "ivy", "jade", "kale",
          "orchid", "palm", "quince", "rose", "thyme", "violet", "yucca", "zinnia"]
VALUES = ["cool", "dry", "hot", "low", "mist", "wet"]
NOTES_PER_LINE = 3
RESERVED_PER_100 = 10  # families held out of training, split evenly between validation and test


def _check_world():
    firsts = [p[0] for p in PLANTS] + [v[0] for v in VALUES] + ["n", "u", "s"]
    assert len(set(firsts)) == len(firsts), "first letters must all differ"
    assert set(FACTS) <= set(PLANTS)
    assert all(len(v) == 8 for v in PHRASINGS.values())


def answer_a(plant, kind):
    days, light, spot = FACTS[plant]
    return {"water": f"every {days} days.", "light": f"{light}.", "spot": f"{spot}."}[kind]


# ------------------------------------------------------------------------------------------------ deterministic splits

def _h(key):
    """A stable number for a key and the seed: splits never depend on Python's hash randomisation or on iteration order."""
    return int(hashlib.sha256(f"garden-notes:{SEED}:{key}".encode()).hexdigest(), 16)


# Type A: one new-wording phrasing per kind; one new-combination phrasing per fact; plants split 6/6 between validation and test.
NEW_WORDING = {k: min(range(8), key=lambda j: _h(f"a:new:{k}:{j}")) for k in KINDS}
A_EVAL_PLANTS = sorted(FACTS, key=lambda p: _h(f"a:plant:{p}"))
A_SPLIT_OF = {p: ("val" if i % 2 == 0 else "test") for i, p in enumerate(A_EVAL_PLANTS)}


def a_phrasings(plant, kind):
    """The phrasing numbers of one fact: 6 trained, 1 new combination (trained for other plants), 1 new wording (never trained)."""
    rest = [j for j in range(8) if j != NEW_WORDING[kind]]
    combo = min(rest, key=lambda j: _h(f"a:combo:{plant}:{kind}:{j}"))
    return {"train": [j for j in rest if j != combo], "combo": combo, "new": NEW_WORDING[kind]}


def chat_a(plant, kind, j, held=None):
    return {"type": "A", "plant": plant, "kind": kind, "phrasing": j, "held": held,
            "text": f"u:{PHRASINGS[kind][j].format(p=plant)}\ns:{answer_a(plant, kind)}", "answer": answer_a(plant, kind)}


def a_lines(split):
    if split == "train":
        return [chat_a(p, k, j) for p in FACTS for k in KINDS for j in a_phrasings(p, k)["train"]]
    out = []
    for p in FACTS:
        if A_SPLIT_OF[p] != split:
            continue
        for k in KINDS:
            ph = a_phrasings(p, k)
            out += [chat_a(p, k, ph["combo"], "combination"), chat_a(p, k, ph["new"], "wording")]
    return out


# Type B: families.
def family(plants, values):
    return ",".join(sorted(plants)) + "|" + ",".join(sorted(values))


def _reserve(fam, key):
    h = _h(f"{key}{fam}")
    if h % 100 >= RESERVED_PER_100:
        return "train"
    return "val" if (h // 100) % 2 == 0 else "test"


# Rev 3 of the family split. Provisional runs and test harnesses read the families revisions 1 and 2 held out, so every family
# an earlier revision held out is retired from evaluation (it may only train), and validation and test are drawn afresh, with a
# new salt, from the rest (provisional/TEST-EXPOSURE.md).
B_FAMILY_SALT = "v3"
RETIRED_KEYS = ("b:family:", "b:family:v2:")  # revisions 1 and 2


def family_split(fam):
    """"train", or "val"/"test" for the reserved families."""
    if any(_reserve(fam, key) != "train" for key in RETIRED_KEYS):
        return "train"
    return _reserve(fam, f"b:family:{B_FAMILY_SALT}:")


def chat_b(plants, values, q):
    notes = " ".join(f"{p}={v}" for p, v in zip(plants, values))
    return {"type": "B", "plants": list(plants), "values": list(values), "query": q, "plant": plants[q],
            "family": family(plants, values), "split": family_split(family(plants, values)),
            "text": f"n:{notes}\nu:{plants[q]}?\ns:{values[q]}.", "answer": f"{values[q]}."}


def random_b(rng, split="train", fixed=None):
    """A note-recall chat from the given split's families. fixed = (plant, value, slot) pins the asked note."""
    while True:
        if fixed:
            qp, qv, slot = fixed
            others = rng.sample([p for p in PLANTS if p != qp], NOTES_PER_LINE - 1)
            plants = others[:slot] + [qp] + others[slot:]
            values = [rng.choice(VALUES) for _ in range(NOTES_PER_LINE)]
            values[slot] = qv
            q = slot
        else:
            plants = rng.sample(PLANTS, NOTES_PER_LINE)
            values = [rng.choice(VALUES) for _ in range(NOTES_PER_LINE)]
            q = rng.randrange(NOTES_PER_LINE)
        if family_split(family(plants, values)) == split:
            return chat_b(plants, values, q)


def chat_c(rng):
    """Type C (off by default): days until the next watering = the remembered interval minus the days since."""
    plant = rng.choice(sorted(FACTS))
    days = FACTS[plant][0]
    since = rng.randrange(1, days) if days > 1 else 0
    return {"type": "C", "plant": plant, "text": f"n:{plant} watered {since} days ago\nu:days until water?\ns:{days - since}.",
            "answer": f"{days - since}."}


A_TRAIN = None


def training_batch(step, size, frac_a=0.25, seed=SEED, with_c=False):
    """The chats for one training step: fresh note-recall chats from training families, and remembered-fact chats drawn from the
    216 trained phrasings. Seeded by the step, so the whole stream is reproducible without being stored."""
    global A_TRAIN
    if A_TRAIN is None:
        A_TRAIN = a_lines("train")
    rng = random.Random(_h(f"stream:{seed}:{step}"))
    out = []
    for _ in range(size):
        r = rng.random()
        if r < frac_a:
            out.append(rng.choice(A_TRAIN))
        elif with_c and r < frac_a + 0.1:
            out.append(chat_c(rng))
        else:
            out.append(random_b(rng))
    return out


def b_balanced(name, split, reps=2):
    """Every (asked plant, its value, its slot) the same number of times: 16 x 6 x 3 x reps chats, no text twice."""
    rng = random.Random(_h(f"set:{name}"))
    out, seen = [], set()
    for _ in range(reps):
        for qp, qv, slot in itertools.product(PLANTS, VALUES, range(NOTES_PER_LINE)):
            while True:
                c = random_b(rng, split, (qp, qv, slot))
                if c["text"] not in seen:
                    break
            seen.add(c["text"])
            out.append(c)
    return out


def b_paired(name, split):
    """Counterfactual groups from held-out families: a base chat and three edits of it. 'swap' moves the values to other plants
    (the asked plant's value changes), 'requery' asks another plant whose value differs (the answer changes), 'reorder' moves
    the asked note to another slot (the answer stays).
    The edits keep the same plants and values, so they stay in the same held-out family."""
    rng = random.Random(_h(f"set:{name}"))
    out = []
    for g, (qp, qv) in enumerate(itertools.product(PLANTS, VALUES)):
        slot = (g + g // len(VALUES)) % NOTES_PER_LINE  # each plant and each value is asked in every slot
        while True:
            base = random_b(rng, split, (qp, qv, slot))
            if len(set(base["values"])) > 1:  # a swap must be able to change the answer
                break
        P, V, q = base["plants"], base["values"], base["query"]
        while True:
            perm = rng.sample(range(NOTES_PER_LINE), NOTES_PER_LINE)
            if V[perm[q]] != V[q]:
                break
        swap = chat_b(P, [V[i] for i in perm], q)
        requery = chat_b(P, V, rng.choice([i for i in range(NOTES_PER_LINE) if V[i] != V[q]]))  # its answer differs
        while True:
            order = rng.sample(range(NOTES_PER_LINE), NOTES_PER_LINE)
            if order.index(q) != q:  # the asked note moves to another slot
                break
        reorder = chat_b([P[i] for i in order], [V[i] for i in order], order.index(q))
        for edit, c in (("base", base), ("swap", swap), ("requery", requery), ("reorder", reorder)):
            out.append({**c, "group": g, "edit": edit})
    return out


SETS = {
    "a_train": lambda: a_lines("train"),          # 216: the gate's trained phrasings
    "a_val": lambda: a_lines("val"),              # 36: 6 plants x 3 kinds x (new combination, new wording)
    "a_test": lambda: a_lines("test"),            # 36: the other 6 plants (touched once)
    "b_val": lambda: b_balanced("b_val", "val"),       # 576: held-out families, balanced; selects checkpoint and head
    "b_test": lambda: b_balanced("b_test", "test"),    # 576: the other held-out families (touched once)
    "b_paired": lambda: b_paired("b_paired", "test"),  # 96 groups x 4: does the answer follow the notes?
    "b_probe": lambda: b_balanced("b_probe", "train"),  # 576: training families (lines may also occur in training): a probe
}


def ids_of(chat):
    ids = tok.chat_ids(chat["text"])
    assert len(ids) <= MAX_IDS, (len(ids), chat["text"])
    return ids


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build():
    _check_world()
    os.makedirs(DATA, exist_ok=True)
    report = {"seed": SEED, "vocab": tok.VOCAB_SIZE, "sets": {},
              "a_new_wording": {k: PHRASINGS[k][j] for k, j in NEW_WORDING.items()}, "a_split_of_plant": A_SPLIT_OF}
    made = {name: make() for name, make in SETS.items()}
    for name, rows in made.items():
        lengths = [len(ids_of(c)) for c in rows]
        path = os.path.join(DATA, f"{name}.jsonl")
        with open(path, "w") as f:
            for c in rows:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        report["sets"][name] = {"lines": len(rows), "max_ids": max(lengths), "sha256": sha256_file(path)}

    # invariants: held-out phrasings are not trained texts; the combination phrasing is trained for some other plant;
    # the new wording is trained for none; held-out families never occur in the training stream; sets do not repeat a text
    train_texts = {c["text"] for c in made["a_train"]}
    trained = {(c["kind"], c["phrasing"]) for c in made["a_train"]}
    for c in made["a_val"] + made["a_test"]:
        assert c["text"] not in train_texts
        assert ((c["kind"], c["phrasing"]) in trained) == (c["held"] == "combination")
    for name in ("b_val", "b_test", "b_probe", "b_paired"):
        assert len({c["text"] for c in made[name]}) == len(made[name])
    assert all(c["split"] == "val" for c in made["b_val"]) and all(c["split"] == "test" for c in made["b_test"] + made["b_paired"])
    assert not {c["family"] for c in made["b_val"]} & {c["family"] for c in made["b_test"]}
    assert all(_reserve(c["family"], "b:family:") == "train" for n in ("b_val", "b_test", "b_paired") for c in made[n])  # none retired
    groups = {}
    for c in made["b_paired"]:
        groups.setdefault(c["group"], {})[c["edit"]] = c
    for g in groups.values():
        b = g["base"]
        assert g["swap"]["answer"] != b["answer"] and g["requery"]["answer"] != b["answer"] and g["reorder"]["answer"] == b["answer"]
        assert g["reorder"]["query"] != b["query"] and len({c["family"] for c in g.values()}) == 1
    stream = [c for s in range(1, 201) for c in training_batch(s, 64)]
    assert all(c["split"] == "train" for c in stream if c["type"] == "B")
    # how much of a training batch is each type, by chats and by target characters
    n_a = sum(c["type"] == "A" for c in stream)
    t_a = sum(len(c["text"]) + 1 for c in stream if c["type"] == "A")
    t_all = sum(len(c["text"]) + 1 for c in stream)
    report["stream_share_a"] = {"chats": n_a / len(stream), "targets": t_a / t_all}
    # the longest chat any generator can make: the three longest plants, the longest value three times, the longest asked
    longest_b = 2 + len("n:") + 3 * max(map(len, PLANTS)) + 3 + 3 * max(map(len, VALUES)) + (NOTES_PER_LINE - 1) \
        + len("\nu:?\ns:.") + max(map(len, PLANTS)) + max(map(len, VALUES))
    report["max_ids_any_b"] = longest_b
    report["max_ids_any_a"] = max(len(ids_of(chat_a(p, k, j))) for p in FACTS for k in KINDS for j in range(8))
    assert longest_b <= MAX_IDS and report["max_ids_any_a"] <= MAX_IDS
    # how many families there are, and how many are held out
    fams = [family(ps, vs) for ps in itertools.combinations(PLANTS, 3)
            for vs in itertools.combinations_with_replacement(VALUES, 3)]
    report["families"] = {s: sum(family_split(f) == s for f in fams) for s in ("train", "val", "test")}
    report["families"]["retired"] = {"rev1": sum(_reserve(f, "b:family:") != "train" for f in fams),
                                     "rev2": sum(_reserve(f, "b:family:v2:") != "train" for f in fams),
                                     "all": sum(any(_reserve(f, k) != "train" for k in RETIRED_KEYS) for f in fams)}
    report["b_family_salt"] = B_FAMILY_SALT
    report["generator_sha256"] = sha256_file(os.path.abspath(__file__))
    report["tokenizer_sha256"] = sha256_file(os.path.join(HERE, "sprout_tokenizer.py"))
    with open(os.path.join(DATA, "splits.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return report


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        print(json.dumps(build(), indent=1, ensure_ascii=False))
    elif cmd == "sample":
        for c in training_batch(0, int(sys.argv[2]) if len(sys.argv) > 2 else 8):
            print(c["text"].replace("\n", " ↵ "), "   |", c["type"])
    else:
        sys.exit("usage: garden_notes.py build|sample [n]")
