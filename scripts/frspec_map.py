#!/usr/bin/env python
"""An FR-Spec hot-token map tuned to our ARC traffic (the draft model can only propose tokens in the map).

Pennyroyal's map (hot_tokens_64k.pt in dfranzen/pennyroyal-v253, sha256 becfa41d...) is a generic 64k list. In the
model outputs of our Franzen-harness runs, 1.2-1.5% of tokens are outside it ("Hmm", " BFS", ".ascii", grid runs such
as "bbbb"), and the draft can never propose those. This script counts output tokens in runs/*/kernel-output/prompts
logs (assistant reasoning and tool calls of each game's last model call), adds every token used at least MIN times
that the map lacks, and drops as many map tokens our traffic never used (highest id first, never a special token such
as </think> or <|im_end|>), so the map keeps its 65,536 entries and the draft's cost.

    uv venv V && uv pip install --python V/bin/python tokenizers torch --index-url https://download.pytorch.org/whl/cpu
    V/bin/python -I scripts/frspec_map.py counts TOKENIZER.json OUT_COUNTS.json LOG...
    V/bin/python -I scripts/frspec_map.py build HOT.pt COUNTS.json TOKENIZER.json OUT.pt [--min 2]
    V/bin/python -I scripts/frspec_map.py coverage HOT.pt TOKENIZER.json LOG...

Read HOT.pt with torch.load(weights_only=True) only (a pickle). The tokenizer is the served model's tokenizer.json
(sha256 06b95093..., the TOKENIZER_SHA of his launcher).
"""
from __future__ import annotations

import collections
import hashlib
import json
import re
import sys

SEGMENT = re.compile(r"^\[ASSISTANT\]\n(.*?)(?=^\[(?:TOOL RESULT|USER|SYSTEM|ASSISTANT\]|MODEL RESPONSE META|"
                     r"TURN TRANSCRIPT|ANALYZER STATUS))", re.S | re.M)


def segments(paths):
    """The distinct model outputs (reasoning + tool calls) in the prompt logs."""
    seen = set()
    for path in paths:
        text = open(path, encoding="utf-8", errors="replace").read()
        for m in SEGMENT.finditer(text):
            seg = m.group(1).replace("[REASONING]\n", "", 1)
            seg = re.sub(r"^\[ASSISTANT TOOL CALL: python\]\nid: \S+\n", "", seg, flags=re.M)
            if seg not in seen:
                seen.add(seg)
                yield seg


def counts(tokenizer_path, out_path, logs):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    c, n = collections.Counter(), 0
    for seg in segments(logs):
        c.update(tok.encode(seg, add_special_tokens=False).ids)
        n += 1
    json.dump({"segments": n, "tokens": sum(c.values()), "logs": len(logs),
               "counts": {str(k): v for k, v in sorted(c.items())}}, open(out_path, "w"))
    print(f"{n} segments, {sum(c.values())} tokens, {len(c)} distinct ids")


def build(hot_path, counts_path, tokenizer_path, out_path, min_count=2):
    import torch
    hot = torch.load(hot_path, weights_only=True)
    special = {a["id"] for a in json.load(open(tokenizer_path))["added_tokens"]}
    c = {int(k): v for k, v in json.load(open(counts_path))["counts"].items()}
    hotset = set(hot)
    add = sorted((i for i, n in c.items() if i not in hotset and n >= min_count), key=lambda i: (-c[i], i))
    unused = sorted((i for i in hot if c.get(i, 0) == 0 and i not in special), reverse=True)
    add = add[:len(unused)]
    drop = set(unused[:len(add)])
    new = sorted((hotset - drop) | set(add))
    assert len(new) == len(hot) and (special & hotset) <= set(new)
    torch.save(new, out_path)
    assert torch.load(out_path, weights_only=True) == new
    sha = hashlib.sha256(open(out_path, "rb").read()).hexdigest()
    print(f"added {len(add)} ids used >= {min_count} times, dropped {len(drop)} never-used regular ids; "
          f"{len(new)} ids; sha256 {sha}")


def coverage(hot_path, tokenizer_path, logs):
    import torch
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_path)
    hot = set(torch.load(hot_path, weights_only=True))
    total = inside = 0
    missing = collections.Counter()
    for seg in segments(logs):
        for i in tok.encode(seg, add_special_tokens=False).ids:
            total += 1
            if i in hot:
                inside += 1
            else:
                missing[i] += 1
    print(f"{total} output tokens, {inside / max(total, 1):.4%} in the map; most frequent missing: "
          + ", ".join(f"{tok.decode([i])!r} x{n}" for i, n in missing.most_common(8)))


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "counts":
        counts(args[0], args[1], args[2:])
    elif cmd == "build":
        min_count = int(args[args.index("--min") + 1]) if "--min" in args else 2
        build(*args[:4], min_count=min_count)
    elif cmd == "coverage":
        coverage(args[0], args[1], args[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
