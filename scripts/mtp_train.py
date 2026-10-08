#!/usr/bin/env python
"""Fine-tune the dense weights of the Flash-Next MTP draft on dumped target hidden states (plan section 3; the dump is
step 1's, scripts/sglang_hc_dump_patch.py + scripts/hc_dump_driver.py; the block is scripts/mtp_replica.py).

    python scripts/mtp_train.py plan  --dump DUMP --snapshots HC_RUN/snapshots.jsonl [--dump DUMP2 --snapshots S2]
    python scripts/mtp_train.py train --draft DRAFT_DIR --target-dir MODEL_DIR --token-map hot_tokens_64k.pt \\
        --dump DUMP --snapshots HC_RUN/snapshots.jsonl --out TRAIN_OUT [--epochs 2] [--batch-rows 16384]
    python scripts/mtp_train.py eval  --draft DRAFT_DIR --target-dir MODEL_DIR --token-map hot_tokens_64k.pt \\
        --dump DUMP --snapshots ... --trained TRAIN_OUT/trained-dense.safetensors --out EVAL_OUT

**Data.** Every loss span the dump kept (the plan file's ``loss_spans``; a span runs from the token after
``<|im_start|>assistant\\n<think>\\n`` to ``<|im_end|>``) becomes one or more windows: the span's rows plus up to
``--context`` rows before it (the dump keeps 256), at most ``--max-rows`` rows (2,048: within that QSA is exactly dense,
so no indexer is needed), long spans split into pieces that each keep ``--context`` rows of left context. An origin
``t`` of a window predicts ``x_{t+2}`` at step 1 (from ``H_t`` and ``e(x_{t+1})``) and ``x_{t+k+1}`` at step k; its
target is the target's own distribution there, ``softmax(lm_head_hot(Mixer_T(H_{t+k})))`` over the FR-Spec rows,
computed from the dumped rows on the fly (plan 3.4). Origins run from the row before the span (the generation prompt's
last row, whose proposal is the span's second token) to ``e - k - 1``; a step counts only while its target token is in
the span and, by default, in the hot map. Windows of the 11 held-out games (scripts/hc_dump_driver.py HOLDOUT, read from
the driver's snapshots.jsonl) are the evaluation set and are never trained on (plan 2.6).

**Draft-extend embeddings** (``--embed``; see scripts/mtp_replica.py): ``turn`` (default) rebuilds the draft KV the
server had when the span was generated: rows inside assistant spans and the last row before each span take
``e(x_{j+1})`` (decoded rows: post-verify extends), other rows of a request with images take the target's input
embedding ``e(x_j)`` or the dumped vision features (prefilled rows; qwen4_exp_mtp.py:137-159). ``shifted`` uses
``e(x_{j+1})`` everywhere (the text-only path).

**Training** (plan 3.1-3.5): the 26 dense tensors other than the QSA indexer are FP32 master weights, computed in BF16;
experts (albucino's INT4, dequantized ``q * scale`` in BF16), embedding, lm_head and the target mixer are frozen
buffers. Each optimizer step packs windows up to ``--batch-rows`` rows and runs the chained training-time test (step k
uses the MTP's own ``G`` of step k-1 and the realized token; MTPReplica.chain), with gradient checkpointing per step and
the KL computed in row chunks (no [rows, 65,536] tensor is kept). Loss = sum_k w_k mean KL_k with w = (0.51, 0.31,
0.18). AdamW (0.9, 0.95), weight decay 0, peak lr 5e-5, 5% linear warm-up, cosine to 10%, clip 1.0.

**Outputs** (``--out``): ``trained-dense.safetensors`` (the trained tensors in BF16, the names and shapes of albucino's
mtp-dense.safetensors; scripts/mtp_write_draft.py builds the draft directory from it), ``ckpt.pt`` (resume with
``--resume``), ``train-log.jsonl``, ``eval-*.json`` and ``train-report.json``. Evaluation (held-out windows, original
weights against trained ones, same rows): per step the KL, top-1 agreement with the target's argmax (hot and full
vocabulary) and with the realized token, and three accept-length proxies along the chain: ``realized`` (leading
proposals equal to the realized tokens), ``greedy`` (leading proposals equal to the target's argmax while the data path
is the greedy one) and ``expected`` (lossless speculative sampling at ``--eval-temperature`` 0.7, top-k 20, top-p 0.95:
step k accepted with the target's processed probability of the proposal; per-step rates estimated where the chain
can be followed, i.e. earlier proposals equal the realized tokens; accept length 1 + a1 + a1 a2 + a1 a2 a3).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtp_replica as mr  # noqa: E402

hd = mr.hd
dd = mr._sibling("hc_dump_driver")  # HOLDOUT and the rid naming (stdlib)

STEP_WEIGHTS = (0.51, 0.31, 0.18)


# --------------------------------------------------------------------------------------------- the dump, as windows


@dataclasses.dataclass
class DumpRequest:
    dump: Path
    rid: str
    game: str
    split: str
    length: int                  # positions 0..length-1 dumped
    spans: dict                  # index -> [index, start, end, think, closed_by]
    loss_spans: list | None      # the plan's spans (None: all)
    context: int                 # the dump's ARC3_HC_DUMP_CONTEXT
    keep: str
    mm: bool                     # a chunk carried mm_input_embeds (the request has images)
    chunks: list = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Window:
    req: DumpRequest
    lo: int        # first row (absolute position)
    hi: int        # last row, inclusive
    o_lo: int      # first origin with a loss
    o_hi: int      # last origin
    span: tuple    # (start, end) of the span

    @property
    def rows(self) -> int:
        return self.hi - self.lo + 1


def _snapshot_meta(snapshot_files) -> dict:
    meta = {}
    for path in snapshot_files or ():
        for line in Path(path).read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                meta[s["rid"]] = {"game": s.get("game"), "split": s.get("split"), "loss_spans": s.get("loss_spans")}
    return meta


def _game_of(rid: str) -> str | None:
    parts = rid.split("-")
    return parts[2] if len(parts) >= 5 and parts[0] == "hc" else None


def read_dump(dump_dir, meta: dict | None = None) -> list[DumpRequest]:
    """The requests of one dump directory from its index.jsonl (the latest attempt of each; requests whose chunks do
    not cover 0..n contiguously or that hit an error are left out)."""
    by_rid = collections.defaultdict(list)
    broken = set()
    for line in hd.read_index(dump_dir):
        if line.get("event") == "error":
            broken.add(line.get("rid"))
        elif "file" in line:
            by_rid[line["rid"]].append(line)
    out = []
    for rid, lines in by_rid.items():
        if rid in broken:
            continue
        attempt = max(x["attempt"] for x in lines)
        lines = sorted((x for x in lines if x["attempt"] == attempt), key=lambda x: x["chunk"])
        covered = 0
        for x in lines:
            if x["start"] != covered:
                covered = -1
                break
            covered += x["n"]
        if covered <= 0:
            continue
        spans = {}
        for x in lines:
            for index, s, e, think, why in x["spans"]:
                if index not in spans or e >= 0:
                    spans[index] = [index, s, e, think, why]
        m = (meta or {}).get(rid, {})
        game = m.get("game") or _game_of(rid) or "?"
        split = m.get("split") or ("holdout" if game in dd.HOLDOUT else "train")
        out.append(DumpRequest(Path(dump_dir), rid, game, split, covered, spans, lines[-1].get("loss_spans"),
                               int(lines[-1].get("context", hd.DEFAULT_CONTEXT)), lines[-1].get("keep", "spans"),
                               any(x.get("mm_embeds") for x in lines), lines))
    return sorted(out, key=lambda r: r.rid)


def make_windows(req: DumpRequest, *, max_rows: int = 2048, context: int = 256, steps: int = 3) -> list[Window]:
    """Windows over the request's loss spans (module docstring)."""
    out = []
    kept_ctx = req.length if req.keep == "all" else req.context
    for index in sorted(req.spans):
        _, s, e, _, _ = req.spans[index]
        if e < 0 or e - s < 1 or (req.loss_spans is not None and index not in req.loss_spans):
            continue
        first = max(0, s - min(kept_ctx, context))  # first dumped row the span may use
        o_lo, o_hi = max(s - 1, first), e - 2
        a = o_lo
        while a <= o_hi:
            lo = max(first, a - context)
            budget = max_rows - (a - lo) - steps
            if budget < 1:
                raise ValueError(f"--max-rows {max_rows} leaves no origin after {context} rows of context")
            b = min(o_hi, a + budget - 1)
            out.append(Window(req, lo, min(e, b + steps), a, b, (s, e)))
            a = b + 1
    return out


class Store:
    """Loaded dump requests (numpy, as written), by (dump, rid)."""

    def __init__(self, cache: int = 0):
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.limit = cache

    def get(self, req: DumpRequest) -> dict:
        key = (str(req.dump), req.rid)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        data = hd.load_request(req.dump, req.rid)
        data["img"] = ({int(p): i for i, p in enumerate(data["img_pos"].tolist())}
                       if data["img_pos"] is not None else {})
        self.cache[key] = data
        if self.limit and len(self.cache) > self.limit:
            self.cache.popitem(last=False)
        return data


def row_embedding_plan(req: DumpRequest, lo: int, hi: int, mode: str) -> np.ndarray:
    """Per row of [lo, hi]: 1 = e(x_{j+1}) (shifted), 0 = the target's input embedding of x_j (``turn`` mode)."""
    n = hi - lo + 1
    if mode == "shifted" or not req.mm:
        return np.ones(n, np.int8)
    if mode != "turn":
        raise ValueError(f"unknown --embed {mode!r}")
    shifted = np.zeros(n, np.int8)
    for _, s, e, _, _ in req.spans.values():
        end = e if e >= 0 else req.length - 1
        a, b = max(lo, s - 1), min(hi, end)
        if a <= b:
            shifted[a - lo:b - lo + 1] = 1
    return shifted


@dataclasses.dataclass
class Batch:
    H: torch.Tensor          # [n, W]
    emb1: torch.Tensor       # [n, hidden]
    pos: torch.Tensor        # [n]
    bounds: list             # [(a, b)] per window
    chain_ids: torch.Tensor  # [n, steps-1]
    valid: torch.Tensor      # [n, steps] bool: origin row i has a loss at step k
    target_row: torch.Tensor  # [n, steps] row index of H_{t+k} (valid rows only)
    realized: torch.Tensor   # [n, steps] x_{t+k+1}
    windows: list
    stats: dict


def build_batch(model: mr.MTPReplica, store: Store, windows: list, *, device, steps: int = 3, mode: str = "turn",
                hot_mask=None) -> Batch:
    """Tensors for a packed list of windows (rows of each window contiguous, in position order)."""
    Hs, embs, poss, chains, valids, trows, reals, bounds = [], [], [], [], [], [], [], []
    stats = collections.Counter()
    start = 0
    for w in windows:
        data = store.get(w.req)
        hc_pos = data["hc_pos"]
        i0 = int(np.searchsorted(hc_pos, w.lo))
        i1 = i0 + w.rows
        if i1 > len(hc_pos) or not np.array_equal(hc_pos[i0:i1], np.arange(w.lo, w.hi + 1)):
            raise ValueError(f"{w.req.rid}: rows {w.lo}..{w.hi} are not all in the dump")
        ids = np.asarray(data["token_ids"], np.int64)
        L = len(ids)
        H = mr.rows_to_tensor(data["hc"][i0:i1], None if data["hc_scale"] is None else data["hc_scale"][i0:i1],
                              device, model.dt)
        rows = np.arange(w.lo, w.hi + 1)
        plan = row_embedding_plan(w.req, w.lo, w.hi, mode)
        nxt = np.where(rows + 1 < L, ids[np.minimum(rows + 1, L - 1)], 0)
        emb_ids = np.where(plan == 1, nxt, ids[rows])
        emb = model.embed(torch.as_tensor(emb_ids, device=device))
        if data["img"]:
            for j in np.flatnonzero(plan == 0):
                p = int(rows[j])
                if ids[p] >= model.cfg.vocab or ids[p] >= hd.MM_PAD_MIN:
                    if p in data["img"]:
                        k = data["img"][p]
                        vec = data["img_embeds"][k:k + 1]
                        emb[j] = mr.rows_to_tensor(vec, None, device, model.dt)[0]
                        stats["image_rows"] += 1
                    else:
                        stats["image_rows_missing"] += 1
        chain = np.stack([np.where(rows + k < L, ids[np.minimum(rows + k, L - 1)], 0) for k in range(2, steps + 1)],
                         axis=1) if steps > 1 else np.zeros((len(rows), 0), np.int64)
        valid = np.zeros((len(rows), steps), bool)
        trow = np.zeros((len(rows), steps), np.int64)
        real = np.zeros((len(rows), steps), np.int64)
        s, e = w.span
        for k in range(1, steps + 1):
            t = rows
            ok = (t >= w.o_lo) & (t <= w.o_hi) & (t + k + 1 <= e) & (t + k <= w.hi)
            tok = np.where(t + k + 1 < L, ids[np.minimum(t + k + 1, L - 1)], 0)
            if hot_mask is not None:
                in_hot = hot_mask[np.clip(tok, 0, len(hot_mask) - 1)] & (tok < len(hot_mask))
                stats[f"step{k}_not_hot"] += int((ok & ~in_hot).sum())
                ok &= in_hot
            valid[:, k - 1] = ok
            trow[:, k - 1] = start + np.arange(len(rows)) + k
            real[:, k - 1] = tok
        Hs.append(H)
        embs.append(emb)
        poss.append(torch.as_tensor(rows, device=device))
        chains.append(torch.as_tensor(chain, device=device))
        valids.append(valid)
        trows.append(trow)
        reals.append(real)
        bounds.append((start, start + len(rows)))
        start += len(rows)
    stats["rows"] = start
    valid = torch.as_tensor(np.concatenate(valids), device=device)
    for k in range(steps):
        stats[f"step{k + 1}_loss_rows"] = int(valid[:, k].sum())
    return Batch(torch.cat(Hs), torch.cat(embs), torch.cat(poss), bounds, torch.cat(chains), valid,
                 torch.as_tensor(np.concatenate(trows), device=device), torch.as_tensor(np.concatenate(reals),
                                                                                        device=device),
                 list(windows), dict(stats))


def pack(windows: list, batch_rows: int, rng: random.Random | None = None) -> list[list]:
    order = list(windows)
    if rng is not None:
        rng.shuffle(order)
    batches, cur, rows = [], [], 0
    for w in order:
        if cur and rows + w.rows > batch_rows:
            batches.append(cur)
            cur, rows = [], 0
        cur.append(w)
        rows += w.rows
    if cur:
        batches.append(cur)
    return batches


# ------------------------------------------------------------------------------------------- the target's head


class TargetHead(torch.nn.Module):
    """The target's final mixer and lm_head (frozen): logits of the token after each dumped row."""

    def __init__(self, cfg: mr.MTPConfig, mixer: dict, lm_head, hot_ids, *, dtype=torch.bfloat16, device="cpu",
                 full_vocab: bool = True):
        super().__init__()
        self.cfg, self.dt = cfg, dtype
        for part in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"):
            self.register_buffer(part.split(".")[0], mixer[part].to(device, dtype), persistent=False)
        hot = torch.as_tensor(hot_ids, dtype=torch.int64, device=device)
        self.register_buffer("hot_ids", hot, persistent=False)
        self.register_buffer("head_hot", lm_head.to(device)[hot].to(dtype), persistent=False)
        self.register_buffer("head", lm_head.to(device, dtype) if full_vocab else None, persistent=False)

    def mix(self, H):
        c = self.cfg
        Xn = mr.gemma_rmsnorm(H.to(self.dt), self.hc_norm, c.eps, group=c.hidden)
        t = F.silu(F.linear(Xn, self.input_mix_weight_down).float() / c.hc).to(self.dt)
        g = torch.sigmoid(F.linear(t, self.input_mix_weight_up).float())
        return (g.view(-1, c.hc, c.hidden) * Xn.float().view(-1, c.hc, c.hidden)).mean(-2).to(self.dt)

    def hot_logits(self, H):
        return F.linear(self.mix(H), self.head_hot).float()

    def full_logits(self, H):
        return F.linear(self.mix(H), self.head).float()


def load_target_head(cfg: mr.MTPConfig, *, target_dir=None, mixer_file=None, lm_head=None, hot_ids, device,
                     full_vocab: bool = True) -> TargetHead:
    files = []
    if mixer_file:
        files.append(Path(mixer_file))
    reader = mr.SafeReader(files) if files else mr.SafeReader.from_dir(target_dir)
    mixer = {part: reader.get(mr.TARGET_MIXER + part)
             for part in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")}
    if lm_head is None:
        lm_head = reader.get(mr.LM_HEAD_NAME)
    return TargetHead(cfg, mixer, lm_head, hot_ids, device=device, full_vocab=full_vocab)


# ------------------------------------------------------------------------------------------------------- loss


def _kl_rows(m, H_target, model: mr.MTPReplica, target: TargetHead, temperature: float):
    with torch.no_grad():
        logp = torch.log_softmax(target.hot_logits(H_target) / temperature, dim=-1)
    logq = torch.log_softmax(model.logits(m), dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1)


def chain_loss(model: mr.MTPReplica, target: TargetHead, batch: Batch, *, steps: int = 3, weights=STEP_WEIGHTS,
               temperature: float = 1.0, chunk: int = 2048, checkpoint: bool = True):
    """The weighted KL of the chained steps (plan 3.3-3.4); returns (loss, per-step mean KL floats)."""
    outs = model.chain(batch.H, batch.emb1, batch.pos, batch.bounds, batch.chain_ids, steps=steps,
                       checkpoint=checkpoint)
    total, per_step = 0.0, []
    norm = sum(weights[:steps])
    for k in range(steps):
        rows = torch.nonzero(batch.valid[:, k]).flatten()
        if rows.numel() == 0:
            per_step.append(None)
            continue
        acc = 0.0
        for a in range(0, rows.numel(), chunk):
            r = rows[a:a + chunk]
            m, Ht = outs[k].index_select(0, r), batch.H.index_select(0, batch.target_row[r, k])
            if checkpoint and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint as ckpt

                kl = ckpt(_kl_rows, m, Ht, model, target, temperature, use_reentrant=False)
            else:
                kl = _kl_rows(m, Ht, model, target, temperature)
            acc = acc + kl.sum()
        mean = acc / rows.numel()
        per_step.append(float(mean.detach()))
        total = total + (weights[k] / norm) * mean
    return total, per_step


# --------------------------------------------------------------------------------------------------- evaluation


def processed_probs(logits, temperature: float = 0.7, top_k: int = 20, top_p: float = 0.95):
    """The sampler's distribution: softmax(z / T), top-k, then top-p, renormalized (full vocabulary rows)."""
    probs = torch.softmax(logits.float() / temperature, dim=-1)
    vals, idx = probs.topk(top_k, dim=-1)
    vals = vals / vals.sum(-1, keepdim=True)
    keep = (vals.cumsum(-1) - vals) < top_p
    vals = torch.where(keep, vals, torch.zeros_like(vals))
    vals = vals / vals.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(-1, idx, vals)


class EvalAccumulator:
    """Per-step metrics of one weight set over evaluation batches."""

    def __init__(self, steps: int):
        self.steps = steps
        self.s = collections.defaultdict(float)

    def add(self, key: str, value: float, n: float = 1.0):
        self.s[key] += value
        self.s[key + "#n"] += n

    def mean(self, key: str):
        n = self.s.get(key + "#n", 0.0)
        return self.s[key] / n if n else None

    def report(self) -> dict:
        out = {"steps": {}}
        rates = {}
        for k in range(1, self.steps + 1):
            row = {name: self.mean(f"{name}{k}") for name in ("kl", "top1_target_hot", "top1_target_full",
                                                             "top1_realized", "rate_realized", "rate_greedy",
                                                             "rate_expected")}
            row["origins"] = self.s.get(f"kl{k}#n", 0.0)
            out["steps"][k] = row
            for name in ("realized", "greedy", "expected"):
                rates.setdefault(name, []).append(row[f"rate_{name}"])
        for name, rs in rates.items():
            acc, prod = 1.0, 1.0
            for r in rs:
                if r is None:
                    break
                prod *= r
                acc += prod
            out[f"accept_{name}"] = acc
        out["accept_realized_per_origin"] = self.mean("accept_realized_origin")
        return out


@torch.no_grad()
def evaluate(model: mr.MTPReplica, target: TargetHead, batches: list, *, steps: int = 3, temperature: float = 0.7,
             top_k: int = 20, top_p: float = 0.95, chunk: int = 512) -> dict:
    acc = EvalAccumulator(steps)
    for batch in batches:
        outs = model.chain(batch.H, batch.emb1, batch.pos, batch.bounds, batch.chain_ids, steps=steps)
        n = batch.H.shape[0]
        props = torch.zeros(n, steps, dtype=torch.int64, device=batch.H.device)
        greedy = torch.zeros(n, steps, dtype=torch.int64, device=batch.H.device)
        alpha = torch.zeros(n, steps, dtype=torch.float32, device=batch.H.device)
        for k in range(steps):
            rows = torch.nonzero(batch.valid[:, k]).flatten()
            for a in range(0, rows.numel(), chunk):
                r = rows[a:a + chunk]
                m, Ht = outs[k].index_select(0, r), batch.H.index_select(0, batch.target_row[r, k])
                ql = model.logits(m)
                hot = target.hot_logits(Ht)
                logp = torch.log_softmax(hot, -1)
                kl = (logp.exp() * (logp - torch.log_softmax(ql, -1))).sum(-1)
                d = model.propose(ql)
                props[r, k] = d
                acc.add(f"kl{k + 1}", float(kl.sum()), r.numel())
                acc.add(f"top1_target_hot{k + 1}", float((ql.argmax(-1) == hot.argmax(-1)).sum()), r.numel())
                acc.add(f"top1_realized{k + 1}", float((d == batch.realized[r, k]).sum()), r.numel())
                if target.head is not None:
                    full = target.full_logits(Ht)
                    g = full.argmax(-1)
                    greedy[r, k] = g
                    acc.add(f"top1_target_full{k + 1}", float((d == g).sum()), r.numel())
                    alpha[r, k] = processed_probs(full, temperature, top_k, top_p).gather(1, d[:, None]).squeeze(1)
        # chain-following rates: step k counts where steps < k proposed the realized tokens (the data path)
        alive = batch.valid[:, 0].clone()
        alive_g = batch.valid[:, 0].clone()
        lead = torch.zeros(n, dtype=torch.int64, device=batch.H.device)
        for k in range(steps):
            v = batch.valid[:, k] & alive
            hit = props[:, k] == batch.realized[:, k]
            acc.add(f"rate_realized{k + 1}", float((hit & v).sum()), float(v.sum()))
            if target.head is not None:
                acc.add(f"rate_expected{k + 1}", float(alpha[:, k][v].sum()), float(v.sum()))
                vg = batch.valid[:, k] & alive_g
                hit_g = props[:, k] == greedy[:, k]
                acc.add(f"rate_greedy{k + 1}", float((hit_g & vg).sum()), float(vg.sum()))
                alive_g = vg & hit_g & (batch.realized[:, k] == greedy[:, k])
            lead = lead + (v & hit & (lead == k)).long()
            alive = v & hit
        origins = batch.valid[:, 0]
        acc.add("accept_realized_origin", float((1 + lead[origins]).sum()), float(origins.sum()))
    return acc.report()


def compare_reports(before: dict, after: dict) -> dict:
    out = {}
    for key in ("accept_realized", "accept_greedy", "accept_expected"):
        if before.get(key) is not None and after.get(key) is not None:
            out[key] = {"original": before[key], "trained": after[key], "delta": after[key] - before[key]}
    for k in before["steps"]:
        for name in ("kl", "top1_target_full", "top1_realized", "rate_expected"):
            b, a = before["steps"][k].get(name), after["steps"][k].get(name)
            if b is not None and a is not None:
                out[f"step{k}_{name}"] = {"original": b, "trained": a, "delta": a - b}
    return out


# ------------------------------------------------------------------------------------------------------ training


def lr_lambda(total: int, warmup: float = 0.05, floor: float = 0.1):
    warm = max(1, int(round(total * warmup)))

    def f(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return f


def collect_windows(args) -> tuple[list, list, list]:
    meta = _snapshot_meta(args.snapshots)
    reqs = []
    for d in args.dump:
        reqs += read_dump(d, meta)
    windows = []
    for r in reqs:
        windows += make_windows(r, max_rows=args.max_rows, context=args.context, steps=args.steps)
    train = [w for w in windows if w.req.split != "holdout"]
    held = [w for w in windows if w.req.split == "holdout"]
    return reqs, train, held


def summarize_windows(reqs, train, held) -> dict:
    def part(ws):
        games = collections.Counter(w.req.game for w in ws)
        return {"windows": len(ws), "rows": sum(w.rows for w in ws),
                "origins": sum(w.o_hi - w.o_lo + 1 for w in ws), "games": dict(sorted(games.items()))}

    return {"requests": len(reqs), "train": part(train), "holdout": part(held)}


def subsample(windows: list, max_rows: int | None, seed: int) -> list:
    if not max_rows:
        return windows
    rng = random.Random(seed)
    order = list(windows)
    rng.shuffle(order)
    out, rows = [], 0
    for w in order:
        if rows + w.rows > max_rows and out:
            break
        out.append(w)
        rows += w.rows
    return out


def build_everything(args, device):
    weights = mr.load_draft_dir(args.draft, device=device)
    hot = mr.load_token_map(args.token_map, weights.cfg.vocab)
    model = mr.MTPReplica(weights, hot_ids=hot, compute_dtype=torch.bfloat16 if device != "cpu" or not args.fp32 else
                          torch.float32, fp8_kv=not args.no_fp8_kv, device=device)
    target = load_target_head(weights.cfg, target_dir=args.target_dir, mixer_file=args.target_mixer,
                              lm_head=weights.lm_head, hot_ids=hot, device=device)
    target.dt = model.dt
    hot_mask = np.zeros(weights.cfg.vocab, bool)
    hot_mask[hot.cpu().numpy()] = True
    del weights
    return model, target, hot_mask if args.require_hot else None


def save_dense(model: mr.MTPReplica, path: Path, names=None) -> None:
    state = model.dense_state(torch.bfloat16)
    keep = names or model.trainable
    mr.save_safetensors(path, {n: state[n] for n in model.names if n in set(keep)},
                        {"format": "arc3-mtp-trained-dense", "source": "scripts/mtp_train.py"})


def train(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reqs, train_w, held_w = collect_windows(args)
    info = summarize_windows(reqs, train_w, held_w)
    print(json.dumps(info), flush=True)
    if not train_w:
        raise SystemExit("no training windows (check --dump/--snapshots and the split)")
    model, target, hot_mask = build_everything(args, device)
    store = Store()
    eval_packs = pack(subsample(held_w, args.eval_rows, args.seed), args.batch_rows) if held_w else []

    def eval_batches():
        return (build_batch(model, store, b, device=device, steps=args.steps, mode=args.embed, hot_mask=hot_mask)
                for b in eval_packs)

    rows_per_epoch = sum(w.rows for w in train_w)
    steps_per_epoch = len(pack(train_w, args.batch_rows))
    total = args.max_steps or int(math.ceil(steps_per_epoch * args.epochs))
    params = [p for n, p in model.p.items() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0, eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total, args.warmup, args.lr_floor))
    step, epoch, done_in_epoch = 0, 0, 0
    ckpt_path = out / "ckpt.pt"
    if args.resume and ckpt_path.is_file():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_dense({n: t for n, t in state["dense"].items()})
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        step, epoch, done_in_epoch = state["step"], state["epoch"], state["done_in_epoch"]
        print(f"resumed at step {step} (epoch {epoch}, {done_in_epoch} batches into it)", flush=True)
    report = {"windows": info, "rows_per_epoch": rows_per_epoch, "steps_per_epoch": steps_per_epoch, "total_steps": total,
              "args": {k: str(v) for k, v in vars(args).items()}}
    if eval_packs and step == 0:
        t0 = time.time()
        report["eval_original"] = evaluate(model, target, eval_batches(), steps=args.steps,
                                           temperature=args.eval_temperature)
        report["eval_original"]["seconds"] = round(time.time() - t0, 1)
        (out / "eval-original.json").write_text(json.dumps(report["eval_original"], indent=1) + "\n")
        print("eval original:", json.dumps({k: v for k, v in report["eval_original"].items() if k != "steps"}),
              flush=True)
    log = open(out / "train-log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    while step < total:
        rng = random.Random(args.seed * 1000 + epoch)
        batches = pack(train_w, args.batch_rows, rng)
        for bi, windows in enumerate(batches):
            if bi < done_in_epoch:
                continue
            if step >= total:
                break
            t0 = time.time()
            batch = build_batch(model, store, windows, device=device, steps=args.steps, mode=args.embed,
                                hot_mask=hot_mask)
            model.train()
            loss, per_step = chain_loss(model, target, batch, steps=args.steps, temperature=args.target_temperature,
                                        checkpoint=not args.no_checkpoint)
            if not torch.is_tensor(loss):  # no step of this batch has a loss row
                done_in_epoch = bi + 1
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip))
            opt.step()
            sched.step()
            step += 1
            done_in_epoch = bi + 1
            line = {"step": step, "epoch": epoch, "loss": float(loss.detach()), "kl": per_step, "grad_norm": gnorm,
                    "lr": sched.get_last_lr()[0], "rows": batch.stats["rows"],
                    "loss_rows": [batch.stats.get(f"step{k + 1}_loss_rows", 0) for k in range(args.steps)],
                    "seconds": round(time.time() - t0, 3),
                    "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device == "cuda" else None}
            log.write(json.dumps(line) + "\n")
            log.flush()
            if step % args.log_every == 0 or step == 1:
                print(json.dumps(line), flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                _checkpoint(model, opt, sched, step, epoch, done_in_epoch, ckpt_path)
                save_dense(model, out / "trained-dense.safetensors")
            if eval_packs and args.eval_every and step % args.eval_every == 0 and step < total:
                ev = evaluate(model, target, eval_batches(), steps=args.steps, temperature=args.eval_temperature)
                (out / f"eval-step{step}.json").write_text(json.dumps(ev, indent=1) + "\n")
                print(f"eval step {step}:", json.dumps({k: v for k, v in ev.items() if k != "steps"}), flush=True)
        else:
            epoch += 1
            done_in_epoch = 0
    log.close()
    _checkpoint(model, opt, sched, step, epoch, done_in_epoch, ckpt_path)
    save_dense(model, out / "trained-dense.safetensors")
    report["train_seconds"] = round(time.time() - t_start, 1)
    report["steps_done"] = step
    if eval_packs:
        report["eval_trained"] = evaluate(model, target, eval_batches(), steps=args.steps,
                                          temperature=args.eval_temperature)
        (out / "eval-trained.json").write_text(json.dumps(report["eval_trained"], indent=1) + "\n")
        if "eval_original" in report:
            report["comparison"] = compare_reports(report["eval_original"], report["eval_trained"])
            print("comparison:", json.dumps(report["comparison"]), flush=True)
    (out / "train-report.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def _checkpoint(model, opt, sched, step, epoch, done_in_epoch, path: Path) -> None:
    state = {"dense": {n: model.p[mr._key(n)].detach().cpu() for n in model.names}, "opt": opt.state_dict(),
             "sched": sched.state_dict(), "step": step, "epoch": epoch, "done_in_epoch": done_in_epoch}
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def run_eval(args) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reqs, train_w, held_w = collect_windows(args)
    model, target, hot_mask = build_everything(args, device)
    windows = subsample(held_w or train_w, args.eval_rows, args.seed)
    store, packs = Store(), pack(windows, args.batch_rows)

    def batches():
        return (build_batch(model, store, b, device=device, steps=args.steps, mode=args.embed, hot_mask=hot_mask)
                for b in packs)

    before = evaluate(model, target, batches(), steps=args.steps, temperature=args.eval_temperature)
    result = {"windows": summarize_windows(reqs, train_w, held_w), "original": before,
              "split": "holdout" if held_w else "train (no holdout windows in the dump)"}
    if args.trained:
        model.load_dense(mr.load_dense_file(args.trained))
        result["trained"] = evaluate(model, target, batches(), steps=args.steps, temperature=args.eval_temperature)
        result["comparison"] = compare_reports(before, result["trained"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval-report.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result.get("comparison") or {k: v for k, v in before.items() if k != "steps"}), flush=True)
    return result


def estimate_memory(rows: int, cfg: mr.MTPConfig = mr.MTPConfig(), steps: int = 3) -> dict:
    """A rough GPU budget (GB) for one packed batch of ``rows`` (plan 3.5)."""
    gb = 1e9
    dense = sum(math.prod(s) for s in mr.dense_shapes(cfg).values())
    return {"trainable_fp32_adam_grads": round(dense * 16 / gb, 2),
            "experts_bf16": round(cfg.experts * 3 * cfg.moe_inter * cfg.hidden * 2 / gb, 2),
            "embed_and_heads_bf16": round(cfg.vocab * cfg.hidden * 2 * 2 / gb + 65536 * cfg.hidden * 4 / gb, 2),
            "activations_checkpointed": round(rows * steps * (6 * cfg.width + 4 * cfg.hidden) * 2 / gb
                                              + rows * cfg.width * 2 * 4 / gb, 2),
            "attention_scores_per_window": round(cfg.heads * 2048 * 2048 * 4 * 2 / gb, 2)}


def plan(args) -> dict:
    reqs, train_w, held_w = collect_windows(args)
    info = summarize_windows(reqs, train_w, held_w)
    batches = len(pack(train_w, args.batch_rows))
    info["batches_per_epoch"] = batches
    info["total_steps"] = args.max_steps or int(math.ceil(batches * args.epochs))
    info["memory_gb"] = estimate_memory(args.batch_rows)
    print(json.dumps(info, indent=1))
    return info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--dump", type=Path, action="append", required=True, help="a dump directory (repeatable)")
        p.add_argument("--snapshots", type=Path, action="append", default=[],
                       help="hc_dump_driver's snapshots.jsonl (game and split of each rid; repeatable)")
        p.add_argument("--max-rows", type=int, default=2048, help="rows per window (QSA is dense up to 2,048)")
        p.add_argument("--context", type=int, default=256, help="rows of left context per window")
        p.add_argument("--steps", type=int, default=3, help="chain steps (the server drafts 3)")
        p.add_argument("--batch-rows", type=int, default=16384, help="rows per optimizer step (packed windows)")
        p.add_argument("--epochs", type=float, default=2)
        p.add_argument("--max-steps", type=int, default=None)
        p.add_argument("--seed", type=int, default=20261008)
        if name == "plan":
            continue
        p.add_argument("--draft", type=Path, required=True, help="albucino's draft directory")
        p.add_argument("--target-dir", type=Path, default=None, help="the target checkpoint (final mixer)")
        p.add_argument("--target-mixer", type=Path, default=None, help="a safetensors file with the target mixer")
        p.add_argument("--token-map", type=Path, required=True, help="the FR-Spec map the server uses")
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--embed", choices=("turn", "shifted"), default="turn")
        p.add_argument("--eval-rows", type=int, default=200_000, help="held-out rows evaluated (0: all)")
        p.add_argument("--eval-temperature", type=float, default=0.7)
        p.add_argument("--require-hot", action=argparse.BooleanOptionalAction, default=True,
                       help="count a step only when its realized token is in the FR-Spec map (plan 3.4)")
        p.add_argument("--no-fp8-kv", action="store_true", help="BF16 K/V (the server's draft KV cache is FP8)")
        p.add_argument("--fp32", action="store_true", help="compute in FP32 (CPU tests)")
        if name == "eval":
            p.add_argument("--trained", type=Path, default=None)
            continue
        p.add_argument("--lr", type=float, default=5e-5)
        p.add_argument("--warmup", type=float, default=0.05)
        p.add_argument("--lr-floor", type=float, default=0.1)
        p.add_argument("--clip", type=float, default=1.0)
        p.add_argument("--target-temperature", type=float, default=1.0)
        p.add_argument("--no-checkpoint", action="store_true", help="no activation checkpointing")
        p.add_argument("--checkpoint-every", type=int, default=100)
        p.add_argument("--eval-every", type=int, default=0)
        p.add_argument("--log-every", type=int, default=10)
        p.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        plan(args)
    elif args.cmd == "train":
        if not args.target_dir and not args.target_mixer:
            ap.error("--target-dir or --target-mixer is required (the target's final mixer)")
        train(args)
    else:
        if not args.target_dir and not args.target_mixer:
            ap.error("--target-dir or --target-mixer is required (the target's final mixer)")
        run_eval(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
