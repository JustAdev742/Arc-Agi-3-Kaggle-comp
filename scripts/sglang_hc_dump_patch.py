"""Dump of the Flash-Next target's hyper-connection states for the MTP draft fine-tune, in Pennyroyal SGLang
(sglang-0.5.19+gd00d88efc8d6; docs/research/beat-tufa/mtp-drafter-finetune.md, sections 2.4 and 9).

One file, two roles (as scripts/sglang_reap_patch.py):

1. **Runtime module.** ``apply`` copies this file into the installed package as ``sglang/srt/arc3_hc_dump.py``. With
   ``ARC3_HC_DUMP=/dir`` in the server's environment, every EXTEND or MIXED forward of the target
   (``Qwen4ExpForConditionalGeneration``; the MTP draft is another class) hands ``self.model.last_hc_hidden_states``
   to :func:`capture`. Per token that is the target's final hyper-connection state: 4 streams x 2560 = 10,240 BF16
   values, the input of its final mixer, which the MTP block consumes as ``H_j``. Per request chunk it keeps
   - the rows of every assistant span: from the token after ``<|im_start|>assistant\\n<think>\\n`` (ids 248045 74455
     198 248068 198 of the served tokenizer, sha256 06b95093...) up to and including ``<|im_end|>`` (248046); the
     exact rule is :func:`scan_spans`;
   - ``ARC3_HC_DUMP_CONTEXT`` rows (default 256) before each span's first row, as attention context. Rows in an
     earlier chunk of the same request come from a ring buffer of the request's last rows;
   and writes one file per chunk (format below) plus one line of ``index.jsonl``. Unset, nothing here runs: the
   patched model file reads the variable once, at import.
2. **Installer** (``python -I scripts/sglang_hc_dump_patch.py apply --site-packages SP``): writes the module and makes
   two anchored edits to ``SP/sglang/srt/models/qwen4_exp.py``: a module-level flag right before
   ``class Qwen4ExpForConditionalGeneration``, and two checks of it in that class's ``forward`` (one before
   ``super().forward``, which keeps the input ids, because the multimodal embedding clamps image rows in place; one
   after it, which dumps). It refuses a file that is neither the analysed wheel's nor that file with
   scripts/sglang_reap_patch.py applied, an anchor that does not occur exactly once, or a partial or different arc3
   HC patch, and exits non-zero. Running it again is a no-op; ``revert`` undoes it. **With REAP, apply REAP first**:
   the REAP installer accepts only the pristine file, so it refuses a file this patch has edited (``revert``, then
   REAP, then this). The edits are disjoint from REAP's, so both orders give the same text.

Server settings for a dump (section 2.4): no CUDA graph for prefill (``--cuda-graph-backend-prefill disabled``; the
deprecated ``--disable-cuda-graph`` disables both phases in this version), so the Python forward runs for every
chunk; ``--disable-radix-cache`` (cached prefix rows are never recomputed, so never dumped); ``--max-running-requests
1``; no speculative decoding; requests with ``max_tokens`` 1 (pure DECODE batches are not dumped). More variables:
``ARC3_HC_DUMP_CONTEXT`` (rows of context, default 256), ``ARC3_HC_DUMP_KEEP=all`` (every row, for the replica check
of section 5.2), ``ARC3_HC_DUMP_DTYPE=bf16`` (raw BF16 rows instead of FP8), ``ARC3_HC_DUMP_SCALE_GROUPS=4`` (one FP8
scale per hyper-connection stream instead of one per row; read ``qerr_stream_max`` in the index first),
``ARC3_HC_DUMP_MAX_GB`` (stop before the files would exceed it; a tmpfs is RAM). A file ``<dir>/plans/<rid>.json`` holding ``{"loss_spans": [k, ...]}``
(scripts/hc_dump_driver.py writes it before it sends request ``rid``) restricts that request's span rows to those
spans (k counts assistant headers in the prompt from 0); the other spans' rows are kept only as context.

Each chunk file ``<dir>/<rid>/c<chunk>-p<start>.safetensors`` uses the safetensors layout (8-byte little-endian header
size, JSON header, little-endian data), so ``safetensors.torch.load_file`` reads it, and so does
:func:`read_safetensors`; its ``__metadata__["arc3_hc_dump"]`` is the chunk's index line plus :data:`TENSORS`, the
description of every tensor. ``<dir>/format.json`` describes the whole dump, :data:`INDEX_FIELDS` the index lines.
:func:`load_request` merges a request's chunks.

Why image rows and chunk boundaries are recorded: for every prefill chunk of a request with images (``contains_mm_inputs``
is per request, so every chunk of it), the MTP draft-extend takes the target's input embeddings
``forward_batch.mm_input_embeds`` as its embedding input, unshifted, and replaces only the chunk's last row by the next
token's embedding (qwen4_exp_mtp.py ``_prepare_input_embeds``); text-only chunks use the shifted token ids. To build
the same draft KV for prompt rows, a trainer needs each chunk's ``start``/``n``, whether it had ``mm_embeds`` (index
line), the token ids, and the vision features at image rows (``img_embeds``). ``mm_input_embeds`` is read after the
forward: without speculative decoding it is the tensor the language model received, which Qwen4ExpModel never writes
in place (its first layer concatenates it into the 4 streams).
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import logging
import os
import re
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

ENV = "ARC3_HC_DUMP"
ENV_CONTEXT = "ARC3_HC_DUMP_CONTEXT"
ENV_KEEP = "ARC3_HC_DUMP_KEEP"
ENV_DTYPE = "ARC3_HC_DUMP_DTYPE"
ENV_MAX_GB = "ARC3_HC_DUMP_MAX_GB"
ENV_SCALE_GROUPS = "ARC3_HC_DUMP_SCALE_GROUPS"
MODEL_FILE = Path("sglang/srt/models/qwen4_exp.py")
MODULE_FILE = Path("sglang/srt/arc3_hc_dump.py")
# sglang/srt/models/qwen4_exp.py in dfranzen/pennyroyal-v253 wheels/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl
# (wheel sha256 d0620216...), as shipped and with scripts/sglang_reap_patch.py applied (the only bases accepted)
BASE_SHA256 = {
    "35a1785ce4c287821f00f1518b3880ab6838ef58d42fd51ece86d8098b2ed666": "the wheel's file",
    "37515e121420e880fd8cd2b5a5b5fab62eb4a3ccf0d6e651b870183843b205e3": "the wheel's file + scripts/sglang_reap_patch.py",
}
MARK = "# >>> arc3 HC dump (scripts/sglang_hc_dump_patch.py)"
END = "# <<< arc3 HC dump"

FORMAT, VERSION = "arc3-hc-dump", 1
# Ids of the served tokenizer (tokenizer.json sha256 06b9509352d2af50..., the TOKENIZER_SHA of Franzen's launcher)
IM_START, IM_END, ASSISTANT, NEWLINE, THINK = 248045, 248046, 74455, 198, 248068
MM_PAD_MIN = 1_000_000  # schedule_batch.py MM_PAD_SHIFT_VALUE: an image row's input id is 1e6 + hash % 2**30
FP8_MAX = 448.0  # largest finite float8_e4m3fn
DEFAULT_CONTEXT = 256
QUANT_BLOCK = 1024  # rows gathered and quantized at a time on the GPU (a few 42 MB float32 temporaries)
TENSORS = {
    "token_ids": "I32 [n]: the chunk's input ids as the scheduler sent them; an image row holds SGLang's pad value "
                 "(>= 1,000,000, i.e. >= the vocabulary size) instead of <|image_pad|>",
    "positions": "I32 [n]: absolute position of each chunk row (forward_batch.positions)",
    "mrope_positions": "I32 [3, n]: M-RoPE (t, h, w) position ids of each chunk row (forward_batch.mrope_positions); "
                       "absent when the batch has none",
    "hc_pos": "I32 [k]: positions of the kept rows, ascending; rows before the chunk's first position were held in "
              "the request's ring buffer and are written with the chunk that needed them",
    "hc_role": "U8 [k]: 1 = row of a kept assistant span, 0 = context row",
    "hc": "F8_E4M3 [k, 10240] (dtype fp8) or BF16 [k, 10240] (dtype bf16): kept rows of the target's "
          "last_hc_hidden_states, 4 hyper-connection streams x 2560, stream-major",
    "hc_scale": "BF16 [k] or [k, G] (fp8 only): the scale of each row, or of each of its G equal column groups "
                "(ARC3_HC_DUMP_SCALE_GROUPS; 4 = one per stream); value = float(hc) * float(its scale), scale = "
                "max|row or group| / 448",
    "img_pos": "I32 [m]: positions of the image rows among the kept rows",
    "img_embeds": "BF16 [m, hidden]: the target's input embedding at those rows (forward_batch.mm_input_embeds, i.e. "
                  "the vision features it consumed); text rows' embeddings follow from token_ids",
}
INDEX_FIELDS = {
    "rid, dir, file": "request id, its directory (safe_rid) and the chunk's file, relative to the dump directory",
    "attempt, chunk": "a request prefilled again from an earlier position starts a new attempt; chunk counts from 0",
    "start, n": "the chunk's first position and its number of rows",
    "kept, carried, span_rows, context_rows": "rows written; carried ones came from the ring (positions before start)",
    "spans": "[index, first row, last row or -1 while open, think 0/1, closed_by] of each span touching the chunk",
    "spans_opened_total": "assistant headers seen so far in the request (after its last chunk: assistant messages + 1)",
    "loss_spans, plan": "the spans the request's plan file keeps (null: all) and that file's name",
    "images, images_kept, mm_embeds, hidden": "image rows in the chunk, image rows written with their embedding, "
                                              "whether the batch had mm_input_embeds, the embedding width",
    "dtype, context, keep, scale_groups": "the settings",
    "qerr_max, qerr_mean, nonfinite_rows": "relative RMS error of the rows quantized for the chunk; rows with NaN/Inf",
    "qerr_stream_max, streams": "the largest relative RMS error of one hyper-connection stream of a row (streams = "
                                "the model's hc_count); the target's final mixer normalizes each stream on its own",
    "missing_prefix": "positions before the request's first chunk, never seen (a radix-cache hit) and not dumped",
    "bytes, bytes_total, t": "the file's size, the dump's size so far, the time",
    "event": "a line with an 'event' key reports a problem: 'error' (that request is no longer dumped) or 'stopped' "
             "(the size cap; nothing more is written)",
}

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------- assistant spans

OUT, START, ROLE, ROLE_NL, AFTER_THINK, SPAN = range(6)


@dataclasses.dataclass
class ScanState:
    """Where a request's token stream stands after the rows scanned so far (kept between its chunks)."""
    phase: int = OUT
    span_start: int = -1    # first row of the open span (phase SPAN); may be one past the last row scanned
    span_think: bool = False
    opened: int = 0         # spans opened so far; the open span is number opened - 1


@dataclasses.dataclass
class Span:
    index: int        # k-th assistant header of the request, from 0
    start: int        # first row (absolute position)
    end: int          # last row, inclusive; -1 while the span is still open at the end of the scanned chunk
    think: bool       # the header had <think>
    closed_by: str    # "im_end", "im_start" (a new header began inside the span) or "" (open)

    def as_list(self) -> list:
        return [self.index, self.start, self.end, int(self.think), self.closed_by]


def scan_spans(ids, start: int, state: ScanState) -> tuple[list[Span], ScanState]:
    """The assistant spans that touch the chunk ``ids`` (rows ``start``..``start+len(ids)-1``), and the next state.

    A span starts after ``<|im_start|> assistant \\n <think>`` plus one ``\\n``; when the token after ``<think>`` is
    not ``\\n`` (``<think>\\n\\n</think>``, empty reasoning, renders as one ``\\n\\n`` token) the span starts at it, and
    an assistant header without ``<think>`` opens a span at the token after ``assistant \\n`` (``think`` False). The
    span ends at ``<|im_end|>`` inclusive. ``<|im_start|>`` inside a span ends it at the row before (closed_by
    "im_start") and starts a new header. A span opened by the chunk's last header token starts one past the chunk
    (``start + len(ids)``), with no rows here; it is reported again by the next chunk. Pure: ``state`` is not changed.
    """
    st = dataclasses.replace(state)
    spans: dict[int, Span] = {}
    if st.phase == SPAN:
        spans[st.opened - 1] = Span(st.opened - 1, st.span_start, -1, st.span_think, "")

    def open_span(first: int, think: bool) -> None:
        st.opened += 1
        st.phase, st.span_start, st.span_think = SPAN, first, think
        spans[st.opened - 1] = Span(st.opened - 1, first, -1, think, "")

    def close_span(last: int, why: str) -> None:
        span = spans[st.opened - 1]
        span.end, span.closed_by = last, why

    for i, tok in enumerate(ids.tolist() if hasattr(ids, "tolist") else list(ids)):
        pos = start + i
        if st.phase == SPAN:
            if tok == IM_END:
                close_span(pos, "im_end")
                st.phase = OUT
            elif tok == IM_START:
                close_span(pos - 1, "im_start")
                st.phase = START
            continue
        if tok == IM_START:
            st.phase = START
        elif st.phase == START:
            st.phase = ROLE if tok == ASSISTANT else OUT
        elif st.phase == ROLE:
            st.phase = ROLE_NL if tok == NEWLINE else OUT
        elif st.phase in (ROLE_NL, AFTER_THINK):
            if st.phase == ROLE_NL and tok == THINK:
                st.phase = AFTER_THINK
            elif st.phase == AFTER_THINK and tok == NEWLINE:
                open_span(pos + 1, True)
            else:  # the span's first token is this one
                open_span(pos, st.phase == AFTER_THINK)
                if tok == IM_END:
                    close_span(pos, "im_end")
                    st.phase = OUT
    return [spans[k] for k in sorted(spans)], st


def select_rows(spans: list[Span], start: int, n: int, context: int, *, keep_all: bool = False,
                loss_spans=None) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Which rows of the chunk to keep: ``role`` [n] (-1 not kept, 0 context, 1 span row) and the absolute ranges
    ``[lo, hi)`` before ``start`` whose rows are needed as context (they come from the ring buffer).

    Span rows are the rows of the spans in ``loss_spans`` (every span when None). A span whose first row is in this
    chunk also gets the ``context`` rows before that row. ``keep_all`` keeps every row (role 1 inside a kept span)."""
    role = np.full(n, -1, np.int8)
    before: list[tuple[int, int]] = []
    last = start + n - 1
    for span in spans:
        if loss_spans is not None and span.index not in loss_spans:
            continue
        end = span.end if span.end >= 0 else last
        lo, hi = max(span.start, start), min(end, last)
        if lo <= hi:
            role[lo - start:hi - start + 1] = 1
        if start <= span.start <= last and span.end != span.start - 1 and context > 0:
            c0 = max(0, span.start - context)
            ctx = role[max(c0, start) - start:span.start - start]
            ctx[ctx < 0] = 0
            if c0 < start:
                before.append((c0, start))
    if keep_all:
        role[role < 0] = 0
        before = []
    return role, before


# --------------------------------------------------------------------------------------- number formats (numpy side)


def bf16_bits(x) -> np.ndarray:
    """float32 -> bfloat16 bit patterns (uint16), round to nearest even, as torch's ``.to(torch.bfloat16)``."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    out = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(np.asarray(x, np.float32))
    return np.where(nan, np.uint16(0x7FC0), out)


def bf16_value(bits) -> np.ndarray:
    return (np.asarray(bits, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _fp8_table() -> np.ndarray:
    values = np.empty(256, np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        exp, man = (code >> 3) & 0xF, code & 0x7
        if exp == 0xF and man == 0x7:
            values[code] = np.nan
        elif exp == 0:
            values[code] = sign * man * 2.0 ** -9
        else:
            values[code] = sign * (1 + man / 8) * 2.0 ** (exp - 7)
    return values


FP8_VALUES = _fp8_table()   # float8_e4m3fn code -> value
_FP8_POS = FP8_VALUES[:127]  # codes 0..126: 0 .. 448, increasing


def fp8_bits(x) -> np.ndarray:
    """float32 -> float8_e4m3fn codes (uint8), round to nearest even, bit-identical to torch 2.14's
    ``.to(torch.float8_e4m3fn)``: |x| beyond 448 (Inf included) saturates to +-448, NaN gives 0x7F. Older torch
    releases turned |x| > 464 into NaN instead; the dump clamps to +-448 before converting, so both agree there."""
    x = np.asarray(x, np.float32)
    a = np.abs(x)
    hi = np.clip(np.searchsorted(_FP8_POS, a, side="left"), 0, 126)
    lo = np.clip(hi - 1, 0, 126)
    with np.errstate(invalid="ignore"):
        d_lo, d_hi = a - _FP8_POS[lo], _FP8_POS[hi] - a
    code = np.where((d_hi < d_lo) | ((d_hi == d_lo) & (hi % 2 == 0)), hi, lo).astype(np.uint8)
    code = (code | (np.signbit(x).astype(np.uint8) << 7)).astype(np.uint8)
    return np.where(np.isnan(a), np.uint8(0x7F), code)


def fp8_value(codes) -> np.ndarray:
    return FP8_VALUES[np.asarray(codes, np.uint8)]


def dequantize(codes, scale_bits=None) -> np.ndarray:
    """Rows of a chunk file as float32: FP8 codes times their BF16 scale (one per row [k], or per column group
    [k, G]), or BF16 bits when there is no scale."""
    if scale_bits is None:
        return bf16_value(codes)
    values, scale = fp8_value(codes), bf16_value(scale_bits)
    if scale.ndim == 1:
        return values * scale[:, None]
    return (values.reshape(len(values), scale.shape[1], -1) * scale[:, :, None]).reshape(values.shape)


# ----------------------------------------------------------------------------------------------- torch / numpy rows


def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _numpy(x) -> np.ndarray | None:
    if x is None:
        return None
    if _is_torch(x):
        return x.detach().to("cpu").numpy()
    return np.asarray(x)


def _take(x, idx: np.ndarray):
    if _is_torch(x):
        import torch

        return x.index_select(0, torch.as_tensor(idx, dtype=torch.long, device=x.device))
    return np.asarray(x)[idx]


def _bf16_rows(x) -> np.ndarray:
    """Rows (torch, any float dtype; or numpy float) as BF16 bit patterns, uint16."""
    if _is_torch(x):
        import torch

        return x.to(torch.bfloat16).contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
    return bf16_bits(np.asarray(x, np.float32))


def quantize_rows(x, dtype: str = "fp8", *, groups: int = 1, streams: int = 1) -> dict:
    """``x`` [k, W] (torch on any device, or numpy float32) -> codes on the CPU and error statistics.

    fp8: each row is cut into ``groups`` equal column groups (1: the whole row; 4: one per hyper-connection stream),
    ``scale = bf16(max|group| / 448)`` (1 for an all-zero group), ``code = fp8(clamp(value / scale, +-448))``; returns
    codes uint8 [k, W] and scale bits uint16 [k] (one group) or [k, groups]. bf16: codes are the BF16 bit patterns,
    no scale. qerr_max / qerr_mean: relative RMS error of each dequantized row; qerr_stream_max: the largest relative
    RMS error of one of a row's ``streams`` equal column groups (the target's final mixer normalizes each stream on
    its own, and a small stream next to a large one is what one scale per row can lose); nonfinite: rows with a NaN
    or Inf."""
    k = int(x.shape[0])
    width = int(x.shape[1]) if len(x.shape) > 1 else 0
    if groups < 1 or streams < 1 or width % groups or width % streams:
        raise ValueError(f"rows of width {width} do not split into {groups} scale groups and {streams} streams")
    out = {"codes": np.zeros((0, width), np.uint8 if dtype == "fp8" else np.uint16), "nonfinite": 0,
           "scale": (np.zeros((0, groups) if groups > 1 else 0, np.uint16)) if dtype == "fp8" else None,
           "qerr_max": 0.0, "qerr_mean": 0.0, "qerr_stream_max": 0.0}
    if k == 0:
        return out
    codes, scales, errs, stream_errs = [], [], [], []
    for b in range(0, k, QUANT_BLOCK):
        xb = x[b:b + QUANT_BLOCK]
        n = int(xb.shape[0])
        if _is_torch(xb):
            import torch

            xf = xb.float().reshape(n, width)
            out["nonfinite"] += int((~torch.isfinite(xf)).any(dim=1).sum())
            if dtype == "bf16":
                codes.append(_bf16_rows(xb))
                continue
            xg = xf.reshape(n, groups, width // groups)
            amax = xg.abs().amax(dim=2)
            scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax)).to(torch.bfloat16)
            q = (xg / scale.float().unsqueeze(2)).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            diff = (q.float() * scale.float().unsqueeze(2)).reshape(n, width) - xf
            errs.append((diff.norm(dim=1) / xf.norm(dim=1).clamp_min(1e-30)).cpu().numpy())
            per_stream = (diff.reshape(n, streams, -1).norm(dim=2)
                          / xf.reshape(n, streams, -1).norm(dim=2).clamp_min(1e-30))
            stream_errs.append(per_stream.amax(dim=1).cpu().numpy())
            codes.append(q.reshape(n, width).view(torch.uint8).cpu().numpy())
            scales.append(scale.view(torch.int16).cpu().numpy().view(np.uint16))
        else:
            xf = np.asarray(xb, np.float32).reshape(n, width)
            out["nonfinite"] += int((~np.isfinite(xf)).any(axis=1).sum())
            if dtype == "bf16":
                codes.append(bf16_bits(xf))
                continue
            xg = xf.reshape(n, groups, width // groups)
            amax = np.abs(xg).max(axis=2)
            with np.errstate(invalid="ignore"):
                scale = bf16_bits(np.where(amax > 0, amax / np.float32(FP8_MAX), np.float32(1.0)))
            sf = bf16_value(scale)[:, :, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                q = fp8_bits(np.clip(xg / sf, -FP8_MAX, FP8_MAX))
                diff = (fp8_value(q) * sf).reshape(n, width) - xf
                errs.append(np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(xf, axis=1), 1e-30))
                per_stream = (np.linalg.norm(diff.reshape(n, streams, -1), axis=2)
                              / np.maximum(np.linalg.norm(xf.reshape(n, streams, -1), axis=2), 1e-30))
                stream_errs.append(per_stream.max(axis=1))
            codes.append(q.reshape(n, width))
            scales.append(scale)
    out["codes"] = np.concatenate(codes)
    if dtype == "fp8":
        scale = np.concatenate(scales)
        out["scale"] = scale[:, 0] if groups == 1 else scale
        for key, values in (("qerr", np.concatenate(errs)), ("qerr_stream", np.concatenate(stream_errs))):
            finite = values[np.isfinite(values)]
            if finite.size:
                out[f"{key}_max"] = float(finite.max())
                if key == "qerr":
                    out["qerr_mean"] = float(finite.mean())
    return out


def quantize_take(x, rows: np.ndarray, dtype: str = "fp8", *, groups: int = 1, streams: int = 1) -> dict:
    """:func:`quantize_rows` of ``x[rows]``, gathered QUANT_BLOCK rows at a time (no full copy on the GPU)."""
    parts = [quantize_rows(_take(x, rows[b:b + QUANT_BLOCK]), dtype, groups=groups, streams=streams)
             for b in range(0, len(rows), QUANT_BLOCK)]
    if len(parts) < 2:
        return parts[0] if parts else quantize_rows(_take(x, rows), dtype, groups=groups, streams=streams)
    sizes = [len(p["codes"]) for p in parts]
    return {"codes": np.concatenate([p["codes"] for p in parts]),
            "scale": np.concatenate([p["scale"] for p in parts]) if dtype == "fp8" else None,
            "nonfinite": sum(p["nonfinite"] for p in parts), "qerr_max": max(p["qerr_max"] for p in parts),
            "qerr_mean": sum(p["qerr_mean"] * k for p, k in zip(parts, sizes)) / sum(sizes),
            "qerr_stream_max": max(p["qerr_stream_max"] for p in parts)}


# ------------------------------------------------------------------------------------------------- safetensors files

ST_DTYPES = {"F8_E4M3": "<u1", "BF16": "<u2", "U8": "<u1", "I32": "<i4", "I64": "<i8", "F32": "<f4"}


def encode_safetensors(tensors: list[tuple[str, str, np.ndarray]], metadata: dict[str, str]) -> list[bytes]:
    """``(name, dtype, array)`` (FP8 and BF16 given as their uint8/uint16 bits) in the safetensors layout: the 8-byte
    header size, the JSON header (space-padded to 8 bytes), then each tensor's little-endian bytes."""
    header: dict = {"__metadata__": metadata}
    blobs, offset = [], 0
    for name, dtype, array in tensors:
        data = np.ascontiguousarray(array, dtype=ST_DTYPES[dtype]).tobytes()
        header[name] = {"dtype": dtype, "shape": list(np.shape(array)), "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    head = json.dumps(header, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    return [struct.pack("<Q", len(head)), head, *blobs]


def write_safetensors(path: Path, parts: list[bytes]) -> int:
    """Write :func:`encode_safetensors` output atomically (a temporary file renamed); returns the file size."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        for data in parts:
            f.write(data)
    os.replace(tmp, path)
    return sum(len(p) for p in parts)


def read_safetensors(path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Arrays by name (FP8 as uint8 codes, BF16 as uint16 bits) and the ``__metadata__`` strings."""
    raw = Path(path).read_bytes()
    size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + size])
    meta = header.pop("__metadata__", {}) or {}
    base, arrays = 8 + size, {}
    for name, info in header.items():
        begin, end = info["data_offsets"]
        dtype = np.dtype(ST_DTYPES[info["dtype"]])
        arrays[name] = np.frombuffer(raw, dtype=dtype, count=(end - begin) // dtype.itemsize,
                                     offset=base + begin).reshape(info["shape"])
    return arrays, meta


def safe_rid(rid: str) -> str:
    """A directory name for a request id: unchanged when it is [A-Za-z0-9._-]{1,120} and not dot-only."""
    rid = str(rid)
    if re.fullmatch(r"[A-Za-z0-9._-]{1,120}", rid) and rid.strip("."):
        return rid
    return re.sub(r"[^A-Za-z0-9._-]", "_", rid)[:80].strip(".") + "-" + hashlib.sha256(rid.encode()).hexdigest()[:12]


# ------------------------------------------------------------------------------------------------------- the dumper


@dataclasses.dataclass
class _Ring:
    """One row of a request's recent past: kept rows are only marked, others carry their data."""
    pos: int
    written: bool
    code: np.ndarray | None = None
    scale: np.ndarray | None = None
    embed: np.ndarray | None = None


@dataclasses.dataclass
class _Request:
    rid: str
    dir: str
    scan: ScanState
    ring: collections.deque
    chunks: int = 0
    next_start: int = 0
    attempt: int = 0
    missing_prefix: int = 0
    loss_spans: frozenset | None = None
    plan: str | None = None


class Dumper:
    """Selects, quantizes and writes rows; one per server process (see :func:`capture`). Arrays may be torch tensors
    (on any device) or numpy arrays (tests)."""

    def __init__(self, out: str | os.PathLike, *, context: int = DEFAULT_CONTEXT, keep: str = "spans",
                 dtype: str = "fp8", max_bytes: int = 0, scale_groups: int = 1, streams: int = 1,
                 max_requests: int = 64):
        if (keep not in ("spans", "all") or dtype not in ("fp8", "bf16") or context < 0 or max_bytes < 0
                or scale_groups < 1 or streams < 1):
            raise ValueError(f"arc3 HC dump: bad settings keep={keep!r} dtype={dtype!r} context={context} "
                             f"max_bytes={max_bytes} scale_groups={scale_groups} streams={streams}")
        self.out, self.context, self.keep, self.dtype, self.max_bytes = Path(out), context, keep, dtype, max_bytes
        self.scale_groups, self.streams = scale_groups, streams
        self.requests: collections.OrderedDict[str, _Request] = collections.OrderedDict()
        self.max_requests = max_requests
        self.bytes = self.files = self.rows = 0
        self.stopped: str | None = None
        self.broken: set[str] = set()
        self.lock = threading.Lock()
        self._ready = False

    @classmethod
    def from_env(cls) -> Dumper:
        out = os.environ.get(ENV, "").strip()
        if not out:
            raise ValueError(f"{ENV} is not set")
        try:
            context = int(os.environ.get(ENV_CONTEXT, "") or DEFAULT_CONTEXT)
            max_bytes = int(float(os.environ.get(ENV_MAX_GB, "") or 0) * 1e9)
            groups = int(os.environ.get(ENV_SCALE_GROUPS, "") or 1)
        except ValueError as exc:
            raise ValueError(f"arc3 HC dump: {ENV_CONTEXT}, {ENV_MAX_GB} and {ENV_SCALE_GROUPS} must be numbers "
                             f"({exc})") from None
        return cls(out, context=context, keep=(os.environ.get(ENV_KEEP, "") or "spans").strip(),
                   dtype=(os.environ.get(ENV_DTYPE, "") or "fp8").strip(), max_bytes=max_bytes, scale_groups=groups)

    def settings(self) -> dict:
        return {"context": self.context, "keep": self.keep, "dtype": self.dtype, "max_bytes": self.max_bytes,
                "scale_groups": self.scale_groups, "streams": self.streams}

    def _setup(self) -> None:
        if self._ready:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "plans").mkdir(exist_ok=True)
        fmt = self.out / "format.json"
        if not fmt.exists():
            fmt.write_text(json.dumps({
                "format": FORMAT, "version": VERSION, "settings": self.settings(),
                "tokens": {"im_start": IM_START, "im_end": IM_END, "assistant": ASSISTANT, "newline": NEWLINE,
                           "think": THINK, "image_rows": f"token id >= {MM_PAD_MIN} (or >= the vocabulary size)",
                           "tokenizer_sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"},
                "span_rule": scan_spans.__doc__, "tensors": TENSORS, "index": INDEX_FIELDS,
                "files": "<rid>/c<chunk>-p<start>.safetensors, one per prefill chunk; index.jsonl has one line per "
                         "chunk (and lines with an 'event' key for problems); plans/<rid>.json are the driver's",
                "source": "scripts/sglang_hc_dump_patch.py (installed as sglang/srt/arc3_hc_dump.py)",
                "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}, indent=1) + "\n")
        self._ready = True

    def _event(self, record: dict) -> None:
        with open(self.out / "index.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**record, "t": round(time.time(), 3)}) + "\n")

    def _request(self, rid: str, start: int) -> _Request:
        req = self.requests.get(rid)
        if req is not None:
            self.requests.move_to_end(rid)
            if start == req.next_start:
                return req
            logger.warning("arc3 HC dump: request %s chunk at %d, expected %d; starting a new attempt", rid, start,
                           req.next_start)
            attempt = req.attempt + 1
        else:
            attempt = 0
        req = _Request(rid, safe_rid(rid), ScanState(), collections.deque(maxlen=self.context), attempt=attempt,
                       next_start=start, missing_prefix=start)
        if start:
            logger.warning("arc3 HC dump: request %s starts at position %d (a cached prefix?); rows before it are "
                           "not dumped", rid, start)
        plan = self.out / "plans" / f"{req.dir}.json"
        if plan.is_file():
            spans = json.loads(plan.read_text()).get("loss_spans")
            req.loss_spans = None if spans is None else frozenset(int(k) for k in spans)
            req.plan = plan.name
        self.requests[rid] = req
        while len(self.requests) > self.max_requests:
            self.requests.popitem(last=False)
        return req

    def process_chunk(self, rid: str, start: int, ids, hc, *, positions=None, mrope=None, mm_embeds=None,
                      image_min_id: int = MM_PAD_MIN) -> dict | None:
        """Dump one chunk of one request: ``ids`` [n] (input ids before the forward), ``hc`` [n, W] (its rows of
        last_hc_hidden_states), optional ``positions`` [n], ``mrope`` [3, n], ``mm_embeds`` [n, hidden] (the batch's
        mm_input_embeds rows, or None for a text-only batch). Returns the index line, or None when nothing was
        written (stopped, broken request, or an error, which is logged and recorded in index.jsonl)."""
        with self.lock:
            if self.stopped or rid in self.broken:
                return None
            try:
                self._setup()
                return self._process(rid, int(start), np.asarray(ids, np.int64).reshape(-1), hc, positions, mrope,
                                     mm_embeds, image_min_id)
            except Exception as exc:  # never take the server down; the driver checks every request's coverage
                logger.exception("arc3 HC dump: chunk of %s at %s failed; the request is no longer dumped", rid, start)
                self.broken.add(rid)
                self.requests.pop(rid, None)
                try:
                    self._event({"event": "error", "rid": rid, "start": int(start), "error": repr(exc)[:500]})
                except OSError:
                    logger.exception("arc3 HC dump: cannot write the index")
                return None

    def _process(self, rid, start, ids, hc, positions, mrope, mm_embeds, image_min_id) -> dict | None:
        n = len(ids)
        if int(hc.shape[0]) != n:
            raise ValueError(f"{int(hc.shape[0])} hidden rows for {n} tokens")
        req = self._request(rid, start)
        spans, req.scan = scan_spans(ids, start, req.scan)
        role, before = select_rows(spans, start, n, self.context, keep_all=self.keep == "all",
                                   loss_spans=req.loss_spans)
        kept = np.flatnonzero(role >= 0)
        feed = np.array([i for i in range(max(0, n - self.context), n) if role[i] < 0], np.int64)
        carried = [r for r in req.ring if not r.written and any(lo <= r.pos < hi for lo, hi in before)]
        image = ids >= image_min_id

        rows = np.union1d(kept, feed).astype(np.int64)  # sorted; q's row j is chunk row rows[j]
        streams = self.streams if int(hc.shape[1]) % self.streams == 0 else 1  # a statistic: never fail on it
        q = quantize_take(hc, rows, self.dtype, groups=self.scale_groups, streams=streams)
        emb_rows = rows[image[rows]] if mm_embeds is not None else np.zeros(0, np.int64)
        emb = _bf16_rows(_take(mm_embeds, emb_rows)) if len(emb_rows) else None
        emb_at = {int(i): j for j, i in enumerate(emb_rows)}

        k_idx = np.searchsorted(rows, kept)
        codes = [r.code[None] for r in carried] + [q["codes"][k_idx]]
        hc_codes = np.concatenate(codes) if carried else q["codes"][k_idx]
        hc_pos = np.array([r.pos for r in carried] + [start + int(i) for i in kept], np.int32)
        hc_role = np.concatenate([np.zeros(len(carried), np.uint8), role[kept].astype(np.uint8)])
        img = [(r.pos, r.embed) for r in carried if r.embed is not None]
        img += [(start + int(i), emb[emb_at[int(i)]]) for i in kept if int(i) in emb_at]
        hidden = int(mm_embeds.shape[1]) if mm_embeds is not None else 0

        tensors = [("token_ids", "I32", ids.astype(np.int32)),
                   ("positions", "I32", (_numpy(positions).reshape(-1) if positions is not None
                                         else np.arange(start, start + n)).astype(np.int32))]
        if mrope is not None:
            tensors.append(("mrope_positions", "I32", _numpy(mrope).reshape(3, n).astype(np.int32)))
        tensors += [("hc_pos", "I32", hc_pos), ("hc_role", "U8", hc_role),
                    ("hc", "F8_E4M3" if self.dtype == "fp8" else "BF16", hc_codes)]
        if self.dtype == "fp8":
            scales = np.array([r.scale for r in carried], np.uint16).reshape((len(carried), *q["scale"].shape[1:]))
            tensors.append(("hc_scale", "BF16", np.concatenate([scales, q["scale"][k_idx]])))
        if img:
            tensors += [("img_pos", "I32", np.array([p for p, _ in img], np.int32)),
                        ("img_embeds", "BF16", np.stack([e for _, e in img]).reshape(len(img), -1))]

        name = f"c{req.chunks:05d}-p{start:07d}" + (f"-a{req.attempt}" if req.attempt else "") + ".safetensors"
        record = {
            "rid": rid, "dir": req.dir, "file": f"{req.dir}/{name}", "attempt": req.attempt, "chunk": req.chunks,
            "start": start, "n": n, "kept": len(hc_pos), "carried": len(carried), "span_rows": int((role == 1).sum()),
            "context_rows": int(len(hc_pos) - (role == 1).sum()), "spans": [s.as_list() for s in spans],
            "spans_opened_total": req.scan.opened, "loss_spans": None if req.loss_spans is None else sorted(
                req.loss_spans), "plan": req.plan, "images": int(image.sum()), "images_kept": len(img),
            "mm_embeds": mm_embeds is not None, "hidden": hidden, "dtype": self.dtype, "context": self.context,
            "keep": self.keep, "qerr_max": round(q["qerr_max"], 5), "qerr_mean": round(q["qerr_mean"], 5),
            "qerr_stream_max": round(q["qerr_stream_max"], 5), "streams": streams,
            "scale_groups": self.scale_groups, "nonfinite_rows": q["nonfinite"], "missing_prefix": req.missing_prefix if req.chunks == 0 else 0,
        }
        meta = {"arc3_hc_dump": json.dumps({"format": FORMAT, "version": VERSION, **record, "tensors": TENSORS})}
        parts = encode_safetensors(tensors, meta)
        size = sum(len(p) for p in parts)
        if self.max_bytes and self.bytes + size > self.max_bytes:
            self.stopped = f"the next file would pass {ENV_MAX_GB} ({self.bytes / 1e9:.2f} GB written)"
            logger.warning("arc3 HC dump: stopped: %s", self.stopped)
            self._event({"event": "stopped", "reason": self.stopped, "rid": rid, "start": start,
                         "bytes_total": self.bytes})
            return None
        (self.out / req.dir).mkdir(exist_ok=True)
        record["bytes"] = write_safetensors(self.out / req.dir / name, parts)

        # the request's ring now ends at this chunk's last row; rows written now are only marked
        for r in carried:
            r.written, r.code, r.embed = True, None, None
        for i in range(max(0, n - self.context), n):
            if role[i] >= 0:
                req.ring.append(_Ring(start + i, True))
            else:
                j = int(np.searchsorted(rows, i))
                req.ring.append(_Ring(start + i, False, q["codes"][j].copy(),
                                      np.array(q["scale"][j]) if q["scale"] is not None else None,
                                      emb[emb_at[i]].copy() if i in emb_at else None))
        req.chunks += 1
        req.next_start = start + n
        self.bytes += record["bytes"]
        self.files += 1
        self.rows += len(hc_pos)
        record["bytes_total"] = self.bytes
        self._event(record)
        if self.files == 1 or self.files % 200 == 0:
            logger.info("arc3 HC dump: %d files, %d rows, %.2f GB in %s (last: %s)", self.files, self.rows,
                        self.bytes / 1e9, self.out, record["file"])
        return record


# --------------------------------------------------------------------------------------------- reading a dump back


def read_index(dump_dir) -> list[dict]:
    path = Path(dump_dir) / "index.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_request(dump_dir, rid: str) -> dict:
    """A request's chunks merged (its latest attempt): token_ids / positions [L], mrope_positions [3, L] or None,
    hc_pos / hc_role [K], hc (codes [K, W]) and hc_scale [K] (None for bf16), img_pos [M] / img_embeds [M, hidden]
    (kept rows and image rows sorted by position), spans {index: Span as list, merged over chunks}, chunks (their
    index lines)."""
    lines = [r for r in read_index(dump_dir) if r.get("rid") == rid and "file" in r]
    if not lines:
        raise KeyError(f"no chunks of {rid} in {dump_dir}")
    attempt = max(r["attempt"] for r in lines)
    lines = sorted((r for r in lines if r["attempt"] == attempt), key=lambda r: r["chunk"])
    parts: dict[str, list] = collections.defaultdict(list)
    spans: dict[int, list] = {}
    for line in lines:
        arrays, _ = read_safetensors(Path(dump_dir) / line["file"])
        for key, value in arrays.items():
            parts[key].append(value)
        for index, s, e, think, why in line["spans"]:
            if index not in spans or e >= 0:
                spans[index] = [index, s, e, think, why]
    out: dict = {"chunks": lines, "spans": spans, "dtype": lines[0]["dtype"]}
    for key in ("token_ids", "positions", "hc_pos", "hc_role", "hc_scale", "img_pos"):
        out[key] = np.concatenate(parts[key]) if parts.get(key) else None
    out["mrope_positions"] = np.concatenate(parts["mrope_positions"], axis=1) if parts.get("mrope_positions") else None
    out["hc"] = np.concatenate(parts["hc"]) if parts.get("hc") else None
    out["img_embeds"] = np.concatenate(parts["img_embeds"]) if parts.get("img_embeds") else None
    order = np.argsort(out["hc_pos"], kind="stable")  # rows carried from the ring sit in a later chunk's file
    for key in ("hc_pos", "hc_role", "hc", "hc_scale"):
        if out[key] is not None:
            out[key] = out[key][order]
    if out["img_pos"] is not None:
        order = np.argsort(out["img_pos"], kind="stable")
        out["img_pos"], out["img_embeds"] = out["img_pos"][order], out["img_embeds"][order]
    return out


# ------------------------------------------------------------------------------------- runtime hooks (in SGLang)

_DUMPER: Dumper | None = None
_WARNED: set[str] = set()


def active() -> bool:
    return bool(os.environ.get(ENV, "").strip())


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message, *args)


def _forward_batch(args, kwargs):
    fb = kwargs.get("forward_batch")
    if fb is None and len(args) > 2:
        fb = args[2]
    return fb if hasattr(fb, "forward_mode") else None


def _dumped(fb) -> bool:
    return getattr(getattr(fb, "forward_mode", None), "name", None) in ("EXTEND", "MIXED")


def input_ids_before(args, kwargs):
    """Called before the target's forward: a copy of an EXTEND/MIXED batch's input ids (None otherwise)."""
    fb = _forward_batch(args, kwargs)
    if fb is None or not _dumped(fb):
        return None
    ids = args[0] if args else kwargs.get("input_ids")
    if ids is None:
        ids = getattr(fb, "input_ids", None)
    if ids is None:
        return None
    return ids.clone() if _is_torch(ids) else np.array(ids)


def _lens(fb, cpu_name: str, name: str) -> list[int] | None:
    value = getattr(fb, cpu_name, None)
    if value is None:
        value = getattr(fb, name, None)
        value = None if value is None else _numpy(value).tolist()
    return None if value is None else [int(v) for v in value]


def capture(model, args, kwargs, hc, ids_before) -> None:
    """Called after the target's forward with its ``last_hc_hidden_states``: dump the batch's rows (EXTEND/MIXED)."""
    global _DUMPER
    if ids_before is None or hc is None:
        return
    fb = _forward_batch(args, kwargs)
    if fb is None:
        return
    if _is_torch(hc):
        import torch

        if hc.is_cuda and torch.cuda.is_current_stream_capturing():
            _warn_once("graph", "arc3 HC dump: the target forward runs under CUDA graph capture; nothing is dumped "
                                "for graph-replayed batches (serve with --cuda-graph-backend-prefill disabled)")
            return
        if torch.distributed.is_available() and torch.distributed.is_initialized() and torch.distributed.get_rank():
            return
    config = getattr(model, "config", None)
    width = getattr(config, "hc_count", 0) * getattr(config, "hidden_size", 0)
    starts, lens = _lens(fb, "extend_prefix_lens_cpu", "extend_prefix_lens"), _lens(fb, "extend_seq_lens_cpu",
                                                                                    "extend_seq_lens")
    rids = list(getattr(fb, "rids", None) or [])
    total = sum(lens or [])
    if not lens or not starts or len(rids) != len(lens) or int(hc.shape[0]) < total or len(ids_before) < total:
        _warn_once("shape", "arc3 HC dump: batch layout not understood (rids %s, extend lens %s, %s hidden rows, "
                            "%s ids); skipped", len(rids), lens, int(hc.shape[0]), len(ids_before))
        return
    if width and int(hc.shape[1]) != width:
        _warn_once("width", "arc3 HC dump: hidden rows are %s wide, expected hc_count x hidden_size = %s; skipped",
                   int(hc.shape[1]), width)
        return
    if _DUMPER is None:
        _DUMPER = Dumper.from_env()
        _DUMPER.streams = int(getattr(config, "hc_count", 0) or 1)  # for qerr_stream_max
        logger.info("arc3 HC dump: writing to %s (%s)", _DUMPER.out, _DUMPER.settings())
    dumper = _DUMPER
    if dumper.stopped:
        return
    vocab = getattr(config, "vocab_size", None)
    image_min_id = min(MM_PAD_MIN, int(vocab)) if vocab else MM_PAD_MIN
    ids = _numpy(ids_before[:total]).astype(np.int64)
    positions = getattr(fb, "positions", None)
    positions = _numpy(positions[:total]) if positions is not None and len(positions.shape) == 1 else None
    mrope = getattr(fb, "mrope_positions", None)
    mrope = _numpy(mrope[:, :total]) if mrope is not None and len(mrope.shape) == 2 else None
    mm = getattr(fb, "mm_input_embeds", None)
    offset = 0
    for rid, start, n in zip(rids, starts, lens):
        part = slice(offset, offset + n)
        dumper.process_chunk(str(rid), start, ids[part], hc[part],
                             positions=None if positions is None else positions[part],
                             mrope=None if mrope is None else mrope[:, part],
                             mm_embeds=None if mm is None else mm[part], image_min_id=image_min_id)
        offset += n


if __name__ != "__main__" and active():
    Dumper.from_env()  # bad settings fail at import, i.e. when the server starts, not at its first request


# -------------------------------------------------------------------------------------------------------- installer

CLASS_ANCHOR = "class Qwen4ExpForConditionalGeneration(Qwen3VLForConditionalGeneration):\n"
CLASS_NEW = (f"{MARK}: {ENV} (an output directory) is read once, here;\n"
             "# unset, the two `if _ARC3_HC_DUMP:` checks in Qwen4ExpForConditionalGeneration.forward are the only "
             "change.\n"
             "import os as _arc3_os\n"
             "\n"
             f'_ARC3_HC_DUMP = bool(_arc3_os.environ.get("{ENV}", "").strip())\n'
             "if _ARC3_HC_DUMP:\n"
             "    from sglang.srt import arc3_hc_dump as _arc3_hc_dump\n"
             f"{END}\n"
             "\n"
             "\n" + CLASS_ANCHOR)
FORWARD_ANCHOR = ("    def forward(self, *args, **kwargs):\n"
                  "        output = super().forward(*args, **kwargs)\n"
                  "        hc_hidden_states = self.model.last_hc_hidden_states\n")
FORWARD_NEW = ("    def forward(self, *args, **kwargs):\n"
               f"        {MARK}: keep the batch's input ids (the multimodal\n"
               "        # embedding clamps image rows in place) for the dump below.\n"
               "        if _ARC3_HC_DUMP:\n"
               "            _arc3_hc_ids = _arc3_hc_dump.input_ids_before(args, kwargs)\n"
               f"        {END}\n"
               "        output = super().forward(*args, **kwargs)\n"
               "        hc_hidden_states = self.model.last_hc_hidden_states\n"
               f"        {MARK}: EXTEND/MIXED batches write their\n"
               "        # selected rows of hc_hidden_states (sglang/srt/arc3_hc_dump.py).\n"
               "        if _ARC3_HC_DUMP:\n"
               "            _arc3_hc_dump.capture(self, args, kwargs, hc_hidden_states, _arc3_hc_ids)\n"
               f"        {END}\n")
EDITS = ((CLASS_ANCHOR, CLASS_NEW, "the module flag before class Qwen4ExpForConditionalGeneration"),
         (FORWARD_ANCHOR, FORWARD_NEW, "Qwen4ExpForConditionalGeneration.forward"))


class PatchError(Exception):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def patch_text(text: str, *, check_hash: bool = True) -> str:
    """qwen4_exp.py with both edits; refuses an unknown file version and anchors that do not occur exactly once."""
    if MARK in text:
        raise PatchError("already patched")
    if check_hash and _sha(text) not in BASE_SHA256:
        raise PatchError(f"{MODEL_FILE} is neither the analysed Pennyroyal file nor that file with the REAP patch "
                         f"(sha256 {_sha(text)[:12]}...; accepted: {', '.join(h[:12] for h in BASE_SHA256)}); "
                         "re-check the forward and the anchors before patching another version")
    for anchor, new, what in EDITS:
        if text.count(anchor) != 1:
            raise PatchError(f"anchor for {what} found {text.count(anchor)} times (expected once)")
        text = text.replace(anchor, new)
    compile(text, str(MODEL_FILE), "exec")  # syntax only; nothing is imported or run
    return text


def unpatch_text(text: str) -> str:
    """The file without this patch's edits; refuses a partial or different arc3 HC patch."""
    for anchor, new, what in EDITS:
        if text.count(new) != 1:
            raise PatchError(f"the arc3 HC edit of {what} found {text.count(new)} times (a different or partial "
                             "arc3 HC patch)")
        text = text.replace(new, anchor)
    if MARK in text or END in text:
        raise PatchError("arc3 HC marks remain after removing both edits (a different or partial arc3 HC patch)")
    return text


def _clear_cache(target: Path) -> None:
    for stale in (target.parent / "__pycache__").glob("qwen4_exp.*.pyc"):
        stale.unlink()


def apply(site_packages: Path, *, check_hash: bool = True) -> str:
    """Patch an installed sglang in SITE_PACKAGES (idempotent). Returns what was done."""
    target, module = site_packages / MODEL_FILE, site_packages / MODULE_FILE
    if not target.is_file():
        raise PatchError(f"{target} not found")
    source = Path(__file__).read_text()
    compile(source, str(MODULE_FILE), "exec")
    text = target.read_text()
    if MARK in text:
        original = unpatch_text(text)
        if check_hash and _sha(original) not in BASE_SHA256:
            raise PatchError(f"{target} carries the arc3 HC patch on an unknown base (sha256 {_sha(original)[:12]}...)")
        state = "already patched"
    else:
        original = text
        target.write_text(patch_text(text, check_hash=check_hash))
        _clear_cache(target)
        state = "patched"
    if not module.is_file() or module.read_text() != source:
        module.write_text(source)
    return f"arc3 HC dump: {target} {state} (base: {BASE_SHA256.get(_sha(original), 'not checked')}); module {module}"


def revert(site_packages: Path) -> str:
    """Undo ``apply`` (the module file stays; it does nothing unless imported)."""
    target = site_packages / MODEL_FILE
    if not target.is_file():
        raise PatchError(f"{target} not found")
    text = target.read_text()
    if MARK not in text:
        return f"arc3 HC dump: {target} not patched; nothing to revert"
    target.write_text(unpatch_text(text))
    _clear_cache(target)
    return f"arc3 HC dump: {target} reverted"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("apply", help="patch an installed sglang (site-packages directory)")
    a.add_argument("--site-packages", type=Path, required=True)
    a.add_argument("--allow-other-version", action="store_true", help="skip the file hash check (anchors still apply)")
    r = sub.add_parser("revert", help="undo the patch in an installed sglang")
    r.add_argument("--site-packages", type=Path, required=True)
    c = sub.add_parser("check-wheel", help="apply the edits to the wheel's qwen4_exp.py in memory (no install)")
    c.add_argument("wheel", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "apply":
            print(apply(args.site_packages, check_hash=not args.allow_other_version))
        elif args.cmd == "revert":
            print(revert(args.site_packages))
        else:
            import zipfile

            with zipfile.ZipFile(args.wheel) as wheel:
                text = wheel.read(str(MODEL_FILE)).decode()
            patched = patch_text(text)
            print(f"arc3 HC dump: {args.wheel.name}: {MODEL_FILE} is {BASE_SHA256[_sha(text)]}, both anchors found "
                  f"once, patched file compiles ({len(patched) - len(text)} bytes added)")
    except (PatchError, ValueError, OSError, KeyError, SyntaxError) as exc:
        print(f"arc3 HC dump patch FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
