#!/usr/bin/env python
"""Replica of the Qwen3.8-Flash-Next MTP draft block as Pennyroyal SGLang serves it, and the replica check of the MTP
draft fine-tune (docs/research/beat-tufa/mtp-drafter-finetune.md, sections 1, 3.7, 5.2 and 10).

    python scripts/mtp_replica.py inspect --draft DRAFT_DIR
    python scripts/mtp_replica.py check --draft DRAFT_DIR --token-map hot_tokens_64k.pt --dump PROBE_DUMP \\
        --probe PROBE_OUT/probe-dump.jsonl --reference fidelity.json --out replica-check.json \\
        [--variants full,dense,cut2048,cut256] [--trained trained-dense.safetensors]

Everything here mirrors sglang-0.5.19+gd00d88efc8d6 (dfranzen/pennyroyal-v253, wheel sha256 d0620216...); the line
numbers below are of that wheel's files. ``[S]`` marks a mirrored source line.

**Weights.** The 29 dense ``mtp.*`` tensors of albucino's ``mtp-dense.safetensors`` (= Intel's BF16 original, section 0)
and the 512 x 3 INT4 RTN g32 experts of ``mtp-routed-experts-int4.safetensors`` (``weight_packed`` int32 [out, in/8],
low nibble first, value + 8; ``weight_scale`` BF16 [out, in/32]; ``weight_shape``), dequantized as ``bf16(q * scale)``
(the Marlin W4A16 kernel multiplies the integer by the BF16 scale in BF16). ``embed_tokens``/``lm_head`` are the
target's (eagle_worker_v2.py:297-369 ``set_embed_and_head``); albucino's copies are bit-identical. The head is cut to
the FR-Spec rows (``--token-map``, a list of ids; spec_utils.py:653-681) and a proposal is ``hot_ids[argmax]``
(eagle_worker_v2.py:834, 1145).

**One step of the block** (``MTPReplica._step``), for rows with hidden input ``H`` (10,240 = 4 streams x 2,560) and
embedding ``e``:

1. fusion (qwen4_exp_mtp.py:105-115): ``u = fc_embedding(rms(e))``; ``Hn = rms_10240(H)`` viewed [4, 2560];
   ``X = u + fc_hidden(Hn)`` per stream. Gemma norms ``(1 + w) * x / rms`` in FP32 (layernorm.py:1103-1121).
2. attention gated residual (hyperconnection.py:207-240, 245-368; per-stream norm, hc_per_branch_norm):
   ``a = mean_s(sigmoid(Up(silu(Down(Xn) / 4))) * Xn)``; after attention ``X' = X + o * 2 sigmoid(Inject(Xn) / 4)``.
3. attention (qwen3_5.py:1015-1261, qwen4_exp.py:1706-1757): ``q_proj`` gives per head [q | gate] (qwen3_5.py:1252);
   per-head Gemma q/k norm; NeoX RoPE on the first 64 of 256 dims, base 1e7. **Positions are the absolute 1-D token
   positions**, not M-RoPE: the runner passes ``forward_batch.positions`` (eager_runner.py:251-255, 348-352) and
   ``Qwen4ExpForCausalLMMTP.forward`` never swaps in ``mrope_positions`` (only the VL target does, qwen3_vl.py:1468),
   so every rotary pair of the draft sees the plain position (mrope.py:173-249 with 1-D positions; qwen3_5.py:994).
   K and V are stored in the FP8 e4m3 KV cache by a plain cast (``--speculative-draft-kv-cache-dtype fp8_e4m3``;
   memory_pool.py:2529-2535) and read back from it; scores x 1/16, FP32 softmax; output x sigmoid(gate)
   (qwen4_exp.py:1745-1755); ``o_proj``.
4. MoE gated residual: router softmax, top-10, renormalized (qwen2_moe.py:311-315, ``norm_topk_prob`` True); 512 SwiGLU
   experts; shared expert x ``sigmoid(shared_expert_gate(x))`` added to the routed sum (qwen2_moe.py:520-530, 683-692).
5. ``G`` = the block's 10,240-wide output; the final mixer (a gated residual without combine, qwen4_exp.py:1881-1882)
   gives the 2,560 values for ``lm_head``. ``G`` is the next step's hidden input (``logits_output.hidden_states``,
   qwen4_exp_mtp.py:161-184, eagle_worker_v2.py:835).

**The chain** (eagle_worker_v2.py:691-860, 898-1175): after a prefill or a verify, a draft-extend runs the block over
the newly committed rows ``j`` with ``(H_j, e(x_{j+1}))`` and writes their draft KV; the row of the last accepted
position ``t`` gives proposal 1 and ``G_t``. Step 2 runs at position ``t+1`` (``positions = seq_lens``,
eagle_worker_common.py:303) on ``(G_t, e(proposal 1))``, step 3 at ``t+2``. Steps 2-3 attend to the draft KV of rows
``<= t`` plus their own chain KV.

**Draft-extend embeddings** (qwen4_exp_mtp.py:137-159; correction in section 9.2 of the plan): in an EXTEND batch of a
request with images the draft takes the target's input embeddings ``mm_input_embeds`` *unshifted* (``e(x_j)``, or the
vision features at image rows), and only each chunk's last row gets ``e(x_{j+1})``; text-only requests and the
post-verify DRAFT_EXTEND_V2 batches use ``e(x_{j+1})``. :func:`extend_embeddings` builds both.

**Sparse attention (QSA)** (layers/attention/qsa/*.py, qwen_sparse_attn_backend.py): the indexer projects 4 query heads
and 1 key head of 128 (``index_qk_proj``); raw keys are averaged in FP32 over complete 4-token blocks, Gemma-normed and
roped at the block's first position (qsa_indexer.py:208-214, 388-394; kernel.py:12-20); a row at position ``t`` scores
the ``(t+1)//4`` complete blocks by ``sum_h relu(q_h . k) / sqrt(128)`` (mqa.py:39-58), keeps the top 512
(metadata.py:20-46, kernel.py:23-73) and appends its incomplete tail ``[4*((t+1)//4), t]`` (kernel.py:266-320). With
512 blocks or fewer every token is visible (dense). Draft steps 2-3 do not run the indexer: they reuse the selection of
the draft-extend row ``t`` plus their own chain positions (QSAMTPSharedSparseIndices, qwen_sparse_attn_backend.py:
127-201, 1407-1530; ``index_share_for_mtp_iteration`` defaults to True for Qwen4-Exp, configs/qwen4_exp.py:33).

**Not mirrored** (numerically small; the replica check measures what is left): the exact BF16 rounding points inside
SGLang's fused kernels (hc mix/combine, fused q/k norm + RoPE, Marlin, silu-and-mul, moe sum), the first prefill
chunk's fresh-BF16 K/V (later chunks and every decode read the FP8 cache), atomics order. Online MXFP8
(``SGLANG_SM120_ONLINE_MXFP8``, off in Franzen's and D''s cell 12) would quantize the draft's projections and hc mix
weights: an arm that turns it on must repeat the replica check.

**The replica check** (section 5.2; ``check``): scripts/mtp_probe_dump.py dumps (``ARC3_HC_DUMP_KEEP=all``, BF16) the
held-out probe requests of ``runs/fidelity-base`` over their prompt plus the greedy output that run recorded. For each,
:func:`simulate_greedy` replays SGLang's greedy chain along that output with the replica and counts verify steps and
accepted drafts, giving SGLang's ``spec_accept_length = completion_tokens / verify_ct`` (tokenizer_manager.py:2854-2857;
the prefill's token included); the result is compared per request with the same run's ``spec_*`` meta info (and
``spec_correct_drafts_histogram`` when recorded).
Variants: ``full`` (QSA emulated over the whole context), ``dense`` (all context, no QSA), ``cutN`` (only the last N
prompt rows, dense: the regime the trainer's windows see). Go: on the gate variant, |mean(replica - SGLang)| <= 0.05
and Pearson r >= 0.9 over at least 8 requests (:data:`GATE`).
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - the tests skip without torch
    raise ImportError("scripts/mtp_replica.py needs torch (the trainer's and the replica check's environment)") from exc


def _sibling(name: str):
    """A module from this file's folder, loaded by path (``python -I`` does not put the folder on sys.path)."""
    key = f"arc3_mtp_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = Path(__file__).with_name(f"{name}.py")
    spec = importlib.util.spec_from_file_location(key, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"{path} not found (keep it next to {Path(__file__).name})")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module  # dataclasses look their module up while the class is built
    spec.loader.exec_module(module)
    return module


hd = _sibling("sglang_hc_dump_patch")  # the dump format (numpy only at import)

GATE = {"max_abs_mean_diff": 0.05, "min_pearson": 0.9, "min_requests": 8}
L0 = "mtp.layers.0."
ATTN, MLP, MIXER = L0 + "attn_hyper_connection.", L0 + "mlp_hyper_connection.", "mtp.hyper_connection_mixer."
INDEXER_NAMES = (L0 + "self_attn.indexer.index_qk_proj.weight", L0 + "self_attn.indexer.k_layernorm.weight",
                 L0 + "self_attn.indexer.q_layernorm.weight")
EXPERT_PREFIX = L0 + "mlp.experts."
PROJS = ("gate_proj", "up_proj", "down_proj")
EMBED_NAME, LM_HEAD_NAME = "model.language_model.embed_tokens.weight", "lm_head.weight"
TARGET_MIXER = "model.language_model.hyper_connection_mixer."


# ------------------------------------------------------------------------------------------------------ configuration


@dataclasses.dataclass(frozen=True)
class MTPConfig:
    hidden: int = 2560
    hc: int = 4
    lowrank: int = 320
    heads: int = 24
    kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1e7
    experts: int = 512
    topk: int = 10
    moe_inter: int = 640
    shared_inter: int = 640
    idx_heads: int = 4
    idx_head_dim: int = 128
    idx_budget: int = 2048
    compress: int = 4
    eps: float = 1e-6
    vocab: int = 248320
    norm_topk_prob: bool = True
    group_size: int = 32

    @property
    def width(self) -> int:
        return self.hc * self.hidden

    @property
    def block_topk(self) -> int:
        return self.idx_budget // self.compress

    @classmethod
    def from_config_json(cls, cfg: dict) -> MTPConfig:
        """From a checkpoint's config.json (the draft's or the target's; ``text_config`` when present)."""
        t = cfg.get("text_config") or cfg
        rp = t.get("rope_parameters") or t.get("rope_scaling") or {}
        head_dim = int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"])
        prf = float(rp.get("partial_rotary_factor", t.get("partial_rotary_factor", 1.0)))
        groups = ((cfg.get("quantization_config") or {}).get("config_groups") or {})
        weights = (groups.get("mtp_routed_experts") or {}).get("weights") or {}
        return cls(hidden=int(t["hidden_size"]), hc=int(t.get("hc_count", 4)), lowrank=int(t.get("hc_lowrank", 320)),
                   heads=int(t["num_attention_heads"]), kv_heads=int(t["num_key_value_heads"]), head_dim=head_dim,
                   rotary_dim=int(head_dim * prf), rope_theta=float(rp.get("rope_theta", t.get("rope_theta", 1e7))),
                   experts=int(t["num_experts"]), topk=int(t["num_experts_per_tok"]),
                   moe_inter=int(t["moe_intermediate_size"]),
                   shared_inter=int(t.get("shared_expert_intermediate_size") or t["moe_intermediate_size"]),
                   idx_heads=int(t.get("indexer_n_heads", 4)), idx_head_dim=int(t.get("indexer_head_dim", 128)),
                   idx_budget=int(t.get("indexer_budget", 2048)), compress=int(t.get("indexer_compress_ratio", 4)),
                   eps=float(t.get("rms_norm_eps", 1e-6)), vocab=int(t["vocab_size"]),
                   norm_topk_prob=bool(t.get("norm_topk_prob", True)), group_size=int(weights.get("group_size", 32)))

    @classmethod
    def tiny(cls, **kw) -> MTPConfig:
        """A configuration small enough for CPU tests (same structure)."""
        base = {"hidden": 32, "hc": 4, "lowrank": 8, "heads": 4, "kv_heads": 2, "head_dim": 16, "rotary_dim": 4,
                "rope_theta": 1e4, "experts": 8, "topk": 2, "moe_inter": 32, "shared_inter": 32, "idx_heads": 2,
                "idx_head_dim": 8, "idx_budget": 16, "compress": 4, "vocab": 96, "group_size": 32}
        base.update(kw)
        return cls(**base)

    def text_config(self) -> dict:
        """The text_config fields this replica reads (for writing test checkpoints)."""
        return {"hidden_size": self.hidden, "hc_count": self.hc, "hc_lowrank": self.lowrank,
                "num_attention_heads": self.heads, "num_key_value_heads": self.kv_heads, "head_dim": self.head_dim,
                "partial_rotary_factor": self.rotary_dim / self.head_dim,
                "rope_parameters": {"rope_theta": self.rope_theta, "partial_rotary_factor": self.rotary_dim / self.head_dim,
                                    "mrope_interleaved": True, "rope_type": "default"},
                "num_experts": self.experts, "num_experts_per_tok": self.topk, "moe_intermediate_size": self.moe_inter,
                "shared_expert_intermediate_size": self.shared_inter, "indexer_n_heads": self.idx_heads,
                "indexer_kv_heads": 1, "indexer_head_dim": self.idx_head_dim, "indexer_budget": self.idx_budget,
                "indexer_compress_ratio": self.compress, "rms_norm_eps": self.eps, "vocab_size": self.vocab,
                "norm_topk_prob": self.norm_topk_prob}


def dense_shapes(c: MTPConfig) -> dict[str, tuple]:
    """The 29 dense ``mtp.*`` tensors (names and order of albucino's mtp-dense.safetensors, minus lm_head/embed)."""
    W, H, R = c.width, c.hidden, c.lowrank
    return {
        "mtp.fc_embedding.weight": (H, H), "mtp.fc_hidden.weight": (H, H),
        MIXER + "hc_norm.weight": (W,), MIXER + "input_mix_weight_down.weight": (R, W),
        MIXER + "input_mix_weight_up.weight": (W, R),
        ATTN + "block_inject_weight.weight": (c.hc, W), ATTN + "hc_norm.weight": (W,),
        ATTN + "input_mix_weight_down.weight": (R, W), ATTN + "input_mix_weight_up.weight": (W, R),
        L0 + "mlp.gate.weight": (c.experts, H), L0 + "mlp.shared_expert.down_proj.weight": (H, c.shared_inter),
        L0 + "mlp.shared_expert.gate_proj.weight": (c.shared_inter, H),
        L0 + "mlp.shared_expert.up_proj.weight": (c.shared_inter, H), L0 + "mlp.shared_expert_gate.weight": (1, H),
        MLP + "block_inject_weight.weight": (c.hc, W), MLP + "hc_norm.weight": (W,),
        MLP + "input_mix_weight_down.weight": (R, W), MLP + "input_mix_weight_up.weight": (W, R),
        INDEXER_NAMES[0]: ((c.idx_heads + 1) * c.idx_head_dim, H), INDEXER_NAMES[1]: (c.idx_head_dim,),
        INDEXER_NAMES[2]: (c.idx_head_dim,), L0 + "self_attn.k_norm.weight": (c.head_dim,),
        L0 + "self_attn.k_proj.weight": (c.kv_heads * c.head_dim, H),
        L0 + "self_attn.o_proj.weight": (H, c.heads * c.head_dim), L0 + "self_attn.q_norm.weight": (c.head_dim,),
        L0 + "self_attn.q_proj.weight": (2 * c.heads * c.head_dim, H),
        L0 + "self_attn.v_proj.weight": (c.kv_heads * c.head_dim, H),
        "mtp.pre_fc_norm_embedding.weight": (H,), "mtp.pre_fc_norm_hidden.weight": (W,),
    }


DENSE_NAMES = tuple(dense_shapes(MTPConfig()))
TRAINABLE_NAMES = tuple(n for n in DENSE_NAMES if n not in INDEXER_NAMES)  # plan 3.1: everything but the indexer


def expert_names(e: int) -> dict[str, str]:
    return {f"{p}.{part}": f"{EXPERT_PREFIX}{e}.{p}.{part}" for p in PROJS
            for part in ("weight_packed", "weight_scale", "weight_shape")}


# ------------------------------------------------------------------------------------------------------- numerics


def gemma_rmsnorm(x, weight, eps: float, group: int | None = None):
    """Gemma RMSNorm in FP32, ``(1 + w) * x / rms(x)``, over the last dim or groups of it; result in x's dtype."""
    xf = x.float()
    if group is not None and group != xf.shape[-1]:
        xg = xf.unflatten(-1, (-1, group))
        xf = (xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)).flatten(-2)
    else:
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


def rope(x, positions, rotary_dim: int, theta: float):
    """NeoX RoPE on the first ``rotary_dim`` dims of x [n, heads, d] at 1-D positions [n] (base.py:153-182)."""
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=x.device) / rotary_dim))
    f = positions.to(torch.float32)[:, None] * inv[None, :]
    cos, sin = f.cos()[:, None, :], f.sin()[:, None, :]
    xr = x[..., :rotary_dim].float()
    x1, x2 = xr[..., :rotary_dim // 2], xr[..., rotary_dim // 2:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(x.dtype)
    return torch.cat([out, x[..., rotary_dim:]], dim=-1)


class _FP8STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        return grad


def fp8_round(x):
    """The FP8 e4m3 KV-cache round trip (a plain cast, no scale), straight-through for gradients."""
    return _FP8STE.apply(x)


def unpack_int4(packed, shape):
    """compressed-tensors pack-quantized INT4 -> int8 values in [-8, 7] (8 per int32, low nibble first, + 8)."""
    out, inp = int(shape[0]), int(shape[1])
    shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
    nib = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & 0xF
    return (nib.reshape(out, -1)[:, :inp] - 8).to(torch.int8)


def dequant_int4(packed, scale, shape, group: int = 32, dtype=torch.bfloat16):
    """``q * scale`` rounded to BF16, as the Marlin W4A16 kernel multiplies (plan 3.7)."""
    q = unpack_int4(packed, shape)
    out, inp = q.shape
    w = q.reshape(out, inp // group, group).to(torch.bfloat16) * scale.to(torch.bfloat16).unsqueeze(-1)
    return w.reshape(out, inp).to(dtype)


def pack_int4_rtn(w, group: int = 32):
    """albucino's RTN recipe (plan section 4): BF16 [out, in] -> (packed int32 [out, in/8], scale BF16 [out, in/g]).
    A group of zeros gets scale 1 (never 0, which would divide by zero)."""
    out = w.shape[0]
    g = w.to(torch.bfloat16).reshape(out, -1, group)
    scale = (g.float().abs().amax(-1) / 7.5).to(torch.bfloat16)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / scale.unsqueeze(-1)), -8, 7)
    vals = (q.to(torch.int64) + 8).reshape(out, -1, 8) << (4 * torch.arange(8, dtype=torch.int64))
    packed = vals.sum(-1)
    packed = torch.where(packed >= 2 ** 31, packed - 2 ** 32, packed).to(torch.int32)
    return packed, scale


# ------------------------------------------------------------------------------------------- safetensors files


ST_TORCH = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I32": torch.int32, "I64": torch.int64,
            "U8": torch.uint8, "F8_E4M3": torch.float8_e4m3fn, "I8": torch.int8}
ST_NP = {"BF16": np.int16, "F16": np.float16, "F32": np.float32, "I32": np.int32, "I64": np.int64, "U8": np.uint8,
         "F8_E4M3": np.uint8, "I8": np.int8}


def read_header(path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
    return header, 8 + size


class SafeReader:
    """Tensors by name from a checkpoint directory (via model.safetensors.index.json) or single files; memory-mapped,
    no dependency on the safetensors package."""

    def __init__(self, files):
        self.where: dict[str, tuple[Path, dict, int]] = {}
        for path in files:
            header, base = read_header(path)
            header.pop("__metadata__", None)
            for name, info in header.items():
                self.where[name] = (Path(path), info, base)
        self._maps: dict[Path, np.memmap] = {}

    @classmethod
    def from_dir(cls, directory) -> SafeReader:
        d = Path(directory)
        index = d / "model.safetensors.index.json"
        if index.is_file():
            names = sorted(set(json.loads(index.read_text())["weight_map"].values()))
            return cls([d / n for n in names])
        return cls(sorted(d.glob("*.safetensors")))

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def info(self, name: str) -> dict:
        return self.where[name][1]

    def get(self, name: str, device="cpu"):
        path, info, base = self.where[name]
        if path not in self._maps:
            self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        begin, end = info["data_offsets"]
        raw = np.asarray(self._maps[path][base + begin:base + end])
        arr = raw.view(ST_NP[info["dtype"]]).reshape(info["shape"])
        t = torch.from_numpy(arr.copy())
        if info["dtype"] == "BF16":
            t = t.view(torch.bfloat16)
        elif info["dtype"] == "F8_E4M3":
            t = t.view(torch.float8_e4m3fn)
        return t.to(device)


def save_safetensors(path, tensors: dict, metadata: dict | None = None) -> int:
    """torch tensors (BF16/F32/I32/I64) -> a plain safetensors file, in the given order; returns its size."""
    items = []
    for name, t in tensors.items():
        t = t.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            items.append((name, "BF16", t.view(torch.int16).numpy().view(np.uint16)))
        elif t.dtype == torch.float32:
            items.append((name, "F32", t.numpy()))
        elif t.dtype in (torch.int32, torch.int64):
            items.append((name, "I32" if t.dtype == torch.int32 else "I64", t.numpy()))
        else:
            raise TypeError(f"{name}: dtype {t.dtype} not supported")
    return hd.write_safetensors(Path(path), hd.encode_safetensors(items, {k: str(v) for k, v in (metadata or {}).items()}))


def load_token_map(path, vocab: int | None = None):
    """The FR-Spec hot-token ids (a pickled list of ints, spec_utils.py:653-681) as int64, ascending, unique."""
    ids = torch.load(path, weights_only=True)
    t = torch.as_tensor(ids, dtype=torch.int64)
    if t.ndim != 1 or (vocab is not None and int(t.max()) >= vocab) or int(t.min()) < 0:
        raise ValueError(f"{path}: not a 1-D id list within the vocabulary")
    return t


@dataclasses.dataclass
class DraftWeights:
    cfg: MTPConfig
    dense: dict          # name -> BF16 tensor (the 29 mtp.* tensors)
    experts: tuple       # (w_gate [E,I,H], w_up [E,I,H], w_down [E,H,I]) dequantized
    embed: object = None    # [V, H] or None
    lm_head: object = None  # [V, H] or None


def load_draft_dir(draft_dir, *, device="cpu", dtype=torch.bfloat16, embed: bool = True, lm_head: bool = True,
                   experts: bool = True) -> DraftWeights:
    """albucino's draft directory (config.json, index, mtp-dense, mtp-routed-experts-int4) -> DraftWeights."""
    d = Path(draft_dir)
    cfg = MTPConfig.from_config_json(json.loads((d / "config.json").read_text()))
    reader = SafeReader.from_dir(d)
    shapes = dense_shapes(cfg)
    dense = {}
    for name, shape in shapes.items():
        t = reader.get(name, device)
        if tuple(t.shape) != shape:
            raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {shape}")
        dense[name] = t.to(torch.bfloat16)
    banks = None
    if experts:
        inter, H = cfg.moe_inter, cfg.hidden
        banks = (torch.empty(cfg.experts, inter, H, dtype=dtype, device=device),
                 torch.empty(cfg.experts, inter, H, dtype=dtype, device=device),
                 torch.empty(cfg.experts, H, inter, dtype=dtype, device=device))
        for e in range(cfg.experts):
            names = expert_names(e)
            for bank, proj in zip(banks, PROJS, strict=True):
                shape = reader.get(names[f"{proj}.weight_shape"]).tolist()
                bank[e] = dequant_int4(reader.get(names[f"{proj}.weight_packed"], device),
                                       reader.get(names[f"{proj}.weight_scale"], device), shape, cfg.group_size, dtype)
    emb = reader.get(EMBED_NAME, device).to(dtype) if embed and EMBED_NAME in reader else None
    head = reader.get(LM_HEAD_NAME, device).to(dtype) if lm_head and LM_HEAD_NAME in reader else None
    return DraftWeights(cfg, dense, banks, emb, head)


def load_dense_file(path) -> dict:
    """A trained-dense.safetensors (or any file of mtp.* tensors) -> name -> tensor."""
    reader = SafeReader([Path(path)])
    return {name: reader.get(name) for name in reader.where}


def random_draft(cfg: MTPConfig, seed: int = 0, *, scale: float = 0.08, dtype=torch.float32) -> DraftWeights:
    """Random weights of the right shapes (tests): norms near 0 (Gemma (1+w)), projections ~N(0, scale)."""
    g = torch.Generator().manual_seed(seed)
    dense = {}
    for name, shape in dense_shapes(cfg).items():
        std = 0.05 if len(shape) == 1 else scale
        dense[name] = (torch.randn(shape, generator=g) * std).to(dtype)
    inter, H = cfg.moe_inter, cfg.hidden
    banks = tuple((torch.randn(s, generator=g) * scale).to(dtype)
                  for s in ((cfg.experts, inter, H), (cfg.experts, inter, H), (cfg.experts, H, inter)))
    emb = (torch.randn(cfg.vocab, H, generator=g) * 0.5).to(dtype)
    head = (torch.randn(cfg.vocab, H, generator=g) * scale).to(dtype)
    return DraftWeights(cfg, dense, banks, emb, head)


# ----------------------------------------------------------------------------------------------------- the model


class MTPReplica(torch.nn.Module):
    """The MTP block with its dense weights as FP32 parameters (trainable ones require grad), the experts, the
    embedding and the hot lm_head rows as frozen buffers. Computation runs in ``compute_dtype`` (BF16 as served; FP32
    in tests), with the rounding points of section 1 of the module docstring."""

    def __init__(self, weights: DraftWeights, *, hot_ids=None, trainable=TRAINABLE_NAMES, compute_dtype=torch.bfloat16,
                 fp8_kv: bool = True, device=None):
        super().__init__()
        cfg = weights.cfg
        self.cfg, self.dt, self.fp8_kv = cfg, compute_dtype, fp8_kv
        dev = device if device is not None else next(iter(weights.dense.values())).device
        self.names = tuple(dense_shapes(cfg))
        self.trainable = tuple(n for n in self.names if n in set(trainable))
        self.p = torch.nn.ParameterDict({_key(n): torch.nn.Parameter(weights.dense[n].detach().to(dev, torch.float32).clone(),
                                                                       requires_grad=n in self.trainable)
                                         for n in self.names})
        gate, up, down = weights.experts
        self.register_buffer("w_gate", gate.to(dev, compute_dtype), persistent=False)
        self.register_buffer("w_up", up.to(dev, compute_dtype), persistent=False)
        self.register_buffer("w_down", down.to(dev, compute_dtype), persistent=False)
        if weights.embed is None or weights.lm_head is None:
            raise ValueError("the replica needs the target's embed_tokens and lm_head")
        self.register_buffer("embed_tokens", weights.embed.to(dev, compute_dtype), persistent=False)
        hot = None if hot_ids is None else torch.as_tensor(hot_ids, dtype=torch.int64, device=dev)
        self.register_buffer("hot_ids", hot if hot is not None else torch.arange(cfg.vocab, device=dev),
                             persistent=False)
        self.register_buffer("lm_head_hot", weights.lm_head.to(dev)[self.hot_ids].to(compute_dtype), persistent=False)

    # ---- weights
    def w(self, name: str):
        return self.p[_key(name)].to(self.dt)

    def lin(self, x, name: str):
        return F.linear(x, self.w(name))

    def dense_state(self, dtype=torch.bfloat16) -> dict:
        return {n: self.p[_key(n)].detach().to(dtype) for n in self.names}

    def load_dense(self, tensors: dict, strict: bool = True) -> list[str]:
        loaded = []
        for name, t in tensors.items():
            if _key(name) not in self.p:
                if strict and name.startswith("mtp."):
                    raise KeyError(f"{name} is not a dense MTP tensor")
                continue
            if tuple(t.shape) != tuple(self.p[_key(name)].shape):
                raise ValueError(f"{name}: shape {tuple(t.shape)}, expected {tuple(self.p[_key(name)].shape)}")
            with torch.no_grad():
                self.p[_key(name)].copy_(t.to(self.p[_key(name)].device, torch.float32))
            loaded.append(name)
        return loaded

    # ---- pieces of the block
    def embed(self, ids):
        return F.embedding(ids.clamp(0, self.cfg.vocab - 1), self.embed_tokens)

    def fuse(self, emb, H):
        c = self.cfg
        u = self.lin(gemma_rmsnorm(emb, self.w("mtp.pre_fc_norm_embedding.weight"), c.eps), "mtp.fc_embedding.weight")
        hn = gemma_rmsnorm(H.to(self.dt), self.w("mtp.pre_fc_norm_hidden.weight"), c.eps)
        enc = self.lin(hn.view(-1, c.hc, c.hidden), "mtp.fc_hidden.weight")
        return (u.unsqueeze(-2) + enc).reshape(-1, c.width)

    def hc_mix(self, prefix: str, X):
        c = self.cfg
        Xn = gemma_rmsnorm(X, self.w(prefix + "hc_norm.weight"), c.eps, group=c.hidden)
        t = F.silu(self.lin(Xn, prefix + "input_mix_weight_down.weight").float() / c.hc).to(self.dt)
        gate = torch.sigmoid(self.lin(t, prefix + "input_mix_weight_up.weight").float())
        out = (gate.view(-1, c.hc, c.hidden) * Xn.float().view(-1, c.hc, c.hidden)).mean(-2).to(self.dt)
        return out, Xn

    def hc_combine(self, prefix: str, out, X, Xn):
        c = self.cfg
        s = 2 * torch.sigmoid(self.lin(Xn, prefix + "block_inject_weight.weight").float() / c.hc)
        y = X.float().view(-1, c.hc, c.hidden) + out.float().unsqueeze(-2) * s.unsqueeze(-1)
        return y.reshape(-1, c.width).to(self.dt)

    def qkv(self, x, pos):
        c = self.cfg
        n = x.shape[0]
        qg = self.lin(x, L0 + "self_attn.q_proj.weight").view(n, c.heads, 2 * c.head_dim)
        q, gate = qg[..., :c.head_dim], qg[..., c.head_dim:]
        k = self.lin(x, L0 + "self_attn.k_proj.weight").view(n, c.kv_heads, c.head_dim)
        v = self.lin(x, L0 + "self_attn.v_proj.weight").view(n, c.kv_heads, c.head_dim)
        q = rope(gemma_rmsnorm(q, self.w(L0 + "self_attn.q_norm.weight"), c.eps), pos, c.rotary_dim, c.rope_theta)
        k = rope(gemma_rmsnorm(k, self.w(L0 + "self_attn.k_norm.weight"), c.eps), pos, c.rotary_dim, c.rope_theta)
        return q, gate, k, v

    def kv_round(self, x):
        return fp8_round(x) if self.fp8_kv else x

    def index_qk(self, x, pos):
        """QSA indexer: roped, normed query heads [n, ih, d] and raw token keys [n, d] (qsa_indexer.py:159-206)."""
        c = self.cfg
        qk = self.lin(x, INDEXER_NAMES[0])
        q = qk[:, :c.idx_heads * c.idx_head_dim].reshape(-1, c.idx_heads, c.idx_head_dim)
        q = rope(gemma_rmsnorm(q, self.w(INDEXER_NAMES[2]), c.eps), pos, c.rotary_dim, c.rope_theta)
        return q, qk[:, c.idx_heads * c.idx_head_dim:]

    def compress_keys(self, k_raw, block_pos):
        """[nb, ratio, d] raw keys of complete blocks -> normed keys roped at each block's first position."""
        c = self.cfg
        pooled = k_raw.float().mean(1).to(k_raw.dtype)
        normed = gemma_rmsnorm(pooled, self.w(INDEXER_NAMES[1]), c.eps)
        return rope(normed.unsqueeze(1), block_pos, c.rotary_dim, c.rope_theta).squeeze(1)

    def output(self, o, gate):
        o = (o.float() * torch.sigmoid(gate.float())).to(self.dt)
        return self.lin(o.reshape(o.shape[0], -1), L0 + "self_attn.o_proj.weight")

    def moe(self, x):
        c = self.cfg
        n = x.shape[0]
        probs = torch.softmax(self.lin(x, L0 + "mlp.gate.weight").float(), dim=-1)
        topw, topi = probs.topk(c.topk, dim=-1)
        if c.norm_topk_prob:
            topw = topw / topw.sum(-1, keepdim=True)
        flat = topi.reshape(-1)
        order = torch.argsort(flat, stable=True)  # (token, slot) pairs grouped by expert, experts ascending
        counts = torch.bincount(flat, minlength=c.experts).tolist()
        tokens = torch.arange(n, device=x.device).repeat_interleave(c.topk)[order]
        xs = x.index_select(0, tokens)
        ys, start = [], 0
        for e, count in enumerate(counts):
            if not count:
                continue
            xe = xs[start:start + count]
            a = (F.silu(F.linear(xe, self.w_gate[e]).float()) * F.linear(xe, self.w_up[e]).float()).to(self.dt)
            ys.append(F.linear(a, self.w_down[e]))
            start += count
        # back to (token, slot) order, weighted, summed over the slots in a fixed order (deterministic; Marlin writes
        # one weighted output per slot and moe_sum reduces them)
        y = torch.cat(ys).index_select(0, torch.argsort(order)).view(n, c.topk, c.hidden)
        routed = (y.float() * topw.unsqueeze(-1)).sum(1)
        sa = (F.silu(self.lin(x, L0 + "mlp.shared_expert.gate_proj.weight").float())
              * self.lin(x, L0 + "mlp.shared_expert.up_proj.weight").float()).to(self.dt)
        so = self.lin(sa, L0 + "mlp.shared_expert.down_proj.weight")
        sg = torch.sigmoid(self.lin(x, L0 + "mlp.shared_expert_gate.weight").float())
        return (routed.to(self.dt).float() + sg * so.float()).to(self.dt)

    def final_mix(self, G):
        return self.hc_mix(MIXER, G)[0]

    def logits(self, m):
        """Draft logits over the hot rows [n, Vh] (FP32)."""
        return F.linear(m, self.lm_head_hot).float()

    def propose(self, logits):
        return self.hot_ids[logits.argmax(-1)]

    # ---- one step for a set of rows; ``attend(q, gate, k8, v8)`` returns the attention output [n, heads, d]
    def _step(self, X, pos, attend):
        a_in, Xn = self.hc_mix(ATTN, X)
        q, gate, k, v = self.qkv(a_in, pos)
        k8, v8 = self.kv_round(k), self.kv_round(v)
        o = attend(q, gate, k8, v8)
        X2 = self.hc_combine(ATTN, self.output(o, gate), X, Xn)
        m_in, X2n = self.hc_mix(MLP, X2)
        G = self.hc_combine(MLP, self.moe(m_in), X2, X2n)
        return self.final_mix(G), G, k8, v8

    # ---- the chained training-time test over packed windows (plan 3.3)
    def chain(self, H, emb1, pos, bounds, chain_ids, steps: int = 3, checkpoint: bool = False):
        """Rows of one or more windows packed along dim 0 (``bounds``: [(start, end)] of each window, rows in position
        order). Step 1 runs every row with ``(H_j, emb1_j)`` at ``pos_j`` under causal attention within its window
        (the draft KV of a draft-extend). Step k >= 2 runs for every row as an origin t: input ``(G^{k-1}_t,
        e(chain_ids[t, k-2]))`` at ``pos_t + k - 1``, attending to the window's step-1 keys at rows <= t plus its own
        chain keys of steps 2..k. Returns the final-mixer outputs [n, hidden] of each step."""
        outs = []
        X = self.fuse(emb1, H)
        run = _maybe_checkpoint(checkpoint)
        m, G, K1, V1 = run(self._step, X, pos, lambda q, g, k, v: self.attend_windows(q, k, v, bounds))
        outs.append(m)
        extras: list = []
        for k in range(2, steps + 1):
            X = self.fuse(self.embed(chain_ids[:, k - 2]), G)
            prior = list(extras)
            m, G, kk, vv = run(self._step, X, pos + (k - 1),
                               lambda q, g, k_, v_, prior=prior: self.attend_windows(q, K1, V1, bounds,
                                                                                    [*prior, (k_, v_)]))
            extras.append((kk, vv))
            outs.append(m)
        return outs

    def attend_windows(self, q, K, V, bounds, extras=(), q_block: int = 1024):
        """Causal attention within each window over K/V rows, plus per-row extra keys (an origin's own chain KV)."""
        c = self.cfg
        g, scale = c.heads // c.kv_heads, c.head_dim ** -0.5
        pieces = []
        for a, b in bounds:
            Ks, Vs = K[a:b].float(), V[a:b].float()
            for qa in range(a, b, q_block):
                qb = min(b, qa + q_block)
                n, m = qb - qa, qb - a
                qs = q[qa:qb].float().view(n, c.kv_heads, g, c.head_dim)
                s = torch.einsum("ikgd,jkd->ikgj", qs, Ks[:m]) * scale
                causal = torch.arange(m, device=q.device)[None, :] <= torch.arange(qa - a, qb - a, device=q.device)[:, None]
                parts = [s.masked_fill(~causal[:, None, None, :], float("-inf"))]
                for ke, _ in extras:
                    parts.append((qs * ke[qa:qb].float().unsqueeze(2)).sum(-1, keepdim=True) * scale)
                p = torch.softmax(torch.cat(parts, dim=-1), dim=-1)
                o = torch.einsum("ikgj,jkd->ikgd", p[..., :m], Vs[:m])
                for j, (_, ve) in enumerate(extras):
                    o = o + p[..., m + j:m + j + 1] * ve[qa:qb].float().unsqueeze(2)
                pieces.append(o.reshape(n, c.heads, c.head_dim))
        return torch.cat(pieces).to(self.dt)

    def attend_sets(self, q, keys: list, values: list):
        """One query row per key set: q [n, heads, d]; keys[i]/values[i] [S_i, kv_heads, d]."""
        c = self.cfg
        g, scale = c.heads // c.kv_heads, c.head_dim ** -0.5
        out = []
        for i in range(q.shape[0]):
            qs = q[i].float().view(c.kv_heads, g, c.head_dim)
            s = torch.einsum("kgd,jkd->kgj", qs, keys[i].float()) * scale
            out.append(torch.einsum("kgj,jkd->kgd", torch.softmax(s, -1), values[i].float()).reshape(c.heads, c.head_dim))
        return torch.stack(out).to(self.dt)


def _key(name: str) -> str:
    return name.replace(".", "__")


def _maybe_checkpoint(enabled: bool):
    if not enabled:
        return lambda fn, *args: fn(*args)
    from torch.utils.checkpoint import checkpoint

    return lambda fn, *args: checkpoint(fn, *args, use_reentrant=False)


# ---------------------------------------------------------------------------------- serving-style draft (no grad)


def extend_embeddings(model: MTPReplica, ids, *, mode: str, mm_rows=None, img=None, chunk_last=None):
    """The draft-extend embedding of every row (rows = positions 0..L-1 of ``ids``, which has L+1 entries so that
    ``ids[j+1]`` exists). ``mode`` "shifted": ``e(x_{j+1})`` (text-only requests, post-verify extends);
    "prefill": rows in ``mm_rows`` (bool [L]: prefilled in a chunk that had mm_input_embeds) take the target's input
    embedding ``e(x_j)``, or ``img[j]`` (a dict position -> [hidden] vision features) at image rows, except rows in
    ``chunk_last`` (bool [L]) which take ``e(x_{j+1})`` (qwen4_exp_mtp.py:137-159)."""
    L = ids.shape[0] - 1
    shifted = model.embed(ids[1:])
    if mode == "shifted" or mm_rows is None:
        return shifted
    if mode != "prefill":
        raise ValueError(f"unknown embedding mode {mode!r}")
    use = mm_rows.clone()
    if chunk_last is not None:
        use &= ~chunk_last
    out = torch.where(use[:, None], model.embed(ids[:L]), shifted)
    for pos, vec in (img or {}).items():
        if 0 <= pos < L and use[pos]:
            out[pos] = vec.to(out.dtype)
    return out


def qsa_select_tokens(q_idx, ck, t: int, compress: int, block_topk: int, head_dim: int):
    """QSA's token selection for a query at position ``t``: the top ``block_topk`` of the ``(t+1)//compress`` complete
    blocks by ``sum_h relu(q_h . k_b) / sqrt(head_dim)`` (all of them when they are no more), expanded to their tokens,
    plus the incomplete tail ``[compress*((t+1)//compress), t]`` (mqa.py:39-58, kernel.py:23-73 and 266-320).
    ``q_idx`` [ih, d] (normed and roped), ``ck`` [nb, d]; returns logical positions, blocks ascending, then the tail."""
    dev = ck.device
    nvis = (t + 1) // compress
    tail = torch.arange(nvis * compress, t + 1, device=dev)
    if nvis <= block_topk:
        blocks = torch.arange(nvis, device=dev)
    else:
        scores = torch.relu(torch.einsum("hd,bd->bh", q_idx.float(), ck[:nvis].float())).sum(-1) / math.sqrt(head_dim)
        blocks = torch.sort(scores.topk(block_topk).indices).values
    tokens = (blocks[:, None] * compress + torch.arange(compress, device=dev)).reshape(-1)
    return torch.cat([tokens, tail])


class DraftKV:
    """The draft-extend state of one request: per row its FP8 K/V and raw index key, and the compressed index keys
    (computed once from rows 0..L-1, as the successive draft-extends of a serving run leave them)."""

    def __init__(self, model: MTPReplica, H, emb, pos, *, chunk: int = 4096):
        self.model, self.H, self.emb, self.pos = model, H, emb, pos
        c = model.cfg
        K, V, KI = [], [], []
        with torch.no_grad():
            for a in range(0, H.shape[0], chunk):
                X = model.fuse(emb[a:a + chunk], H[a:a + chunk])
                a_in, _ = model.hc_mix(ATTN, X)
                _, _, k, v = model.qkv(a_in, pos[a:a + chunk])
                K.append(model.kv_round(k))
                V.append(model.kv_round(v))
                KI.append(model.index_qk(a_in, pos[a:a + chunk])[1])
            self.K, self.V, self.k_idx = torch.cat(K), torch.cat(V), torch.cat(KI)
            nb = H.shape[0] // c.compress
            self.ck = (model.compress_keys(self.k_idx[:nb * c.compress].view(nb, c.compress, -1),
                                           pos[:nb * c.compress:c.compress]) if nb else self.k_idx[:0])

    def select(self, t: int, *, mode: str = "full", start: int = 0):
        """Logical positions row ``t`` attends to. full: QSA (top-k complete blocks by the indexer + the incomplete
        tail); dense: every row <= t; cut: rows ``start``..t."""
        c = self.model.cfg
        if mode in ("dense", "cut"):
            return torch.arange(start if mode == "cut" else 0, t + 1, device=self.K.device)
        q = None
        if (t + 1) // c.compress > c.block_topk:
            with torch.no_grad():
                X = self.model.fuse(self.emb[t:t + 1], self.H[t:t + 1])
                a_in, _ = self.model.hc_mix(ATTN, X)
                q = self.model.index_qk(a_in, self.pos[t:t + 1])[0][0]
        return qsa_select_tokens(q, self.ck, t, c.compress, c.block_topk, c.idx_head_dim)

    @torch.no_grad()
    def chain(self, t: int, *, steps: int = 3, mode: str = "full", start: int = 0, inputs=None):
        """The draft chain at last accepted row ``t`` (its embedding row already holds e(x_{t+1})). Returns the
        proposals and the logits of each step. ``inputs`` (k-1 ids) teacher-forces the step inputs instead of the
        proposals (tests)."""
        model = self.model
        sel = self.select(t, mode=mode, start=start)
        Ks, Vs = self.K[sel], self.V[sel]
        X = model.fuse(self.emb[t:t + 1], self.H[t:t + 1])
        m, G, _, _ = model._step(X, self.pos[t:t + 1], lambda q, g, k, v: model.attend_sets(q, [Ks], [Vs]))
        logits = [model.logits(m)]
        props = [model.propose(logits[-1])]
        chain_k, chain_v = [], []
        for k in range(2, steps + 1):
            token = props[-1] if inputs is None else torch.as_tensor(inputs[k - 2], device=props[-1].device).view(1)
            X = model.fuse(model.embed(token), G)
            pos = self.pos[t:t + 1] + (k - 1)

            def attend(q, g, k_, v_):
                return model.attend_sets(q, [torch.cat([Ks, *chain_k, k_])], [torch.cat([Vs, *chain_v, v_])])

            m, G, kk, vv = model._step(X, pos, attend)
            chain_k.append(kk)
            chain_v.append(vv)
            logits.append(model.logits(m))
            props.append(model.propose(logits[-1]))
        return [int(p) for p in props], logits


def simulate_greedy(kv: DraftKV, prompt_len: int, out_ids, *, mode: str = "full", start: int = 0, steps: int = 3):
    """SGLang's greedy speculative loop along a recorded greedy output ``out_ids`` (out_ids[0] = the prefill's token
    at position ``prompt_len``): each verify compares the chain's proposals with the next recorded tokens, accepts the
    leading matches and adds the bonus. Targets beyond the recorded output are unknown, so a last verify counts only
    the matches it can see. Returns verify_ct, correct_drafts, the histogram, ``accept_length`` as SGLang reports it
    (completion tokens / verify_ct) and ``accept_length_strict`` ((verify_ct + correct_drafts) / verify_ct)."""
    n = len(out_ids)
    produced, t = 1, prompt_len - 1
    verify, correct, hist, proposals, origins = 0, 0, [0] * (steps + 1), [], []
    while produced < n:
        props, _ = kv.chain(t, steps=steps, mode=mode, start=start)
        a = 0
        for k in range(steps):
            if produced + k >= n or props[k] != int(out_ids[produced + k]):
                break
            a += 1
        verify += 1
        correct += a
        hist[a] += 1
        proposals.append(props)
        origins.append(t)
        produced += a + 1
        t += a + 1
    return {"verify_ct": verify, "correct_drafts": correct, "histogram": hist,
            "accept_length": n / verify if verify else None,
            "accept_length_strict": (verify + correct) / verify if verify else None, "proposals": proposals,
            "origins": origins}


# ------------------------------------------------------------------------------------------- the replica check


def rows_to_tensor(codes, scale, device, dtype=torch.bfloat16, chunk: int = 8192):
    """Dump rows (FP8 codes with BF16 scales, or BF16 bits) -> a tensor on ``device``."""
    out = torch.empty(codes.shape, dtype=dtype, device=device)
    for a in range(0, codes.shape[0], chunk):
        c = torch.from_numpy(np.ascontiguousarray(codes[a:a + chunk]))
        if scale is None:
            out[a:a + chunk] = c.view(torch.int16).view(torch.bfloat16).to(device, dtype)
            continue
        s = torch.from_numpy(np.ascontiguousarray(scale[a:a + chunk]).view(np.int16)).view(torch.bfloat16).float()
        v = c.view(torch.float8_e4m3fn).to(device).float()
        s = s.to(device)
        groups = 1 if s.ndim == 1 else s.shape[1]  # one scale per row, or per equal column group
        v = (v.view(v.shape[0], groups, -1) * s.view(s.shape[0], groups, 1)).reshape(v.shape)
        out[a:a + chunk] = v.to(dtype)
    return out


def chunk_last_rows(L: int, prefill_start: int, prompt_len: int, chunk: int = 8192):
    """Rows that ended a prefill chunk when the prompt was served (the chunk grid from the cache hit)."""
    last = torch.zeros(L, dtype=torch.bool)
    for end in range(prefill_start + chunk, prompt_len, chunk):
        last[end - 1] = True
    if prompt_len >= 1:
        last[prompt_len - 1] = True
    return last


def probe_request_state(model: MTPReplica, dump_dir, record: dict, *, embed_mode: str = "prefill", device="cpu",
                        chunk: int = 8192) -> dict:
    """Load one probe dump (prompt + recorded output, KEEP=all) -> H, embeddings, positions and the output check."""
    req = hd.load_request(dump_dir, record["rid"])
    ids = np.asarray(req["token_ids"], np.int64)
    P, out = int(record["prompt_tokens"]), [int(x) for x in record["token_ids"]]
    if len(req["hc_pos"]) != len(ids) or not np.array_equal(req["hc_pos"], np.arange(len(ids))):
        raise ValueError(f"{record['rid']}: the dump does not hold every row (serve with ARC3_HC_DUMP_KEEP=all)")
    dumped = ids[P:P + len(out)].tolist()
    match = next((i for i, (a, b) in enumerate(zip(dumped, out)) if a != b), min(len(dumped), len(out)))
    L = len(ids)
    H = rows_to_tensor(req["hc"], req["hc_scale"], device, model.dt, chunk)
    ids_t = torch.as_tensor(np.append(ids, ids[-1]), device=device)
    ids_t = torch.where(ids_t >= model.cfg.vocab, torch.full_like(ids_t, model.cfg.vocab - 1), ids_t)
    mm = torch.zeros(L, dtype=torch.bool)
    for line in req["chunks"]:
        if line.get("mm_embeds"):
            mm[line["start"]:line["start"] + line["n"]] = True
    mm[P:] = False  # output rows were decoded: their draft KV came from post-verify extends
    img = {}
    if req["img_pos"] is not None:
        emb_rows = rows_to_tensor(req["img_embeds"], None, device, model.dt)
        img = {int(p): emb_rows[i] for i, p in enumerate(req["img_pos"].tolist())}
    last = chunk_last_rows(L, int(record.get("cached_tokens") or 0), P)
    emb = extend_embeddings(model, ids_t, mode=embed_mode, mm_rows=mm.to(device), img=img,
                            chunk_last=last.to(device))
    pos = torch.as_tensor(req["positions"], dtype=torch.int64, device=device)
    return {"H": H, "emb": emb, "pos": pos, "prompt_len": P, "out": out[:match], "match": match,
            "out_len": len(out), "rows": L}


def _pearson(a: list, b: list):
    if len(a) < 3 or statistics.pstdev(a) == 0 or statistics.pstdev(b) == 0:
        return None
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / len(a)
    return cov / (statistics.pstdev(a) * statistics.pstdev(b))


def step_rates(hist: list) -> list:
    """P(step k accepted | steps < k accepted) from a correct-drafts histogram [n_0, n_1, ..., n_K]."""
    rates, alive = [], sum(hist)
    for k in range(1, len(hist)):
        passed = sum(hist[k:])
        rates.append(passed / alive if alive else None)
        alive = passed
    return rates


def summarize(rows: list, variant: str) -> dict:
    ok = [r for r in rows if r["variants"].get(variant) and r["reference"].get("accept_length") is not None]
    rep = [r["variants"][variant]["accept_length"] for r in ok]
    ref = [r["reference"]["accept_length"] for r in ok]
    diffs = [a - b for a, b in zip(rep, ref)]
    hist = [sum(x) for x in zip(*(r["variants"][variant]["histogram"] for r in ok))] if ok else []
    ref_hist = [r["reference"].get("histogram") for r in ok]
    out = {"requests": len(ok), "replica_mean": statistics.fmean(rep) if rep else None,
           "sglang_mean": statistics.fmean(ref) if ref else None,
           "mean_diff": statistics.fmean(diffs) if diffs else None,
           "mae": statistics.fmean(abs(d) for d in diffs) if diffs else None, "pearson": _pearson(rep, ref),
           "replica_step_rates": step_rates(hist) if hist else None,
           "replica_correct_per_verify": (sum(r["variants"][variant]["correct_drafts"] for r in ok)
                                          / max(1, sum(r["variants"][variant]["verify_ct"] for r in ok))),
           "sglang_correct_per_verify": (sum(r["reference"]["correct_drafts"] for r in ok)
                                         / max(1, sum(r["reference"]["verify_ct"] for r in ok))) if ok else None}
    if ok and all(h for h in ref_hist):
        out["sglang_step_rates"] = step_rates([sum(x) for x in zip(*ref_hist)])
    return out


def gate_verdict(summary: dict, gate: dict = GATE) -> dict:
    reasons = []
    if summary["requests"] < gate["min_requests"]:
        reasons.append(f"only {summary['requests']} comparable requests (< {gate['min_requests']})")
    if summary["mean_diff"] is None or abs(summary["mean_diff"]) > gate["max_abs_mean_diff"]:
        reasons.append(f"mean difference {summary['mean_diff']} beyond +-{gate['max_abs_mean_diff']}")
    if summary["pearson"] is None or summary["pearson"] < gate["min_pearson"]:
        reasons.append(f"per-request correlation {summary['pearson']} below {gate['min_pearson']}")
    return {"go": not reasons, "reasons": reasons, "gate": gate}


def parse_variant(name: str) -> tuple[str, int | None]:
    if name in ("full", "dense"):
        return name, None
    if name.startswith("cut") and name[3:].isdigit():
        return "cut", int(name[3:])
    raise ValueError(f"unknown variant {name!r} (full, dense, cutN)")


def replica_check(model: MTPReplica, dump_dir, records: list, *, variants=("full", "dense", "cut2048", "cut256"),
                  embed_mode: str = "prefill", device="cpu", gate_variant: str = "full", gate: dict = GATE,
                  logger=print) -> dict:
    """Compare the replica's greedy chain with SGLang's per-request speculative counts (see the module docstring)."""
    rows = []
    for rec in records:
        t0 = time.time()
        state = probe_request_state(model, dump_dir, rec, embed_mode=embed_mode, device=device)
        kv = DraftKV(model, state["H"], state["emb"], state["pos"])
        row = {"id": rec.get("id"), "rid": rec["rid"], "game": rec.get("game"), "prompt_tokens": state["prompt_len"],
               "out_len": state["out_len"], "out_match": state["match"], "reference": dict(rec.get("spec") or {}),
               "variants": {}}
        ref = row["reference"]
        if "spec_accept_length" in ref:
            ref = {"accept_length": ref["spec_accept_length"], "verify_ct": ref.get("spec_verify_ct"),
                   "correct_drafts": ref.get("spec_num_correct_drafts"),
                   "histogram": ref.get("spec_correct_drafts_histogram")}
            row["reference"] = ref
        if state["match"] < state["out_len"]:
            row["reference"] = {"accept_length": None, "note": "the dumped output differs from the recorded one at "
                                                               f"token {state['match']}; the request is not compared"}
        for name in variants:
            mode, cut = parse_variant(name)
            start = max(0, state["prompt_len"] - cut) if cut is not None else 0
            sim = simulate_greedy(kv, state["prompt_len"], state["out"], mode=mode, start=start)
            row["variants"][name] = {k: v for k, v in sim.items() if k not in ("proposals", "origins")}
            # first proposal per chain origin (row t): variants whose paths diverge still compare at shared rows
            row["variants"][name]["first_proposals"] = {int(o): int(p[0]) for o, p in zip(sim["origins"], sim["proposals"])}
        names = list(row["variants"])
        if len(names) > 1:  # how often the variants propose the same first token from the same row
            base = row["variants"][names[0]]
            row["agree_with_" + names[0]] = {n: _agreement(base["first_proposals"], row["variants"][n]["first_proposals"])
                                            for n in names[1:]}
        row["seconds"] = round(time.time() - t0, 2)
        rows.append(row)
        logger(f"{row['id']}: P={row['prompt_tokens']} out={row['out_match']}/{row['out_len']} "
               + " ".join(f"{n}={row['variants'][n]['accept_length']:.3f}" for n in names if
                          row['variants'][n]['accept_length'] is not None)
               + f" sglang={row['reference'].get('accept_length')}")
    summaries = {name: summarize(rows, name) for name in variants}
    for n in list(variants)[1:]:
        key = "agree_with_" + variants[0]
        vals = [r[key][n] for r in rows if r.get(key, {}).get(n) is not None]
        summaries[n]["first_proposal_" + key] = statistics.fmean(vals) if vals else None
    verdict = gate_verdict(summaries[gate_variant], gate) if gate_variant in summaries else None
    return {"requests": rows, "summary": summaries, "gate_variant": gate_variant, "verdict": verdict,
            "embed_mode": embed_mode}


def _agreement(a: dict, b: dict):
    shared = a.keys() & b.keys()
    return sum(a[o] == b[o] for o in shared) / len(shared) if shared else None


def load_probe_records(probe_jsonl, reference_json=None, pass_name: str = "seq") -> list:
    """scripts/mtp_probe_dump.py's probe-dump.jsonl (rid, id, prompt_tokens, token_ids, spec ...), with the reference
    record's spec fields filled in from fidelity.json when the line lacks them."""
    lines = [json.loads(x) for x in Path(probe_jsonl).read_text().splitlines() if x.strip()]
    lines = [x for x in lines if x.get("dump_ok", True)]
    if reference_json:
        ref = json.loads(Path(reference_json).read_text())
        by_id = {r["id"]: r for r in ref["passes"][pass_name]["records"]}
        for line in lines:
            r = by_id.get(line["id"])
            if r is not None:
                line.setdefault("spec", r.get("spec"))
                line.setdefault("token_ids", r.get("token_ids"))
    return lines


# ---------------------------------------------------------------------------------------------------------- CLI


def build_model(args, device) -> MTPReplica:
    weights = load_draft_dir(args.draft, device=device)
    hot = load_token_map(args.token_map, weights.cfg.vocab) if args.token_map else None
    model = MTPReplica(weights, hot_ids=hot, compute_dtype=torch.bfloat16, fp8_kv=not args.no_fp8_kv, device=device)
    if getattr(args, "trained", None):
        loaded = model.load_dense(load_dense_file(args.trained))
        print(f"loaded {len(loaded)} trained tensors from {args.trained}", flush=True)
    model.eval()
    return model


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("inspect", help="print the draft's configuration and tensor summary")
    i.add_argument("--draft", type=Path, required=True)
    c = sub.add_parser("check", help="the replica check of section 5.2 on probe dumps")
    c.add_argument("--draft", type=Path, required=True, help="albucino's draft directory (original weights)")
    c.add_argument("--token-map", type=Path, required=True, help="the FR-Spec map the reference run served with")
    c.add_argument("--dump", type=Path, required=True, help="the probe dump directory (KEEP=all, BF16)")
    c.add_argument("--probe", type=Path, required=True, help="scripts/mtp_probe_dump.py's probe-dump.jsonl")
    c.add_argument("--reference", type=Path, default=None, help="the reference fidelity.json (spec_* per request)")
    c.add_argument("--pass", dest="pass_name", default="seq")
    c.add_argument("--variants", default="full,dense,cut2048,cut256")
    c.add_argument("--gate-variant", default="full")
    c.add_argument("--embed", choices=("prefill", "shifted"), default="prefill")
    c.add_argument("--trained", type=Path, default=None, help="dense tensors to load over the original ones")
    c.add_argument("--no-fp8-kv", action="store_true", help="keep K/V in BF16 (the server uses FP8)")
    c.add_argument("--max-requests", type=int, default=None)
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.cmd == "inspect":
        weights = load_draft_dir(args.draft, experts=False, embed=False, lm_head=False)
        print(json.dumps(dataclasses.asdict(weights.cfg), indent=1))
        for name, t in weights.dense.items():
            print(f"{name:70s} {tuple(t.shape)} {t.dtype} rms={t.float().pow(2).mean().sqrt():.4g}")
        return 0
    model = build_model(args, device)
    records = load_probe_records(args.probe, args.reference, args.pass_name)
    if args.max_requests:
        records = records[:args.max_requests]
    result = replica_check(model, args.dump, records, variants=tuple(v for v in args.variants.split(",") if v),
                           embed_mode=args.embed, device=device, gate_variant=args.gate_variant)
    result["args"] = {k: str(v) for k, v in vars(args).items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for name, s in result["summary"].items():
        print(f"{name:>8}: n={s['requests']} replica {s['replica_mean']} sglang {s['sglang_mean']} "
              f"diff {s['mean_diff']} mae {s['mae']} r {s['pearson']}", flush=True)
    verdict = result["verdict"] or {"go": False, "reasons": ["no gate variant"]}
    print(f"replica check: {'GO' if verdict['go'] else 'NO-GO'} {verdict['reasons']}", flush=True)
    return 0 if verdict["go"] else 2


if __name__ == "__main__":
    sys.exit(main())
