"""Test helpers for the MTP draft fine-tune (step 2): a tiny albucino-style draft directory and tiny step-1 dumps,
written with numpy only (the writer's tests run without torch)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mtp_write_draft as wd  # noqa: E402
import sglang_hc_dump_patch as hd  # noqa: E402

# albucino's runtime/mtp-int4-g32/config.json quantization_config (reap/dl-albucino-meta/mtp-config.json.txt)
ALBUCINO_QUANT = {
    "config_groups": {"mtp_routed_experts": {
        "format": "pack-quantized", "input_activations": None, "output_activations": None, "targets": ["RoutedExperts"],
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": 32, "num_bits": 4,
                    "observer": "minmax", "observer_kwargs": {}, "strategy": "group", "symmetric": True,
                    "type": "int"}}},
    "format": "pack-quantized", "global_compression_ratio": None, "ignore": [], "kv_cache_scheme": None,
    "quant_method": "compressed-tensors", "quantization_status": "compressed", "sparsity_config": {},
    "transform_config": {}}

# the real text_config fields the replica reads (albucino's config.json, section 1 of the plan)
REAL_TEXT = {"hidden_size": 2560, "hc_count": 4, "hc_lowrank": 320, "num_attention_heads": 24, "num_key_value_heads": 2,
             "head_dim": 256, "partial_rotary_factor": 0.25,
             "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                                 "rope_theta": 10000000, "rope_type": "default"},
             "num_experts": 512, "num_experts_per_tok": 10, "moe_intermediate_size": 640,
             "shared_expert_intermediate_size": 640, "indexer_n_heads": 4, "indexer_kv_heads": 1,
             "indexer_head_dim": 128, "indexer_budget": 2048, "indexer_compress_ratio": 4, "rms_norm_eps": 1e-6,
             "vocab_size": 248320, "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False}

# albucino's mtp-dense.safetensors header (scratchpad dl-albucino/dense-header.json): the 29 mtp.* shapes
ALBUCINO_DENSE = {
    "mtp.fc_embedding.weight": (2560, 2560), "mtp.fc_hidden.weight": (2560, 2560),
    "mtp.hyper_connection_mixer.hc_norm.weight": (10240,),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (320, 10240),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": (4, 10240),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": (10240,),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": (320, 10240),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.mlp.gate.weight": (512, 2560), "mtp.layers.0.mlp.shared_expert.down_proj.weight": (2560, 640),
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight": (640, 2560),
    "mtp.layers.0.mlp.shared_expert.up_proj.weight": (640, 2560),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 2560),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight": (4, 10240),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": (10240,),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight": (320, 10240),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": (10240, 320),
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (640, 2560),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (128,),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (128,), "mtp.layers.0.self_attn.k_norm.weight": (256,),
    "mtp.layers.0.self_attn.k_proj.weight": (512, 2560), "mtp.layers.0.self_attn.o_proj.weight": (2560, 6144),
    "mtp.layers.0.self_attn.q_norm.weight": (256,), "mtp.layers.0.self_attn.q_proj.weight": (12288, 2560),
    "mtp.layers.0.self_attn.v_proj.weight": (512, 2560), "mtp.pre_fc_norm_embedding.weight": (2560,),
    "mtp.pre_fc_norm_hidden.weight": (10240,)}

TINY_TEXT = {"hidden_size": 32, "hc_count": 4, "hc_lowrank": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
             "head_dim": 16, "partial_rotary_factor": 0.25,
             "rope_parameters": {"mrope_interleaved": True, "mrope_section": [1, 1, 0], "partial_rotary_factor": 0.25,
                                 "rope_theta": 10000.0, "rope_type": "default"},
             "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
             "shared_expert_intermediate_size": 32, "indexer_n_heads": 2, "indexer_kv_heads": 1, "indexer_head_dim": 8,
             "indexer_budget": 16, "indexer_compress_ratio": 4, "rms_norm_eps": 1e-6, "vocab_size": 96}
TINY_GROUP = 32  # the launcher requires group_size 32; the tiny widths are multiples of it
TOKENIZER_STUBS = {"tokenizer.json": '{"stub": "draft"}', "tokenizer_config.json": "{}",
                   "chat_template.jinja": "{{ messages }}", "generation_config.json": "{}",
                   "preprocessor_config.json": "{}", "video_preprocessor_config.json": "{}", "vocab.json": "{}"}


def bf16_bits(x: np.ndarray) -> np.ndarray:
    return hd.bf16_bits(np.asarray(x, np.float32))


def pack_int4(q: np.ndarray) -> np.ndarray:
    """int values in [-8, 7] [out, in] -> int32 [out, in/8], low nibble first, + 8 (compressed-tensors)."""
    out = q.shape[0]
    u = (q.astype(np.int64) + 8).reshape(out, -1, 8)
    packed = (u << (4 * np.arange(8, dtype=np.int64))).sum(-1)
    return np.where(packed >= 2 ** 31, packed - 2 ** 32, packed).astype(np.int32)


def tiny_tensors(text: dict, seed: int = 0, group: int = TINY_GROUP, scale: float = 0.08):
    """Random dense (float32), embed/lm_head (float32) and expert (packed, scale, shape) arrays."""
    rng = np.random.default_rng(seed)
    dense = {n: (rng.standard_normal(s) * (0.05 if len(s) == 1 else scale)).astype(np.float32)
             for n, s in wd.dense_layout(text).items()}
    V, H = int(text["vocab_size"]), int(text["hidden_size"])
    embed = (rng.standard_normal((V, H)) * 0.5).astype(np.float32)
    head = (rng.standard_normal((V, H)) * scale).astype(np.float32)
    experts = {}
    for name, spec in wd.expert_layout(text, group).items():
        if name.endswith("weight_packed"):
            o, packed_in = spec[1]
            experts[name] = pack_int4(rng.integers(-8, 8, size=(o, packed_in * 8)))
        elif name.endswith("weight_scale"):
            experts[name] = bf16_bits(np.abs(rng.standard_normal(spec[1])) * 0.02 + 0.005)
        else:
            experts[name] = np.array(spec[2], np.int64)
    return dense, embed, head, experts


def write_tiny_draft(path: Path, *, text: dict | None = None, seed: int = 0, group: int = TINY_GROUP,
                     quant: dict | None = None) -> dict:
    """An albucino-style runtime/mtp-int4-g32 directory: config.json, the index, mtp-dense.safetensors (lm_head and
    embed_tokens first, then the 29 dense tensors in name order, as albucino's), mtp-routed-experts-int4.safetensors
    and the small tokenizer files. Returns the arrays written."""
    text = dict(text or TINY_TEXT)
    path.mkdir(parents=True, exist_ok=True)
    dense, embed, head, experts = tiny_tensors(text, seed, group)
    q = json.loads(json.dumps(quant or ALBUCINO_QUANT))
    cfg = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp", "quantization_config": q,
           "text_config": {**text, "model_type": "qwen4_exp_text"}, "tie_word_embeddings": False}
    (path / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    items = [("lm_head.weight", "BF16", bf16_bits(head)),
             ("model.language_model.embed_tokens.weight", "BF16", bf16_bits(embed))]
    items += [(n, "BF16", bf16_bits(dense[n])) for n in sorted(dense)]
    hd.write_safetensors(path / "mtp-dense.safetensors", hd.encode_safetensors(items, {}))
    eitems = []
    for name in sorted(experts):
        arr = experts[name]
        dtype = {"weight_packed": "I32", "weight_scale": "BF16", "weight_shape": "I64"}[name.rsplit(".", 1)[1]]
        eitems.append((name, dtype, arr))
    hd.write_safetensors(path / "mtp-routed-experts-int4.safetensors", hd.encode_safetensors(eitems, {}))
    weight_map = {n: "mtp-dense.safetensors" for n, _, _ in items}
    weight_map.update(dict.fromkeys(experts, "mtp-routed-experts-int4.safetensors"))
    (path / wd.INDEX).write_text(json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}, indent=2))
    for name, body in TOKENIZER_STUBS.items():
        (path / name).write_text(body)
    (path / "compact_sources.json").write_text('{"format": "qwen38-flash-next-mtp-int4-g32-compact-v1"}')
    return {"dense": dense, "embed": embed, "head": head, "experts": experts, "text": text}


def write_trained(path: Path, text: dict, names, seed: int = 1, scale: float = 0.08) -> dict:
    """A trained-dense.safetensors (BF16) for ``names``."""
    rng = np.random.default_rng(seed)
    shapes = wd.dense_layout(text)
    arrays = {n: (rng.standard_normal(shapes[n]) * scale).astype(np.float32) for n in names}
    hd.write_safetensors(path, hd.encode_safetensors([(n, "BF16", bf16_bits(arrays[n])) for n in names], {}))
    return arrays


def write_target_mixer(path: Path, text: dict, seed: int = 2, scale: float = 0.08) -> dict:
    """model.language_model.hyper_connection_mixer.* (the target's final mixer) at the tiny shapes."""
    rng = np.random.default_rng(seed)
    H, hc, R = int(text["hidden_size"]), int(text["hc_count"]), int(text["hc_lowrank"])
    arrays = {"hc_norm.weight": rng.standard_normal(hc * H) * 0.05,
              "input_mix_weight_down.weight": rng.standard_normal((R, hc * H)) * scale,
              "input_mix_weight_up.weight": rng.standard_normal((hc * H, R)) * scale}
    hd.write_safetensors(path, hd.encode_safetensors(
        [("model.language_model.hyper_connection_mixer." + k, "BF16", bf16_bits(v)) for k, v in arrays.items()], {}))
    return arrays


# --- tiny step-1 dumps ---------------------------------------------------------------------------------------------

S, E, A, NL, T = hd.IM_START, hd.IM_END, hd.ASSISTANT, hd.NEWLINE, hd.THINK
USER = 846
PAD = hd.MM_PAD_MIN + 7


def conversation(turn_lengths, *, user_len: int = 9, images: int = 2, body_vocab: int = 90, seed: int = 0) -> list:
    """Token ids of a rendered conversation: a system turn, then per assistant turn a user turn (with image pads) and
    the assistant turn (header, body, <|im_end|>), then the generation prompt."""
    rng = np.random.default_rng(seed)
    ids = [S, 8678, NL] + [int(x) for x in rng.integers(3, body_vocab, 5)] + [E, NL]
    for n in turn_lengths:
        ids += [S, USER, NL] + [PAD] * images + [int(x) for x in rng.integers(3, body_vocab, user_len)] + [E, NL]
        ids += [S, A, NL, T, NL] + [int(x) for x in rng.integers(3, body_vocab, n)] + [E, NL]
    ids += [S, USER, NL] + [int(x) for x in rng.integers(3, body_vocab, user_len)] + [E, NL]
    return ids + [S, A, NL, T, NL]


def dump_request(dumper, rid: str, ids: list, width: int, *, chunk: int = 32, hidden: int = 32, seed: int = 0,
                 plan: dict | None = None, mm: bool = True) -> np.ndarray:
    """Dump one request through the real Dumper (as the patched server would, chunk by chunk); returns its H rows."""
    rng = np.random.default_rng(seed)
    hc = rng.standard_normal((len(ids), width)).astype(np.float32)
    emb = rng.standard_normal((len(ids), hidden)).astype(np.float32)
    if plan is not None:
        dumper._setup()
        (dumper.out / "plans" / f"{hd.safe_rid(rid)}.json").write_text(json.dumps(plan))
    arr = np.asarray(ids, np.int64)
    for lo in range(0, len(ids), chunk):
        dumper.process_chunk(rid, lo, arr[lo:lo + chunk], hc[lo:lo + chunk],
                             positions=np.arange(lo, min(len(ids), lo + chunk)),
                             mm_embeds=emb[lo:lo + chunk] if mm else None)
    return hc
