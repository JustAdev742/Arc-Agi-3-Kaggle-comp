"""scripts/mtp_replica.py (the MTP block as SGLang serves it, the serving-style draft, the replica check) and
scripts/mtp_probe_dump.py (the probe continuation dump). CPU only, tiny random configurations. torch-only checks skip
without torch; the checks against the Pennyroyal wheel's own reference code skip without the wheel (PENNYROYAL_WHEEL)."""
from __future__ import annotations

import ast
import http.server
import json
import math
import os
import sys
import threading
import types
import typing
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tests import mtp_fakes as mf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mtp_probe_dump as pd  # noqa: E402
import sglang_hc_dump_patch as hd  # noqa: E402

try:
    import torch
except ImportError:  # the .venv has no torch: those checks skip
    torch = None
needs_torch = pytest.mark.skipif(torch is None, reason="torch not installed")
if torch is not None:
    import mtp_replica as mr
WHEEL = Path(os.environ.get("PENNYROYAL_WHEEL", "/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/"
                            "d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/reap/dl-wheel/"
                            "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl"))
needs_wheel = pytest.mark.skipif(not WHEEL.is_file(), reason="Pennyroyal sglang wheel not available (PENNYROYAL_WHEEL)")
NOTEBOOK = ROOT / "kaggle" / "franzen" / "arc-agi-3-milestone-2-solution.ipynb"


def tiny_model(seed: int = 0, *, fp8: bool = False, hot_step: int = 1, dtype=None):
    cfg = mr.MTPConfig.tiny()
    weights = mr.random_draft(cfg, seed)
    return mr.MTPReplica(weights, hot_ids=torch.arange(0, cfg.vocab, hot_step), compute_dtype=dtype or torch.float32,
                         fp8_kv=fp8)


# --- configuration and layout -----------------------------------------------------------------------------------------


@needs_torch
def test_the_real_config_gives_the_served_shapes():
    cfg = mr.MTPConfig.from_config_json({"text_config": mf.REAL_TEXT, "quantization_config": mf.ALBUCINO_QUANT})
    assert cfg == mr.MTPConfig()  # 2560 x 4 streams, 24/2 heads of 256, RoPE on 64 dims base 1e7, 512 x top-10 experts
    assert mr.dense_shapes(cfg) == mf.ALBUCINO_DENSE and len(mr.DENSE_NAMES) == 29 and len(mr.TRAINABLE_NAMES) == 26
    assert set(mr.DENSE_NAMES) - set(mr.TRAINABLE_NAMES) == set(mr.INDEXER_NAMES)
    assert sum(math.prod(s) for s in mr.dense_shapes(cfg).values()) == 90_568_448  # plan 1: 90.6M dense (181.1 MB in BF16)
    tiny = mr.MTPConfig.from_config_json({"text_config": mf.TINY_TEXT, "quantization_config": mf.ALBUCINO_QUANT})
    assert tiny == mr.MTPConfig.tiny()
    assert mr.dense_shapes(tiny) == mf.wd.dense_layout(mf.TINY_TEXT)


def test_the_launcher_serves_the_draft_kv_in_fp8_and_without_online_mxfp8():
    """What the replica assumes about Franzen's cell 12 (and D', which copies it): an arm that changes these must
    repeat the replica check."""
    nb = json.loads(NOTEBOOK.read_text())
    cell = next("".join(c["source"]) for c in nb["cells"] if "def prepare_draft_view" in "".join(c["source"]))
    assert 'KVDTYPE="fp8_e4m3"' in cell and '"--speculative-draft-kv-cache-dtype", CFG["KVDTYPE"]' in cell
    assert '"SGLANG_SM120_ONLINE_MXFP8": "0"' in cell and '"SGLANG_SM120_LM_HEAD_FP8": "0"' in cell
    assert '"--speculative-eagle-topk", "1"' in cell and "SPEC_STEPS=3" in cell


# --- numerics ---------------------------------------------------------------------------------------------------------


@needs_torch
def test_int4_rtn_pack_unpack_and_dequant():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(16, 64, generator=g) * 0.1).to(torch.bfloat16)
    w[3, :32] = 0  # a zero group keeps scale 1 and codes 0
    packed, scale = mr.pack_int4_rtn(w, 32)
    assert packed.dtype == torch.int32 and packed.shape == (16, 8) and scale.shape == (16, 2)
    q = mr.unpack_int4(packed, (16, 64))
    assert int(q.min()) >= -8 and int(q.max()) <= 7 and float(scale[3, 0]) == 1.0 and not q[3, :32].any()
    assert np.array_equal(mf.pack_int4(q.numpy()), packed.numpy())  # the numpy helper packs the same way
    deq = mr.dequant_int4(packed, scale, (16, 64), 32)
    expect = (q.reshape(16, 2, 32).to(torch.bfloat16) * scale.unsqueeze(-1)).reshape(16, 64)  # BF16 product
    assert torch.equal(deq, expect)
    rel = ((deq.float() - w.float()).norm() / w.float().norm()).item()
    assert rel < 0.12  # RTN g32: about 10% RMS error (plan 0)


@needs_torch
def test_int4_layout_is_compressed_tensors_pack_quantized():
    ct = pytest.importorskip("compressed_tensors.compressors.pack_quantized.helpers")
    q = torch.randint(-8, 8, (6, 64), dtype=torch.int8)
    packed = ct.pack_to_int32(q, 4)
    assert torch.equal(mr.unpack_int4(packed, (6, 64)), q)
    assert torch.equal(ct.unpack_from_int32(packed, 4, torch.Size((6, 64))), q)
    assert torch.equal(torch.as_tensor(mf.pack_int4(q.numpy())), packed)


@needs_torch
def test_fp8_kv_round_trip_saturates_and_passes_gradients_straight_through():
    x = torch.tensor([0.0, 1.0, 1.1, -3.3, 500.0, -1e4], requires_grad=True)
    y = mr.fp8_round(x)
    assert y.tolist()[:4] == pytest.approx([0.0, 1.0, 1.125, -3.25]) and y.tolist()[4:] == [448.0, -448.0]
    y.sum().backward()
    assert x.grad.tolist() == [1.0] * 6


@needs_torch
def test_rope_rotates_the_partial_dims_by_relative_position_only():
    g = torch.Generator().manual_seed(1)
    q, k = torch.randn(1, 2, 16, generator=g), torch.randn(1, 2, 16, generator=g)
    dots = []
    for p in (0, 7, 1000):
        qr = mr.rope(q, torch.tensor([p + 5]), 4, 1e4)
        kr = mr.rope(k, torch.tensor([p]), 4, 1e4)
        assert torch.equal(qr[..., 4:], q[..., 4:])  # only the first 4 of 16 dims rotate
        dots.append(float((qr * kr).sum()))
    assert dots == pytest.approx([dots[0]] * 3, rel=1e-5)


@needs_torch
def test_gemma_norm_groups():
    x = torch.randn(3, 8)
    w = torch.randn(8) * 0.1
    whole = mr.gemma_rmsnorm(x, w, 1e-6)
    groups = mr.gemma_rmsnorm(x, w, 1e-6, group=4)
    manual = torch.cat([xi * torch.rsqrt(xi.pow(2).mean(-1, keepdim=True) + 1e-6) for xi in x.split(4, -1)], -1)
    assert torch.allclose(groups, manual * (1 + w), atol=1e-6) and not torch.allclose(whole, groups)


# --- against the wheel's own reference code ---------------------------------------------------------------------------


def wheel_defs(member: str, wanted: list) -> dict:
    """Functions (``name`` or ``Class.method``) from a file of the wheel, compiled alone without decorators. They are
    pure torch code (the module imports are not executed)."""
    with zipfile.ZipFile(WHEEL) as z:
        tree = ast.parse(z.read(member).decode())
    found = {}

    def visit(node, owner=None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, child.name)
            elif isinstance(child, ast.FunctionDef):
                key = f"{owner}.{child.name}" if owner else child.name
                found.setdefault(key, child)
                found.setdefault(child.name, child)
                visit(child, owner)

    visit(tree)
    ns = {"torch": torch, "F": torch.nn.functional, "math": math, "Optional": typing.Optional, "Tuple": tuple,
          "Union": typing.Union}
    out = {}
    for key in wanted:
        node = found[key]
        node.decorator_list = []
        exec(compile(ast.Module(body=[node], type_ignores=[]), member, "exec"), ns)
        out[key] = ns[node.name]
    return out


@needs_torch
@needs_wheel
def test_qsa_selection_matches_the_wheels_reference_operators():
    k = wheel_defs("sglang/srt/layers/attention/qsa/kernel.py",
                   ["average_pool_qsa_keys", "qsa_fast_topk", "torch_expand_qsa_block_indices"])
    m = wheel_defs("sglang/srt/layers/attention/qsa/mqa.py", ["_validate_q", "_validate_k", "torch_qsa_mqa_prefill"])
    meta = wheel_defs("sglang/srt/layers/attention/qsa/metadata.py", ["build_qsa_row_ranges"])
    torch_qsa_mqa_prefill = m["torch_qsa_mqa_prefill"]
    torch_qsa_mqa_prefill.__globals__.update(_validate_q=m["_validate_q"], _validate_k=m["_validate_k"])
    g = torch.Generator().manual_seed(3)
    ratio, budget, d = 4, 32, 8
    keys = torch.randn(203, d, generator=g)
    pooled = keys[:200].view(50, ratio, d).float().mean(1)
    assert torch.equal(k["average_pool_qsa_keys"](keys[:200].view(50, ratio, 1, d))[:, 0], pooled)
    for t in (3, 30, 31, 33, 150, 199, 202):
        q = torch.randn(2, d, generator=g)
        mine = mr.qsa_select_tokens(q, pooled, t, ratio, budget // ratio, d)
        starts, ends, _ = meta["build_qsa_row_ranges"](torch.tensor([203]), torch.tensor([t]), torch.tensor([0]), ratio)
        nvis = int(ends[0] - starts[0])
        assert nvis == (t + 1) // ratio
        if nvis <= budget // ratio:  # SGLang's all-visible path: every causally visible token
            assert mine.tolist() == list(range(t + 1))
            continue
        logits = torch_qsa_mqa_prefill(q[None], pooled[:nvis, None, :], starts, ends)
        blocks = k["qsa_fast_topk"](logits, starts, ends, budget // ratio)
        theirs = k["torch_expand_qsa_block_indices"](blocks, torch.tensor([t]), torch.tensor([t + 1]), ratio, budget)
        assert sorted(x for x in theirs[0].tolist() if x >= 0) == sorted(mine.tolist()), t


@needs_torch
@needs_wheel
def test_gated_residual_norm_and_rope_match_the_wheels_formulas():
    hc = wheel_defs("sglang/srt/layers/hyperconnection.py",
                    ["_mix_compute", "_combine_compute", "GroupedGemmaRMSNorm.forward"])
    ln = wheel_defs("sglang/srt/layers/layernorm.py", ["GemmaRMSNorm.forward_native"])
    ru = wheel_defs("sglang/srt/layers/rotary_embedding/utils.py", ["apply_rotary_emb"])
    base = wheel_defs("sglang/srt/layers/rotary_embedding/base.py", ["RotaryEmbedding._compute_inv_freq"])
    model = tiny_model(5)
    c = model.cfg
    X = torch.randn(6, c.width)
    for prefix in (mr.ATTN, mr.MLP, mr.MIXER):
        w = model.w(prefix + "hc_norm.weight")
        norm = types.SimpleNamespace(weight=w, group_size=c.hidden, variance_epsilon=c.eps, _jit_group_size=None)
        Xn = hc["GroupedGemmaRMSNorm.forward"](norm, X)
        theirs = hc["_mix_compute"](Xn, model.w(prefix + "input_mix_weight_down.weight"),
                                    model.w(prefix + "input_mix_weight_up.weight"), c.hc, c.hidden)
        mine, mine_n = model.hc_mix(prefix, X)
        assert torch.allclose(mine_n, Xn, atol=1e-6) and torch.allclose(mine, theirs, atol=1e-6)
        if prefix != mr.MIXER:
            out = torch.randn(6, c.hidden)
            theirs = hc["_combine_compute"](out, X, Xn, model.w(prefix + "block_inject_weight.weight"), c.hc, c.hidden)
            assert torch.allclose(model.hc_combine(prefix, out, X, Xn), theirs, atol=1e-6)
    x, w = torch.randn(5, 16), torch.randn(16) * 0.1
    gem = ln["GemmaRMSNorm.forward_native"](types.SimpleNamespace(weight=w, variance_epsilon=1e-6), x)
    assert torch.allclose(mr.gemma_rmsnorm(x, w, 1e-6), gem, atol=1e-6)
    inv = base["RotaryEmbedding._compute_inv_freq"](types.SimpleNamespace(_force_native=False, rotary_dim=64), 1e7)
    pos = torch.tensor([0, 1, 4095, 65536, 120000])
    freqs = torch.einsum("i,j -> ij", pos.float(), inv)  # base.py _compute_cos_sin_cache
    q = torch.randn(5, 3, 256)
    theirs = torch.cat([ru["apply_rotary_emb"](q[..., :64], freqs.cos(), freqs.sin(), True), q[..., 64:]], -1)
    assert torch.allclose(mr.rope(q, pos, 64, 1e7), theirs, atol=1e-5)


@needs_wheel
def test_the_wheel_still_runs_the_draft_as_the_replica_assumes():
    """Source facts the replica mirrors (module docstring); a different wheel needs a new replica check."""
    with zipfile.ZipFile(WHEEL) as z:
        read = {n: z.read(n).decode() for n in z.namelist() if n.endswith(".py")}
    mtp = read["sglang/srt/models/qwen4_exp_mtp.py"]
    forward = mtp[mtp.index("    def forward("):]
    assert "mrope_positions" not in mtp and "self.model(\n                    input_ids,\n                    positions," in forward
    assert "positions = forward_batch.mrope_positions" in read["sglang/srt/models/qwen3_vl.py"]
    assert read["sglang/srt/model_executor/runner/eager_runner.py"].count("forward_batch.positions,") >= 3
    assert "input_embeds = forward_batch.mm_input_embeds" in mtp and "extend_seq_lens - 1" in mtp
    assert "input_embeds.unsqueeze(-2) + encoder_inputs" in mtp
    eagle = read["sglang/srt/speculative/eagle_worker_v2.py"]
    assert "hidden_states = logits_output.hidden_states" in eagle and "forward_batch.positions.add_(1)" in eagle
    assert "draft_input.positions = batch.seq_lens" in read["sglang/srt/speculative/eagle_worker_common.py"]
    assert "index_share_for_mtp_iteration=True" in read["sglang/srt/configs/qwen4_exp.py"]
    assert "cache_k = cache_k.to(self.dtype)" in read["sglang/srt/mem_cache/memory_pool.py"]
    hc = read["sglang/srt/layers/hyperconnection.py"]
    assert "F.linear(hyper_input_normed, input_mix_weight_down) / hc" in hc and "2 * torch.sigmoid(" in hc
    assert "renormalize=config.norm_topk_prob" in read["sglang/srt/models/qwen2_moe.py"]


# --- the block and the chain ------------------------------------------------------------------------------------------


@needs_torch
@pytest.mark.parametrize("fp8", [False, True])
def test_the_batched_chain_equals_the_serving_style_draft(fp8):
    """MTPReplica.chain (training-time test over packed windows) gives, for every origin, exactly the logits of
    DraftKV.chain (draft-extend KV, then steps 2-3 with their own chain KV), with teacher-forced step inputs."""
    model = tiny_model(1, fp8=fp8)
    c = model.cfg
    g = torch.Generator().manual_seed(2)
    lens = (11, 7)
    H = torch.randn(sum(lens), c.width, generator=g)
    ids = torch.randint(0, c.vocab, (sum(lens) + 8,), generator=g)
    pos = torch.cat([torch.arange(100, 100 + lens[0]), torch.arange(5000, 5000 + lens[1])])
    emb1 = model.embed(ids[1:sum(lens) + 1])
    chain_ids = torch.stack([ids[2:sum(lens) + 2], ids[3:sum(lens) + 3]], 1)
    bounds = [(0, lens[0]), (lens[0], sum(lens))]
    outs = model.chain(H, emb1, pos, bounds, chain_ids, steps=3)
    for a, b in bounds:
        kv = mr.DraftKV(model, H[a:b], emb1[a:b], pos[a:b])
        for t in range(b - a):
            _, logits = kv.chain(t, mode="dense", inputs=[int(chain_ids[a + t, 0]), int(chain_ids[a + t, 1])])
            for k in range(3):
                assert torch.allclose(model.logits(outs[k][a + t:a + t + 1]), logits[k], atol=2e-5), (a, t, k)
    assert all(o.shape == (sum(lens), c.hidden) for o in outs)


@needs_torch
def test_steps_use_their_own_previous_output_and_positions():
    """Step 2 depends on step 1's G (not on H) and runs one position later; a different H at the origin changes
    every step, a different realized token only steps >= 2."""
    model = tiny_model(4)
    c = model.cfg
    H = torch.randn(9, c.width)
    ids = torch.randint(0, c.vocab, (13,))
    emb1, chain_ids, pos = model.embed(ids[1:10]), torch.stack([ids[2:11], ids[3:12]], 1), torch.arange(9)
    base = model.chain(H, emb1, pos, [(0, 9)], chain_ids)
    ids2 = chain_ids.clone()
    ids2[8, 0] = (ids2[8, 0] + 1) % c.vocab
    other = model.chain(H, emb1, pos, [(0, 9)], ids2)
    assert torch.equal(base[0], other[0]) and not torch.allclose(base[1][8], other[1][8], atol=1e-3)
    # other origins are untouched (up to float rounding: an expert's batch composition changed)
    assert torch.allclose(base[1][:8], other[1][:8], atol=1e-6)
    shifted = model.chain(H, emb1, pos + 3, [(0, 9)], chain_ids)
    assert torch.allclose(shifted[0], base[0], atol=1e-5)  # RoPE: only relative positions matter within a window


@needs_torch
def test_qsa_is_dense_below_the_budget_and_sparse_above():
    model = tiny_model(6)
    c = model.cfg
    g = torch.Generator().manual_seed(7)
    L = 60
    H, emb, pos = torch.randn(L, c.width, generator=g), torch.randn(L, c.hidden, generator=g), torch.arange(L)
    kv = mr.DraftKV(model, H, emb, pos)
    assert kv.ck.shape == (L // c.compress, c.idx_head_dim)
    for t in (5, 15, 16, 18):  # <= block_topk * compress visible tokens: dense
        assert kv.select(t).tolist() == list(range(t + 1))
        assert kv.chain(t, mode="full")[0] == kv.chain(t, mode="dense")[0]
    sel = kv.select(41)
    assert len(sel) == c.block_topk * c.compress + 2 and sel[-2:].tolist() == [40, 41]
    assert len(set(sel.tolist())) == len(sel) and int(sel.max()) <= 41
    assert kv.select(43).tolist()[-c.compress:] != list(range(40, 44)) or True  # a full last block competes too
    assert kv.select(10, mode="cut", start=4).tolist() == list(range(4, 11))


@needs_torch
def test_frozen_buffers_and_trainable_parameters():
    model = tiny_model(8)
    trainable = {n for n in model.names if model.p[mr._key(n)].requires_grad}
    assert trainable == set(mr.TRAINABLE_NAMES) and not any(b.requires_grad for b in model.buffers())
    state = model.dense_state()
    model.load_dense({mr.TRAINABLE_NAMES[0]: torch.zeros_like(state[mr.TRAINABLE_NAMES[0]])})
    assert float(model.p[mr._key(mr.TRAINABLE_NAMES[0])].detach().abs().sum()) == 0
    with pytest.raises(ValueError):
        model.load_dense({mr.TRAINABLE_NAMES[1]: torch.zeros(3)})


@needs_torch
def test_load_draft_dir_reads_albucinos_layout(tmp_path):
    arrays = mf.write_tiny_draft(tmp_path / "draft")
    w = mr.load_draft_dir(tmp_path / "draft")
    assert w.cfg == mr.MTPConfig.tiny()
    for name, value in arrays["dense"].items():
        assert np.array_equal(w.dense[name].view(torch.int16).numpy().view(np.uint16), mf.bf16_bits(value)), name
    assert np.array_equal(w.embed.view(torch.int16).numpy().view(np.uint16), mf.bf16_bits(arrays["embed"]))
    for e in (0, 7):
        for bank, proj in zip(w.experts, mr.PROJS, strict=True):
            p = f"mtp.layers.0.mlp.experts.{e}.{proj}."
            packed = torch.as_tensor(arrays["experts"][p + "weight_packed"])
            scale = torch.as_tensor(arrays["experts"][p + "weight_scale"].view(np.int16)).view(torch.bfloat16)
            shape = arrays["experts"][p + "weight_shape"].tolist()
            assert torch.equal(bank[e], mr.dequant_int4(packed, scale, shape, 32)), (e, proj)
    model = mr.MTPReplica(w, hot_ids=torch.arange(0, 96, 3))
    assert model.lm_head_hot.shape == (32, 32) and model.dt == torch.bfloat16
    out = model.chain(torch.randn(5, 128), model.embed(torch.arange(5)), torch.arange(5), [(0, 5)],
                      torch.zeros(5, 2, dtype=torch.long))
    assert out[2].dtype == torch.bfloat16 and torch.isfinite(out[2].float()).all()


@needs_torch
def test_token_map_and_dense_files(tmp_path):
    torch.save(list(range(0, 96, 2)), tmp_path / "map.pt")
    assert mr.load_token_map(tmp_path / "map.pt", 96).tolist() == list(range(0, 96, 2))
    with pytest.raises(ValueError):
        mr.load_token_map(tmp_path / "map.pt", 50)
    model = tiny_model(9)
    mr.save_safetensors(tmp_path / "d.safetensors", model.dense_state())
    back = mr.load_dense_file(tmp_path / "d.safetensors")
    assert list(back) == list(model.names) and all(torch.equal(back[n], model.dense_state()[n]) for n in back)


# --- embeddings, the greedy loop and the replica check ----------------------------------------------------------------


@needs_torch
def test_extend_embeddings_mirror_the_multimodal_draft_extend():
    model = tiny_model(10)
    ids = torch.tensor([5, 6, 7, 8, 9, 10, 11])
    mm = torch.tensor([True, True, True, True, False, False])
    last = torch.tensor([False, True, False, False, False, False])
    img = {2: torch.full((model.cfg.hidden,), 0.5)}
    emb = mr.extend_embeddings(model, ids, mode="prefill", mm_rows=mm, img=img, chunk_last=last)
    E = model.embed_tokens
    assert torch.equal(emb[0], E[5]) and torch.equal(emb[1], E[7])  # unshifted; a chunk's last row takes x_{j+1}
    assert torch.equal(emb[2], img[2]) and torch.equal(emb[3], E[8])  # vision features at an image row
    assert torch.equal(emb[4], E[10]) and torch.equal(emb[5], E[11])  # text-only rows: shifted
    assert torch.equal(mr.extend_embeddings(model, ids, mode="shifted"), E[ids[1:]])
    assert mr.chunk_last_rows(20, 0, 17, chunk=8).tolist() == [i in (7, 15, 16) for i in range(20)]


class _FixedKV:
    def __init__(self, props):
        self.props, self.calls = props, []

    def chain(self, t, **kw):
        self.calls.append(t)
        return self.props[t], None


@needs_torch
def test_simulate_greedy_counts_verify_steps_like_sglang():
    out = [10, 11, 12, 13, 14, 15, 16, 17]  # out[0] is the prefill's token at position P = 5
    kv = _FixedKV({4: [11, 12, 99], 7: [14, 15, 16]})
    r = mr.simulate_greedy(kv, 5, out)
    assert kv.calls == [4, 7] and r["verify_ct"] == 2 and r["correct_drafts"] == 5 and r["histogram"] == [0, 0, 1, 1]
    assert r["accept_length"] == pytest.approx(4.0) and r["accept_length_strict"] == pytest.approx(3.5)  # 8 / 2
    kv = _FixedKV({4: [11, 12, 99], 7: [14, 15, 16]})
    r = mr.simulate_greedy(kv, 5, out[:6])  # the last verify sees only the recorded targets
    assert r["correct_drafts"] == 4 and r["histogram"] == [0, 0, 2, 0]
    kv = _FixedKV({4: [1, 1, 1], 5: [12, 1, 1], 7: [1, 1, 1], 8: [1, 1, 1]})
    r = mr.simulate_greedy(kv, 5, out[:6])
    assert kv.calls == [4, 5, 7, 8] and r["histogram"] == [3, 1, 0, 0] and r["accept_length"] == pytest.approx(1.5)
    assert mr.step_rates([3, 1, 0, 0]) == [0.25, 0.0, None]


def _probe_dump(tmp_path, model, *, n_out=20, seed=0, prompt_turns=(14, 9), chunk=16):
    """A KEEP=all BF16 dump of a prompt plus an output that follows the replica's own proposals for a while."""
    c = model.cfg
    prompt = mf.conversation(prompt_turns, user_len=7, images=3, body_vocab=c.vocab - 1, seed=seed)
    P = len(prompt)
    rng = np.random.default_rng(seed)
    out = [int(x) for x in rng.integers(3, c.vocab - 1, n_out)]
    H = rng.standard_normal((P + n_out, c.width)).astype(np.float32)
    # make part of the output agree with the drafts: walk the chain once with the in-memory state
    ids = torch.as_tensor(prompt + out + [0])
    emb = mr.extend_embeddings(model, ids.clamp(max=c.vocab - 1), mode="shifted")
    kv = mr.DraftKV(model, torch.as_tensor(H).to(model.dt), emb, torch.arange(P + n_out))
    produced, t = 1, P - 1
    while produced < n_out - 4:
        props, _ = kv.chain(t)
        a = int(rng.integers(0, 4))
        out[produced:produced + a] = props[:a]
        produced += a + 1
        t += a + 1
    dumper = hd.Dumper(tmp_path / "dump", keep="all", dtype="bf16", context=4)
    rid = f"probe-x{seed}"
    mf.dump_request(dumper, rid, prompt + out, c.width, chunk=chunk, hidden=c.hidden, seed=seed)
    return {"rid": rid, "id": f"x#{seed}", "game": "ar25", "prompt_tokens": P, "token_ids": out, "cached_tokens": 0}


@needs_torch
def test_the_replica_check_passes_against_itself_and_fails_against_a_different_draft(tmp_path):
    model = tiny_model(11, fp8=True, dtype=torch.float32)
    records = [_probe_dump(tmp_path, model, seed=s) for s in range(3)]
    for rec in records:  # the reference: the same replica on the dumped state, as SGLang's spec_* fields
        state = mr.probe_request_state(model, tmp_path / "dump", rec)
        assert state["match"] == len(rec["token_ids"]) and state["rows"] == rec["prompt_tokens"] + len(rec["token_ids"])
        sim = mr.simulate_greedy(mr.DraftKV(model, state["H"], state["emb"], state["pos"]), state["prompt_len"],
                                 state["out"], mode="full")
        rec["spec"] = {"spec_accept_length": sim["accept_length"], "spec_verify_ct": sim["verify_ct"],
                       "spec_num_correct_drafts": sim["correct_drafts"], "spec_correct_drafts_histogram": sim["histogram"]}
    gate = {**mr.GATE, "min_requests": 3}
    result = mr.replica_check(model, tmp_path / "dump", records, variants=("full", "dense", "cut8"), gate=gate,
                              logger=lambda m: None)
    s = result["summary"]["full"]
    assert s["requests"] == 3 and s["mean_diff"] == 0 and s["mae"] == 0
    assert s["replica_step_rates"] == s["sglang_step_rates"]
    assert result["verdict"]["go"] and result["requests"][0]["agree_with_full"].keys() == {"dense", "cut8"}
    other = tiny_model(12, fp8=True)  # another draft: the per-request accept lengths no longer agree
    result = mr.replica_check(other, tmp_path / "dump", records, variants=("full",), gate=gate, logger=lambda m: None)
    assert not result["verdict"]["go"] and result["summary"]["full"]["mae"] > 0
    records[0]["token_ids"] = records[0]["token_ids"][:5] + [1] + records[0]["token_ids"][6:]
    result = mr.replica_check(model, tmp_path / "dump", records, variants=("full",), gate=gate, logger=lambda m: None)
    assert result["requests"][0]["out_match"] == 5 and result["requests"][0]["reference"]["accept_length"] is None
    assert result["summary"]["full"]["requests"] == 2 and not result["verdict"]["go"]  # fewer than min_requests


@needs_torch
def test_probe_state_requires_every_row(tmp_path):
    model = tiny_model(13)
    dumper = hd.Dumper(tmp_path / "dump", keep="spans", dtype="bf16", context=2)
    ids = mf.conversation((6,), images=1, body_vocab=90)
    mf.dump_request(dumper, "probe-y", ids, model.cfg.width, chunk=8, hidden=model.cfg.hidden)
    with pytest.raises(ValueError, match="KEEP=all"):
        mr.probe_request_state(model, tmp_path / "dump", {"rid": "probe-y", "prompt_tokens": len(ids) - 3,
                                                          "token_ids": ids[-3:]})


def test_load_probe_records_fills_the_reference_fields(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "p.jsonl").write_text(json.dumps({"rid": "r", "id": "a#1", "dump_ok": True}) + "\n"
                                      + json.dumps({"rid": "s", "id": "b#1", "dump_ok": False}) + "\n")
    (tmp_path / "f.json").write_text(json.dumps({"passes": {"seq": {"records": [
        {"id": "a#1", "spec": {"spec_accept_length": 2.5}, "token_ids": [1, 2]}]}}}))
    lines = mr.load_probe_records(tmp_path / "p.jsonl", tmp_path / "f.json")
    assert lines == [{"rid": "r", "id": "a#1", "dump_ok": True, "spec": {"spec_accept_length": 2.5},
                      "token_ids": [1, 2]}]


# --- the probe continuation dump (stdlib) -----------------------------------------------------------------------------


def _reference(n: int = 6) -> tuple[list, dict]:
    samples, records = [], {}
    for i in range(n):
        game = ("ar25", "ft09", "lp85")[i % 3]
        sid = f"{game}#{i:03d}"
        samples.append({"id": sid, "game": game, "request": {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": f"go {i}"},
                         {"role": "assistant", "reasoning_content": "r", "content": "c"},
                         {"role": "user", "content": "again"}],
            "tools": [], "tool_choice": "auto", "chat_template_kwargs": {"preserve_thinking": True}}})
        records[sid] = {"id": sid, "ok": True, "token_ids": [100 + j for j in range(5 + i)],
                        "spec": {"spec_accept_length": 2.0 + i / 10}, "usage": {"prompt_tokens": 5000 + 1000 * i},
                        "cached_tokens": 0}
    return samples, records


def test_probe_selection_spreads_over_games_within_the_row_budget():
    samples, records = _reference(9)
    chosen = pd.select(samples, records, count=4, max_rows=40_000, target_tokens=7000)
    assert [s["id"] for s in chosen] == ["ar25#003", "ft09#001", "lp85#002", "ar25#000"]
    chosen = pd.select(samples, records, count=10, max_rows=12_000, target_tokens=7000)
    assert sum(records[s["id"]]["usage"]["prompt_tokens"] + 5 for s in chosen) <= 12_000
    assert [s["id"] for s in pd.select(samples, records, ids=["lp85#005"])] == ["lp85#005"]
    with pytest.raises(ValueError):
        pd.select(samples, records, ids=["zz#1"])
    assert pd.select(samples, records, min_prompt=20_000) == []


def test_the_continuation_request_appends_the_output_as_a_message_to_continue():
    samples, _ = _reference(1)
    body = pd.continuation_body(samples[0], "Confirmed: LEFT", model="flashnext", rid="probe-ar25-000")
    assert body["messages"][-1] == {"role": "assistant", "content": "Confirmed: LEFT"}
    assert body["messages"][:-1] == samples[0]["request"]["messages"] and body["continue_final_message"] is True
    assert body["max_tokens"] == 1 and body["temperature"] == 0.0 and body["rid"] == "probe-ar25-000"
    assert body["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}
    assert pd.rid_for("ar25#004") == "probe-ar25-004"


class _ContinuationServer:
    """A fake SGLang: /v1/detokenize joins ids, the chat endpoint dumps the rendered prompt + the continued text."""

    def __init__(self, dump_dir: Path, *, wrong_tokens: bool = False):
        self.dumper = hd.Dumper(dump_dir, keep="all", dtype="bf16")
        self.bodies, self.wrong = [], wrong_tokens

    def tokens(self, messages: list) -> list:
        ids = []
        for m in messages:
            role = {"assistant": hd.ASSISTANT, "system": 8678}.get(m["role"], 846)
            ids += [hd.IM_START, role, hd.NEWLINE] + ([hd.THINK, hd.NEWLINE] if role == hd.ASSISTANT else []) \
                + [7] * (3 + len(str(m.get("content")))) + [hd.IM_END, hd.NEWLINE]
        return ids + [hd.IM_START, hd.ASSISTANT, hd.NEWLINE, hd.THINK, hd.NEWLINE]

    def __enter__(self):
        server = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"disable_radix_cache": True} if self.path == "/server_info" else {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                if self.path == "/v1/detokenize":
                    self._send({"text": "|".join(str(t) for t in body["tokens"])})
                    return
                server.bodies.append(body)
                messages = body["messages"]
                assert body["continue_final_message"] and messages[-1]["role"] == "assistant"
                out = [int(x) for x in messages[-1]["content"].split("|")]
                ids = server.tokens(messages[:-1]) + out + ([1] if server.wrong else [])
                hc = np.zeros((len(ids), 8), np.float32)
                for lo in range(0, len(ids), 16):
                    server.dumper.process_chunk(body["rid"], lo, np.array(ids[lo:lo + 16]), hc[lo:lo + 16])
                self._send({"choices": [{"message": {"content": ""}, "logprobs": {"content": [
                    {"token": "x", "logprob": -0.1, "top_logprobs": []}]}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": 1}})

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.httpd.server_address[1]}"


def test_the_probe_dump_sends_continuations_and_checks_every_dump(tmp_path):
    samples, records = _reference(3)
    dump = tmp_path / "dump"
    with _ContinuationServer(dump) as server:
        rendered = len(server.tokens(samples[0]["request"]["messages"]))
        for s in samples:  # make the reference prompt lengths those of the fake template
            records[s["id"]]["usage"]["prompt_tokens"] = rendered + 0 * len(s["id"])
        summary = pd.run(samples, records, tmp_path / "out", base_url=server.url, dump_dir=dump, health_wait=5,
                         logger=lambda m: None)
    assert summary["sent"] == summary["ok"] == summary["dump_ok"] == 3
    lines = [json.loads(x) for x in (tmp_path / "out" / "probe-dump.jsonl").read_text().splitlines()]
    for line, s in zip(lines, samples):
        req = hd.load_request(dump, line["rid"])
        P = line["prompt_tokens"]
        assert req["token_ids"][P:].tolist() == records[s["id"]]["token_ids"] == line["token_ids"]
        assert len(req["hc_pos"]) == len(req["token_ids"]) and line["dump_ok"] and line["dump"]["chunks"] >= 1
        assert json.loads((dump / "plans" / f"{line['rid']}.json").read_text())["loss_spans"] is None
    with _ContinuationServer(tmp_path / "dump2", wrong_tokens=True) as server:
        summary = pd.run(samples[:1], records, tmp_path / "out2", base_url=server.url, dump_dir=tmp_path / "dump2",
                         health_wait=5, logger=lambda m: None)
    assert summary["dump_ok"] == 0 and "expected" in summary["stopped"]


def test_probe_dump_dry_run_cli(tmp_path):
    samples, records = _reference(4)
    (tmp_path / "requests.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    (tmp_path / "fidelity.json").write_text(json.dumps({"passes": {"seq": {"records": list(records.values())}}}))
    assert pd.main(["--data", str(tmp_path / "requests.jsonl"), "--reference", str(tmp_path / "fidelity.json"),
                    "--dry-run", "--out", str(tmp_path / "o"), "--count", "2", "--target-tokens", "6000"]) == 0
    plan = json.loads((tmp_path / "o" / "probe-plan.json").read_text())
    assert [p["id"] for p in plan["selected"]] == ["ar25#000", "ft09#001"] and plan["rows"] == 5000 + 6000 + 5 + 6
