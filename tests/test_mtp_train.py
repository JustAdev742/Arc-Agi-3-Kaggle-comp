"""scripts/mtp_train.py: windows over step-1 dumps (written by the real Dumper), the batch tensors, the chained KL loss,
frozen experts, a short training run that lowers the loss, the held-out evaluation, checkpoints/resume, and the hand-off
to scripts/mtp_write_draft.py. CPU only, tiny random configurations; skipped without torch."""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests import mtp_fakes as mf  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import mtp_replica as mr  # noqa: E402
import mtp_train as mt  # noqa: E402
import mtp_write_draft as wd  # noqa: E402

hd = mf.hd
CFG = mr.MTPConfig.tiny()


def rid(game: str, k: int) -> str:
    return f"hc-0123abcd-{game}-p0-r{k:04d}"


def make_dump(root: Path, *, dtype: str = "fp8", context: int = 16) -> tuple[Path, Path, dict]:
    """Two train-game and one held-out request through the real Dumper, with plan files and snapshots.jsonl."""
    dump = root / "dump"
    dumper = hd.Dumper(dump, context=context, keep="spans", dtype=dtype)
    convs = {rid("ls20", 1): ((30, 12, 50), [0, 2], 0), rid("ls20", 2): ((9, 70), [1], 1),
             rid("sb26", 1): ((25, 40), [0, 1], 2)}
    ids_of = {}
    lines = []
    for r, (turns, loss, seed) in convs.items():
        ids = mf.conversation(turns, user_len=7, images=2, body_vocab=CFG.vocab - 1, seed=seed)
        mf.dump_request(dumper, r, ids, CFG.width, chunk=16, hidden=CFG.hidden, seed=seed,
                        plan={"loss_spans": loss})
        ids_of[r] = ids
        game = r.split("-")[2]
        lines.append({"rid": r, "game": game, "split": "holdout" if game == "sb26" else "train", "loss_spans": loss})
    snaps = root / "snapshots.jsonl"
    snaps.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return dump, snaps, ids_of


def tiny_model(seed: int = 0, hot_step: int = 1):
    w = mr.random_draft(CFG, seed)
    return mr.MTPReplica(w, hot_ids=torch.arange(0, CFG.vocab, hot_step), compute_dtype=torch.float32, fp8_kv=True)


def tiny_target(model, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    mixer = {"hc_norm.weight": torch.randn(CFG.width, generator=g) * 0.05,
             "input_mix_weight_down.weight": torch.randn(CFG.lowrank, CFG.width, generator=g) * 0.08,
             "input_mix_weight_up.weight": torch.randn(CFG.width, CFG.lowrank, generator=g) * 0.08}
    head = torch.randn(CFG.vocab, CFG.hidden, generator=g) * 0.3
    return mt.TargetHead(CFG, mixer, head, model.hot_ids, dtype=torch.float32)


# --- windows ----------------------------------------------------------------------------------------------------------


def test_windows_cover_each_loss_span_with_context_and_split_long_spans():
    req = mt.DumpRequest(Path("."), "r", "ls20", "train", 400, {0: [0, 20, 30, 1, "im_end"], 1: [1, 100, 300, 1, "im_end"],
                                                                 2: [2, 330, 331, 1, "im_end"], 3: [3, 400, -1, 1, ""]},
                         None, 6, "spans", True)
    ws = mt.make_windows(req, max_rows=64, context=6, steps=3)
    first = ws[0]
    assert (first.lo, first.o_lo, first.o_hi, first.hi) == (14, 19, 28, 30)  # 6 rows before the span; t from s-1 to e-2
    long = [w for w in ws if w.span == (100, 300)]
    assert long[0].lo == 94 and long[0].o_lo == 99 and all(w.rows <= 64 for w in long)
    assert [w.o_lo for w in long[1:]] == [w.o_hi + 1 for w in long[:-1]] and long[-1].o_hi == 298
    assert all(w.lo == max(94, w.o_lo - 6) for w in long)  # later pieces keep 6 rows of left context
    assert [w.span for w in ws if w.span[0] in (330, 400)] == [(330, 331)]  # a 2-row span has one origin; open: none
    only = mt.make_windows(mt.DumpRequest(Path("."), "r", "x", "train", 400, req.spans, [1], 6, "spans", True),
                           max_rows=4096)
    assert {w.span for w in only} == {(100, 300)} and only[0].rows == 300 - 94 + 1
    with pytest.raises(ValueError):
        mt.make_windows(req, max_rows=8, context=6)


def test_read_dump_and_build_batch_on_a_real_dump(tmp_path):
    dump, snaps, ids_of = make_dump(tmp_path)
    reqs = mt.read_dump(dump, mt._snapshot_meta([snaps]))
    assert [r.rid for r in reqs] == sorted(ids_of) and {r.split for r in reqs} == {"train", "holdout"}
    r = next(x for x in reqs if x.rid == rid("ls20", 1))
    assert r.loss_spans == [0, 2] and r.context == 16 and r.mm and r.length == len(ids_of[r.rid])
    ws = mt.make_windows(r, context=16)
    assert {w.span for w in ws} == {(s, e) for k, (_, s, e, _, _) in r.spans.items() if k in (0, 2)}
    model = tiny_model()
    batch = mt.build_batch(model, mt.Store(), ws, device="cpu", steps=3, mode="turn")
    ids = np.asarray(ids_of[r.rid])
    assert batch.H.shape == (sum(w.rows for w in ws), CFG.width) and batch.bounds[-1][1] == batch.H.shape[0]
    a, b = batch.bounds[0]
    w = ws[0]
    rows = np.arange(w.lo, w.hi + 1)
    assert batch.pos[a:b].tolist() == rows.tolist()
    data = hd.load_request(dump, r.rid)
    exact = hd.dequantize(data["hc"], data["hc_scale"])[np.searchsorted(data["hc_pos"], w.lo) + np.arange(w.rows)]
    assert torch.allclose(batch.H[a:b], torch.as_tensor(exact), atol=1e-6)  # FP8 rows dequantized as written
    s, e = w.span
    for i, t in enumerate(rows):
        for k in range(1, 4):
            want = w.o_lo <= t <= w.o_hi and t + k + 1 <= e
            assert bool(batch.valid[a + i, k - 1]) == want, (t, k)
            if want:
                assert int(batch.target_row[a + i, k - 1]) == a + i + k and batch.realized[a + i, k - 1] == ids[t + k + 1]
        if t + 3 < len(ids):
            assert batch.chain_ids[a + i].tolist() == [ids[t + 2], ids[t + 3]]
    # turn mode: rows before the span (other than its last header row) were prefilled -> the target's own embedding,
    # image rows -> the dumped vision features; span rows and the row before them -> e(x_{j+1})
    E = model.embed_tokens
    for i, t in enumerate(rows):
        if t >= s - 1:
            assert torch.equal(batch.emb1[a + i], E[min(ids[t + 1], CFG.vocab - 1)])
        elif ids[t] >= hd.MM_PAD_MIN:
            k = int(np.flatnonzero(data["img_pos"] == t)[0])
            assert torch.allclose(batch.emb1[a + i], torch.as_tensor(hd.bf16_value(data["img_embeds"][k])))
        else:
            assert torch.equal(batch.emb1[a + i], E[min(ids[t], CFG.vocab - 1)])
    assert batch.stats["image_rows"] > 0 and not batch.stats.get("image_rows_missing")
    shifted = mt.build_batch(model, mt.Store(), ws[:1], device="cpu", mode="shifted")
    assert torch.equal(shifted.emb1, E[torch.as_tensor(np.minimum(ids[rows + 1], CFG.vocab - 1))])


def test_hot_mask_drops_steps_whose_token_the_draft_cannot_propose(tmp_path):
    dump, snaps, _ = make_dump(tmp_path)
    reqs = mt.read_dump(dump, mt._snapshot_meta([snaps]))
    ws = [w for r in reqs for w in mt.make_windows(r)]
    model = tiny_model(hot_step=2)
    mask = np.zeros(CFG.vocab, bool)
    mask[::2] = True
    full = mt.build_batch(model, mt.Store(), ws, device="cpu")
    hot = mt.build_batch(model, mt.Store(), ws, device="cpu", hot_mask=mask)
    assert hot.valid.sum() < full.valid.sum() and bool((~hot.valid | full.valid).all())
    tokens = hot.realized[hot.valid]
    assert bool((tokens % 2 == 0).all()) and hot.stats["step1_not_hot"] > 0


# --- loss, gradients, optimization ------------------------------------------------------------------------------------


def _batch(tmp_path, model):
    dump, snaps, _ = make_dump(tmp_path)
    reqs = mt.read_dump(dump, mt._snapshot_meta([snaps]))
    ws = [w for r in reqs if r.split == "train" for w in mt.make_windows(r)]
    return mt.build_batch(model, mt.Store(), ws, device="cpu")


def test_frozen_experts_get_no_gradient_and_the_dense_weights_do(tmp_path):
    model = tiny_model()
    target = tiny_target(model)
    batch = _batch(tmp_path, model)
    loss, per_step = mt.chain_loss(model, target, batch, checkpoint=False)
    loss.backward()
    assert all(v is not None and v > 0 for v in per_step)
    for name in model.names:
        p = model.p[mr._key(name)]
        if name in mr.TRAINABLE_NAMES:
            assert p.grad is not None and float(p.grad.abs().sum()) > 0, name
        else:  # the QSA indexer
            assert p.grad is None, name
    assert not any(b.requires_grad or b.grad is not None for b in (model.w_gate, model.w_up, model.w_down,
                                                                   model.embed_tokens, model.lm_head_hot))


def test_activation_checkpointing_gives_the_same_loss_and_gradients(tmp_path):
    model = tiny_model()
    target = tiny_target(model)
    batch = _batch(tmp_path, model)
    grads = []
    for ckpt in (False, True):
        model.zero_grad(set_to_none=True)
        loss, _ = mt.chain_loss(model, target, batch, checkpoint=ckpt, chunk=7)
        loss.backward()
        grads.append((float(loss.detach()), {n: model.p[mr._key(n)].grad.clone() for n in mr.TRAINABLE_NAMES}))
    assert grads[0][0] == pytest.approx(grads[1][0], rel=1e-6)
    for n in mr.TRAINABLE_NAMES:
        assert torch.allclose(grads[0][1][n], grads[1][1][n], rtol=1e-4, atol=1e-7), n


def test_a_few_optimizer_steps_lower_the_chained_kl(tmp_path):
    model = tiny_model()
    target = tiny_target(model)
    batch = _batch(tmp_path, model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    losses = []
    for _ in range(15):
        loss, per_step = mt.chain_loss(model, target, batch, checkpoint=False)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    assert all(b < a for a, b in itertools.pairwise(losses)) and losses[-1] < 0.85 * losses[0], losses
    assert per_step[1] is not None and per_step[2] is not None  # the chain's later steps train too


def test_kl_weights_and_target_temperature():
    model = tiny_model()
    target = tiny_target(model)
    H = torch.randn(4, CFG.width)
    m = torch.randn(4, CFG.hidden)
    kl1 = mt._kl_rows(m, H, model, target, 1.0)
    p = torch.softmax(target.hot_logits(H), -1)
    q = torch.softmax(model.logits(m), -1)
    assert torch.allclose(kl1, (p * (p.log() - q.log())).sum(-1), atol=1e-5) and bool((kl1 >= 0).all())
    assert not torch.allclose(mt._kl_rows(m, H, model, target, 0.7), kl1)
    assert sum(mt.STEP_WEIGHTS) == pytest.approx(1.0) and mt.STEP_WEIGHTS[0] / mt.STEP_WEIGHTS[1] == pytest.approx(
        1 / 0.6, rel=0.02)


def test_schedule_and_processed_probabilities():
    f = mt.lr_lambda(100, 0.05, 0.1)
    assert f(0) == pytest.approx(0.2) and f(4) == 1.0 and f(99) == pytest.approx(0.1, abs=1e-3)
    assert all(f(i) >= f(i + 1) for i in range(5, 99))
    logits = torch.randn(3, 500) * 3
    p = mt.processed_probs(logits, 0.7, 20, 0.95)
    assert torch.allclose(p.sum(-1), torch.ones(3)) and int((p > 0).sum(-1).max()) <= 20
    top = torch.softmax(logits / 0.7, -1).argmax(-1)
    assert torch.equal(p.argmax(-1), top)
    sharp = torch.zeros(1, 50)
    sharp[0, 3] = 30.0
    assert mt.processed_probs(sharp)[0, 3] == 1.0  # top-p keeps the mode alone when it holds >= 95%


# --- evaluation and the full run --------------------------------------------------------------------------------------


def test_evaluation_reports_per_step_metrics_and_accept_proxies(tmp_path):
    model = tiny_model()
    target = tiny_target(model)
    batch = _batch(tmp_path, model)
    rep = mt.evaluate(model, target, [batch])
    assert set(rep["steps"]) == {1, 2, 3}
    s1 = rep["steps"][1]
    assert s1["origins"] == float(batch.valid[:, 0].sum()) and 0 <= s1["top1_realized"] <= 1
    for key in ("accept_realized", "accept_greedy", "accept_expected"):
        assert 1.0 <= rep[key] <= 4.0, key
    assert rep["accept_realized_per_origin"] == pytest.approx(rep["accept_realized"], abs=0.25)
    same = mt.compare_reports(rep, mt.evaluate(model, target, [batch]))
    assert all(v["delta"] == pytest.approx(0, abs=1e-9) for v in same.values())


def _train_args(tmp_path, dump, snaps, out, *extra):
    draft = tmp_path / "draft"
    if not draft.exists():
        mf.write_tiny_draft(draft, seed=5)
        mf.write_target_mixer(tmp_path / "mixer.safetensors", mf.TINY_TEXT)
        torch.save(list(range(0, CFG.vocab)), tmp_path / "map.pt")
    return ["train", "--dump", str(dump), "--snapshots", str(snaps), "--draft", str(draft), "--target-mixer",
            str(tmp_path / "mixer.safetensors"), "--token-map", str(tmp_path / "map.pt"), "--out", str(out),
            "--fp32", "--batch-rows", "120", "--max-rows", "64", "--lr", "2e-3", "--log-every", "1000",
            "--eval-rows", "0", *extra]


def test_train_end_to_end_resume_and_write_the_draft(tmp_path, capsys):
    dump, snaps, _ = make_dump(tmp_path)
    out = tmp_path / "run"
    assert mt.main(_train_args(tmp_path, dump, snaps, out, "--max-steps", "4", "--checkpoint-every", "2")) == 0
    report = json.loads((out / "train-report.json").read_text())
    assert report["steps_done"] == 4 and report["windows"]["holdout"]["games"] == {"sb26": 2}
    assert set(report["windows"]["train"]["games"]) == {"ls20"}
    assert "eval_original" in report and "eval_trained" in report and "accept_expected" in report["comparison"]
    log = [json.loads(x) for x in (out / "train-log.jsonl").read_text().splitlines()]
    assert [x["step"] for x in log] == [1, 2, 3, 4] and all(math.isfinite(x["loss"]) for x in log)
    trained = mr.load_dense_file(out / "trained-dense.safetensors")
    assert sorted(trained) == sorted(mr.TRAINABLE_NAMES) and all(t.dtype == torch.bfloat16 for t in trained.values())
    original = mr.load_draft_dir(tmp_path / "draft", experts=False, embed=False, lm_head=False).dense
    assert any(not torch.equal(trained[n], original[n]) for n in trained)
    # resume: two more steps continue from the checkpoint (same schedule length)
    state = torch.load(out / "ckpt.pt", weights_only=False)
    assert state["step"] == 4
    assert mt.main(_train_args(tmp_path, dump, snaps, out, "--max-steps", "6", "--resume")) == 0
    assert "resumed at step 4" in capsys.readouterr().out
    assert json.loads((out / "train-report.json").read_text())["steps_done"] == 6
    # the hand-off: the trained file makes a draft directory the launcher accepts, frozen tensors untouched
    manifest = wd.write_draft(tmp_path / "draft", out / "trained-dense.safetensors", tmp_path / "new-draft")
    check = wd.check_draft_dir(tmp_path / "new-draft", reference=tmp_path / "draft",
                               trained_names=manifest["replaced"])
    assert check["ok"], check["problems"]
    assert set(check["changed"]) <= set(mr.TRAINABLE_NAMES) and check["changed"]
    reloaded = mr.load_draft_dir(tmp_path / "new-draft")
    trained = mr.load_dense_file(out / "trained-dense.safetensors")  # after the resumed steps
    for name in mr.TRAINABLE_NAMES:
        assert torch.equal(reloaded.dense[name], trained[name]), name


def test_eval_and_plan_commands(tmp_path, capsys):
    dump, snaps, _ = make_dump(tmp_path, dtype="bf16")
    out = tmp_path / "run"
    mt.main(_train_args(tmp_path, dump, snaps, out, "--max-steps", "2"))
    capsys.readouterr()
    assert mt.main(["plan", "--dump", str(dump), "--snapshots", str(snaps), "--max-rows", "64",
                    "--batch-rows", "120"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["train"]["windows"] > 0 and info["holdout"]["rows"] > 0 and info["total_steps"] >= 1
    assert set(info["memory_gb"]) >= {"experts_bf16", "trainable_fp32_adam_grads"}
    assert mt.main(["eval", "--dump", str(dump), "--snapshots", str(snaps), "--draft", str(tmp_path / "draft"),
                    "--target-mixer", str(tmp_path / "mixer.safetensors"), "--token-map", str(tmp_path / "map.pt"),
                    "--out", str(tmp_path / "ev"), "--fp32", "--max-rows", "64", "--batch-rows", "120",
                    "--trained", str(out / "trained-dense.safetensors")]) == 0
    result = json.loads((tmp_path / "ev" / "eval-report.json").read_text())
    assert result["split"] == "holdout" and set(result["comparison"]) >= {"accept_realized", "step1_kl"}
    assert mt.estimate_memory(16384)["experts_bf16"] == pytest.approx(5.03, abs=0.01)
