"""scripts/sglang_hc_dump_patch.py: the hyper-connection dump for the MTP draft fine-tune (the span rule, the ring
buffer of context rows, the FP8 format, the files, the SGLang hooks) and the anchored edit of the installed sglang,
alone and with the REAP patch. CPU only: numpy everywhere, torch where installed (else those checks skip), the
Pennyroyal wheel where available (PENNYROYAL_WHEEL; else those checks skip)."""
from __future__ import annotations

import ast
import difflib
import hashlib
import itertools
import json
import os
import sys
import textwrap
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import sglang_hc_dump_patch as hd  # noqa: E402
import sglang_reap_patch as rp  # noqa: E402

KEPT = ROOT / "kaggle" / "franzen" / "reap448_kept_experts.json"
WHEEL = Path(os.environ.get("PENNYROYAL_WHEEL", "/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/"
                            "d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/reap/dl-wheel/"
                            "sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl"))
needs_wheel = pytest.mark.skipif(not WHEEL.is_file(), reason="Pennyroyal sglang wheel not available (PENNYROYAL_WHEEL)")
S, E, A, NL, T = hd.IM_START, hd.IM_END, hd.ASSISTANT, hd.NEWLINE, hd.THINK
USER, NLNL, PAD = 846, 271, 1_000_000 + 12345  # 'user', '\n\n', an image row's pad value
C = 12  # context rows in the dumps below
BASE = {key: ("pristine" if "+" not in value else "reap") for key, value in hd.BASE_SHA256.items()}
DIRS = itertools.count()


def turn(body, think=True, nl=True):
    """A past assistant turn as the template renders it (body = reasoning + content tokens)."""
    return [S, A, NL] + ([T] + ([NL] if nl else []) if think else []) + list(body) + [E, NL]


def user(body):
    return [S, USER, NL] + list(body) + [E, NL]


GEN_PROMPT = [S, A, NL, T, NL]


def merged_spans(pieces):
    """Spans over several scan_spans calls, merged by index as a reader of the index does."""
    out = {}
    for spans in pieces:
        for s in spans:
            if s.index not in out or s.end >= 0:
                out[s.index] = s.as_list()
    return [out[k] for k in sorted(out)]


def scan_in_pieces(ids, cuts):
    state, pieces, bounds = hd.ScanState(), [], [0, *cuts, len(ids)]
    for lo, hi in itertools.pairwise(bounds):
        spans, state = hd.scan_spans(np.array(ids[lo:hi], np.int64), lo, state)
        pieces.append(spans)
    return merged_spans(pieces), state


# --- the span rule --------------------------------------------------------------------------------------------------


def test_spans_of_a_rendered_conversation():
    ids = [S, 8678, NL, 1, 2, E, NL] + user([3, 4]) + turn([10, 11, 12]) + user([5]) + turn([13]) + GEN_PROMPT
    spans, state = hd.scan_spans(np.array(ids), 0, hd.ScanState())
    first, second = ids.index(10), ids.index(13)
    assert [s.as_list() for s in spans] == [[0, first, first + 3, 1, "im_end"],     # 10 11 12 <|im_end|>
                                            [1, second, second + 1, 1, "im_end"],  # 13 <|im_end|>
                                            [2, len(ids), -1, 1, ""]]               # the generation prompt
    assert ids[first - 5:first] == ids[second - 5:second] == [S, A, NL, T, NL] and ids[first + 3] == E
    assert state.phase == hd.SPAN and state.opened == 3 and state.span_start == len(ids)


@pytest.mark.parametrize("variant", ["plain", "empty reasoning", "no think", "nested", "unterminated", "interrupted"])
def test_scan_is_the_same_for_every_chunk_split(variant):
    ids = user([1]) + turn([10, 11]) + user([PAD, PAD, 2]) + turn([12, 13, 14])
    ids += {"plain": [], "empty reasoning": [S, A, NL, T, NLNL, 248069, NLNL, 15, E, NL],
            "no think": turn([16, 17], think=False), "nested": [S, A, NL, T, NL, 18, S, A, NL, T, NL, 19, E, NL],
            "unterminated": [S, A, NL, T, NL, 20, 21], "interrupted": [S, A, 5, T, NL, 22, E, NL, S, A, NL, 23]}[variant]
    whole, state = scan_in_pieces(ids, [])
    for cut in range(len(ids) + 1):
        assert scan_in_pieces(ids, [cut]) == (whole, state), cut
    rng = np.random.default_rng(len(variant))
    for _ in range(40):
        cuts = sorted(set(rng.integers(0, len(ids) + 1, size=int(rng.integers(2, 8))).tolist()))
        assert scan_in_pieces(ids, cuts) == (whole, state), cuts


def test_header_variants():
    # empty reasoning renders <think>\n\n</think> and '\n\n' is one token: the span starts at it
    spans, _ = hd.scan_spans(np.array([S, A, NL, T, NLNL, 248069, NLNL, 7, E]), 0, hd.ScanState())
    assert [s.as_list() for s in spans] == [[0, 4, 8, 1, "im_end"]]
    # a turn rendered without <think> (preserve_thinking false) starts after 'assistant \n'
    spans, _ = hd.scan_spans(np.array([S, A, NL, 7, 8, E]), 0, hd.ScanState())
    assert [s.as_list() for s in spans] == [[0, 3, 5, 0, "im_end"]]
    # other roles, a role token outside a header, a broken header, a stray <|im_end|>: no span
    for ids in ([S, USER, NL, T, NL, 1, E], [A, NL, T, NL, 1, E], [S, A, 9, T, NL, 1, E], [E, E, 1]):
        spans, state = hd.scan_spans(np.array(ids), 0, hd.ScanState())
        assert spans == [] and state.opened == 0, ids
    # an empty assistant message: the span is just its <|im_end|>
    spans, _ = hd.scan_spans(np.array([S, A, NL, E, NL]), 0, hd.ScanState())
    assert [s.as_list() for s in spans] == [[0, 3, 3, 0, "im_end"]]


def test_nested_and_unterminated_spans():
    ids = [S, A, NL, T, NL, 10, 11, S, A, NL, T, NL, 12, E, NL, S, A, NL, T, NL, S]
    spans, state = hd.scan_spans(np.array(ids), 0, hd.ScanState())
    assert [s.as_list() for s in spans] == [
        [0, 5, 6, 1, "im_start"],    # a header inside a span ends it at the row before
        [1, 12, 13, 1, "im_end"],    # ... and opens the next one
        [2, 20, 19, 1, "im_start"],  # opened and closed at once: an empty span (end = start - 1)
    ]
    assert state.phase == hd.START and state.opened == 3
    # unterminated: the span stays open (end -1) and the next chunk continues it
    spans, state = hd.scan_spans(np.array([S, A, NL, T, NL, 10, 11]), 100, hd.ScanState())
    assert [s.as_list() for s in spans] == [[0, 105, -1, 1, ""]]
    spans, state = hd.scan_spans(np.array([12, E, NL]), 107, state)
    assert [s.as_list() for s in spans] == [[0, 105, 108, 1, "im_end"]] and state.phase == hd.OUT
    # the scanner never changes the state it was given
    before = hd.ScanState(phase=hd.SPAN, span_start=3, opened=1)
    hd.scan_spans(np.array([E]), 5, before)
    assert before == hd.ScanState(phase=hd.SPAN, span_start=3, opened=1)


# --- row selection and the ring buffer -------------------------------------------------------------------------------


def test_select_rows_marks_span_rows_and_the_context_before_each_span():
    spans = [hd.Span(0, 12, 15, True, "im_end"), hd.Span(1, 22, -1, True, "")]
    role, before = hd.select_rows(spans, 10, 15, 4)  # rows 10..24
    assert role.tolist() == [0, 0, 1, 1, 1, 1, -1, -1, 0, 0, 0, 0, 1, 1, 1]
    assert before == [(8, 10)]  # span 0's context 8..11: rows 8 and 9 are in an earlier chunk
    role, before = hd.select_rows(spans, 10, 15, 4, loss_spans=frozenset({1}))
    assert role.tolist() == [-1] * 8 + [0, 0, 0, 0, 1, 1, 1] and before == []
    role, before = hd.select_rows(spans, 10, 15, 4, keep_all=True)
    assert role.tolist() == [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1] and before == []
    # a span continuing from an earlier chunk gets no context here; a pending one (starts past the chunk) neither
    role, before = hd.select_rows([hd.Span(0, 3, 12, True, "im_end"), hd.Span(1, 25, -1, True, "")], 10, 15, 4)
    assert role.tolist() == [1, 1, 1] + [-1] * 12 and before == []


def stream():
    """A request as the server sees it: system, user turns ending in image rows, assistant turns of several shapes
    (the third has empty reasoning), the generation prompt."""
    ids = [S, 8678, NL] + list(range(20, 32)) + [E, NL]
    ids += user([40, 41] + [PAD] * 6)
    ids += turn(range(50, 64))
    ids += user([42] + [PAD + 1] * 3)
    ids += turn([65, 66])
    ids += user([43, 44])
    ids += [S, A, NL, T, NLNL, 248069, NLNL, 67, E, NL]  # empty reasoning
    ids += user([PAD + 2] * 4)
    ids += turn(range(70, 90))
    return ids + GEN_PROMPT


def dump(tmp_path, ids, cuts, *, context=C, keep="spans", dtype="fp8", loss_spans=None, rid="req-1", seed=0,
         mm=True):
    rng = np.random.default_rng(seed)
    n = len(ids)
    hc = hd.bf16_value(hd.bf16_bits(rng.standard_normal((n, 32)).astype(np.float32) * np.exp(rng.uniform(-3, 3, (n, 1)))))
    embeds = rng.standard_normal((n, 6)).astype(np.float32)
    mrope = np.stack([np.arange(n), np.arange(n) * 2, np.arange(n) * 3])
    out = tmp_path / f"dump{next(DIRS)}"
    if loss_spans is not None:
        (out / "plans").mkdir(parents=True)
        (out / "plans" / f"{rid}.json").write_text(json.dumps({"loss_spans": loss_spans}))
    dumper = hd.Dumper(out, context=context, keep=keep, dtype=dtype)
    bounds = [0, *cuts, n]
    records = []
    for lo, hi in itertools.pairwise(bounds):
        records.append(dumper.process_chunk(rid, lo, np.array(ids[lo:hi]), hc[lo:hi], positions=np.arange(lo, hi),
                                            mrope=mrope[:, lo:hi], mm_embeds=embeds[lo:hi] if mm else None))
    return out, hc, embeds, records


def test_a_dump_does_not_depend_on_how_the_prompt_was_chunked(tmp_path):
    ids = stream()
    ref_dir, hc, embeds, _ = dump(tmp_path, ids, [])
    ref = hd.load_request(ref_dir, "req-1")
    assert len(set(ref["hc_pos"].tolist())) == len(ref["hc_pos"])  # every row once
    rng = np.random.default_rng(3)
    cut_sets = [list(range(k, len(ids), k)) for k in (1, 2, 3, 4, 5, 7, 16)]
    cut_sets += [sorted(set(rng.integers(1, len(ids), size=int(rng.integers(1, 12))).tolist())) for _ in range(25)]
    for cuts in cut_sets:
        out, _, _, records = dump(tmp_path, ids, cuts)
        got = hd.load_request(out, "req-1")
        assert len(set(got["hc_pos"].tolist())) == len(got["hc_pos"]), cuts
        for key in ("token_ids", "positions", "mrope_positions", "hc_pos", "hc_role", "hc", "hc_scale", "img_pos",
                    "img_embeds"):
            assert np.array_equal(got[key], ref[key]), (key, cuts)
        assert got["spans"] == ref["spans"] and len(records) == len(cuts) + 1 and all(records)
        assert sum(r["n"] for r in records) == len(ids) and records[-1]["spans_opened_total"] == 5
    # what was kept: every span row, the C rows before each span, nothing else; values within FP8 error
    spans = ref["spans"]
    assert [(s[2] if s[2] >= 0 else len(ids) - 1) - s[1] + 1 for s in spans.values()] == [15, 3, 5, 21, 0]
    assert spans[4] == [4, len(ids), -1, 1, ""]  # the generation prompt: no rows, so no context either
    want = {}
    for _, start, end, _, _ in list(spans.values())[:4]:
        for p in range(max(0, start - C), start):
            want.setdefault(p, 0)
        for p in range(start, end + 1):
            want[p] = 1
    assert dict(zip(ref["hc_pos"].tolist(), ref["hc_role"].tolist())) == want
    rows = hd.dequantize(ref["hc"], ref["hc_scale"])
    exact = hc[ref["hc_pos"]]
    assert np.all(np.abs(rows - exact) <= np.abs(exact) * 2.0 ** -4 + np.abs(exact).max(1, keepdims=True) * 2.0 ** -18)
    # image rows inside a kept range carry the input embedding the target consumed (BF16); token ids keep pad values
    images = [p for p in want if ids[p] >= hd.MM_PAD_MIN]
    assert ref["img_pos"].tolist() == sorted(images) and len(images) == 5 + 3 + 4
    assert np.array_equal(hd.bf16_value(ref["img_embeds"]), hd.bf16_value(hd.bf16_bits(embeds[sorted(images)])))
    assert ref["token_ids"].tolist() == ids and ref["positions"].tolist() == list(range(len(ids)))
    assert ref["mrope_positions"][2].tolist() == [3 * p for p in range(len(ids))]


def test_context_rows_older_than_one_small_chunk_come_from_the_ring(tmp_path):
    ids = user(list(range(100, 130))) + turn([1, 2, 3])
    out, _, _, records = dump(tmp_path, ids, list(range(1, len(ids))))  # one token per chunk
    got = hd.load_request(out, "req-1")
    first = ids.index(1)  # the span's first row; the header's last token opened it one chunk earlier
    assert got["hc_pos"].tolist() == list(range(first - C, first + 4)) and got["hc_role"].tolist() == [0] * C + [1] * 4
    carriers = [r for r in records if r["carried"]]
    assert len(carriers) == 1 and carriers[0]["start"] == first and carriers[0]["carried"] == C
    assert all(r["kept"] == 0 for r in records if r["start"] < first)
    assert records[first - 1]["spans"] == [[0, first, -1, 1, ""]]  # opened at the header's end: no rows yet


def test_a_plan_keeps_only_its_loss_spans_and_their_context(tmp_path):
    ids = stream()
    out, _, _, records = dump(tmp_path, ids, [9, 40, 77], loss_spans=[3])
    got = hd.load_request(out, "req-1")
    _, start, end, _, _ = got["spans"][3]
    assert got["hc_pos"].tolist() == list(range(start - C, end + 1))
    assert got["hc_role"].tolist() == [0] * C + [1] * (end - start + 1)
    assert records[0]["plan"] == "req-1.json" and records[-1]["loss_spans"] == [3]
    assert len(got["spans"]) == 5  # every span is still reported


def test_keep_all_and_bf16_rows(tmp_path):
    ids = stream()
    out, hc, _, _ = dump(tmp_path, ids, [17, 50], keep="all", dtype="bf16")
    got = hd.load_request(out, "req-1")
    assert got["hc_pos"].tolist() == list(range(len(ids))) and got["hc_scale"] is None and got["dtype"] == "bf16"
    assert np.array_equal(hd.dequantize(got["hc"]), hc)  # raw BF16 rows, bit-exact
    assert int(got["hc_role"].sum()) == 15 + 3 + 5 + 21


def test_text_only_requests_have_no_image_embeddings(tmp_path):
    ids = stream()
    out, _, _, records = dump(tmp_path, ids, [30], mm=False)
    got = hd.load_request(out, "req-1")
    assert got["img_pos"] is None and got["img_embeds"] is None
    assert records[0]["images"] > 0 and records[0]["images_kept"] == 0 and records[0]["mm_embeds"] is False


# --- number formats and files ------------------------------------------------------------------------------------------


def test_fp8_rows_round_trip_within_their_error_bound():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((64, 10240)).astype(np.float32) * np.exp(rng.uniform(-8, 8, (64, 1))).astype(np.float32)
    x[:, :4] *= 300  # outlier channels, as residual streams have
    x[3] = 0.0
    x = hd.bf16_value(hd.bf16_bits(x))
    q = hd.quantize_rows(x)
    assert q["codes"].dtype == np.uint8 and q["codes"].shape == x.shape and q["scale"].shape == (64,)
    back = hd.dequantize(q["codes"], q["scale"])
    scale = hd.bf16_value(q["scale"])[:, None]
    assert np.all(np.abs(back - x) <= np.abs(x) * 2.0 ** -4 + scale * 2.0 ** -10 + 1e-30)  # half an FP8 step
    assert np.all(np.isfinite(back)) and np.all(back[3] == 0) and hd.bf16_value(q["scale"][3]) == 1.0
    assert np.all(np.abs(back).max(1) <= 448 * scale[:, 0])
    assert 0.01 < q["qerr_mean"] < 0.035 and q["qerr_max"] < 0.04 and q["nonfinite"] == 0
    assert hd.quantize_rows(x[:0])["codes"].shape == (0, 10240)


def test_bf16_and_fp8_codecs_on_known_values():
    one = np.float32(1.0)
    cases = {one: 0x3F80, one + np.float32(2.0 ** -8): 0x3F80, one + np.float32(3 * 2.0 ** -8): 0x3F82,
             np.float32(-2.0): 0xC000}
    for value, bits in cases.items():
        assert int(hd.bf16_bits(np.array([value]))[0]) == bits  # ties to even
    assert int(hd.bf16_bits(np.array([np.nan], np.float32))[0]) == 0x7FC0
    values = np.array([0, 2.0 ** -9, 2.0 ** -10, 1.5 * 2.0 ** -9, 1.0625, 1.1875, 448, 460, -0.0, 1e6], np.float32)
    assert hd.fp8_bits(values).tolist() == [0x00, 0x01, 0x00, 0x02, 0x38, 0x3A, 0x7E, 0x7E, 0x80, 0x7E]
    assert hd.fp8_value(np.array([0x7E, 0x08, 0x07, 0xFE])).tolist() == [448.0, 2.0 ** -6, 7 * 2.0 ** -9, -448.0]
    assert np.isnan(hd.fp8_value(np.array([0x7F]))[0])


def test_the_numpy_codecs_match_torch_bit_for_bit():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    finite = hd.FP8_VALUES[np.isfinite(hd.FP8_VALUES)]
    pos = np.unique(np.abs(finite))
    mids = (pos[1:] + pos[:-1]) / 2
    x = np.concatenate([finite, mids, -mids, np.nextafter(mids, 0), np.nextafter(mids, 1e9),
                        (rng.standard_normal(200_000) * np.exp(rng.uniform(-14, 7, 200_000)))]).astype(np.float32)
    x = np.clip(x, -448, 448)  # the dump clamps before converting (torch releases differ beyond 448)
    assert np.array_equal(hd.fp8_bits(x), torch.from_numpy(x).to(torch.float8_e4m3fn).view(torch.uint8).numpy())
    y = (rng.standard_normal(200_000) * np.exp(rng.uniform(-30, 30, 200_000))).astype(np.float32)
    assert np.array_equal(hd.bf16_bits(y), torch.from_numpy(y).to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16))
    rows = hd.bf16_value(hd.bf16_bits(rng.standard_normal((70, 10240)).astype(np.float32) * 5))
    rows[2] = 0
    a, b = hd.quantize_rows(rows), hd.quantize_rows(torch.from_numpy(rows).to(torch.bfloat16))
    assert np.array_equal(a["codes"], b["codes"]) and np.array_equal(a["scale"], b["scale"])
    assert abs(a["qerr_max"] - b["qerr_max"]) < 1e-5


def test_chunk_files_are_self_describing_safetensors(tmp_path):
    out, _, _, records = dump(tmp_path, stream(), [33])
    arrays, meta = hd.read_safetensors(out / records[1]["file"])
    info = json.loads(meta["arc3_hc_dump"])
    assert info["format"] == hd.FORMAT and info["version"] == hd.VERSION and info["tensors"] == hd.TENSORS
    assert {k: info[k] for k in ("rid", "start", "n", "kept")} == {k: records[1][k] for k in ("rid", "start", "n", "kept")}
    assert set(arrays) == {"token_ids", "positions", "mrope_positions", "hc_pos", "hc_role", "hc", "hc_scale",
                           "img_pos", "img_embeds"}
    assert arrays["hc"].shape == (records[1]["kept"], 32) and arrays["mrope_positions"].shape == (3, records[1]["n"])
    raw = (out / records[1]["file"]).read_bytes()
    size = int.from_bytes(raw[:8], "little")
    header = json.loads(raw[8:8 + size])
    assert header["hc"]["dtype"] == "F8_E4M3" and header["hc_scale"]["dtype"] == "BF16" and size % 8 == 0
    assert records[1]["bytes"] == len(raw)
    fmt = json.loads((out / "format.json").read_text())
    assert fmt["tokens"]["im_start"] == 248045 and fmt["settings"]["context"] == C and "span_rule" in fmt
    index = hd.read_index(out)
    assert [r["file"] for r in index] == [r["file"] for r in records] and index[-1]["bytes_total"] == sum(
        r["bytes"] for r in records)


def test_the_safetensors_library_reads_the_files(tmp_path):
    torch = pytest.importorskip("torch")
    st = pytest.importorskip("safetensors.torch")
    out, hc, _, records = dump(tmp_path, stream(), [33])
    tensors = st.load_file(str(out / records[1]["file"]))
    assert tensors["hc"].dtype == torch.float8_e4m3fn and tensors["hc_scale"].dtype == torch.bfloat16
    rows = tensors["hc"].float() * tensors["hc_scale"].float()[:, None]
    exact = torch.from_numpy(hc[tensors["hc_pos"].long().numpy()])
    assert torch.all((rows - exact).abs() <= exact.abs() / 16 + exact.abs().amax(1, keepdim=True) * 2.0 ** -18)


def test_the_size_cap_stops_the_dump_and_errors_stay_inside_it(tmp_path):
    ids = stream()
    rng = np.random.default_rng(0)
    hc = rng.standard_normal((len(ids), 32)).astype(np.float32)
    first = hd.Dumper(tmp_path / "probe", context=5).process_chunk("a", 0, np.array(ids[:40]), hc[:40])["bytes"]
    dumper = hd.Dumper(tmp_path / "cap", context=5, max_bytes=first + 100)
    assert dumper.process_chunk("a", 0, np.array(ids[:40]), hc[:40])["bytes"] == first
    assert dumper.process_chunk("a", 40, np.array(ids[40:]), hc[40:]) is None and dumper.stopped
    assert dumper.process_chunk("b", 0, np.array(ids), hc) is None
    events = [r for r in hd.read_index(tmp_path / "cap") if "event" in r]
    assert [e["event"] for e in events] == ["stopped"] and len(list((tmp_path / "cap").rglob("*.safetensors"))) == 1
    dumper = hd.Dumper(tmp_path / "err", context=5)
    assert dumper.process_chunk("bad", 0, np.array(ids[:10]), hc[:9]) is None  # 9 rows for 10 ids: logged, not raised
    assert dumper.process_chunk("bad", 10, np.array(ids[10:20]), hc[10:20]) is None  # the request stays skipped
    assert dumper.process_chunk("good", 0, np.array(ids), hc) is not None
    index = hd.read_index(tmp_path / "err")
    assert index[0]["event"] == "error" and "9 hidden rows for 10 tokens" in index[0]["error"]
    with pytest.raises(ValueError, match="bad settings"):
        hd.Dumper(tmp_path, keep="some")


def test_a_chunk_out_of_order_starts_a_new_attempt(tmp_path):
    ids = stream()
    hc = np.random.default_rng(1).standard_normal((len(ids), 32)).astype(np.float32)
    dumper = hd.Dumper(tmp_path, context=5)
    dumper.process_chunk("r", 0, np.array(ids[:50]), hc[:50])
    dumper.process_chunk("r", 0, np.array(ids[:50]), hc[:50])  # re-prefilled from the start (a retraction)
    dumper.process_chunk("r", 50, np.array(ids[50:]), hc[50:])
    records = hd.read_index(tmp_path)
    assert [r["attempt"] for r in records] == [0, 1, 1] and records[1]["file"].endswith("-a1.safetensors")
    got = hd.load_request(tmp_path, "r")
    assert [c["attempt"] for c in got["chunks"]] == [1, 1] and got["token_ids"].tolist() == ids
    assert hd.safe_rid("hc-ab12-ls20-p0-r0005") == "hc-ab12-ls20-p0-r0005"
    assert hd.safe_rid("../x y") != "../x y" and "/" not in hd.safe_rid("../x y") and hd.safe_rid("..") != ".."


# --- the hooks SGLang calls ----------------------------------------------------------------------------------------------


class FakeBatch(types.SimpleNamespace):
    """The ForwardBatch fields capture reads."""


def _batch(mode, reqs, *, numpy_only=True, mm=True, cpu_lens=True):
    ids = np.concatenate([np.array(r[2], np.int64) for r in reqs])
    total = len(ids)
    rng = np.random.default_rng(total)
    fb = FakeBatch(forward_mode=types.SimpleNamespace(name=mode), rids=[r[0] for r in reqs], input_ids=ids,
                   positions=np.concatenate([np.arange(r[1], r[1] + len(r[2])) for r in reqs]),
                   mrope_positions=np.tile(np.arange(total), (3, 1)),
                   mm_input_embeds=rng.standard_normal((total, 8)).astype(np.float32) if mm else None)
    starts, lens = [r[1] for r in reqs], [len(r[2]) for r in reqs]
    if cpu_lens:
        fb.extend_prefix_lens_cpu, fb.extend_seq_lens_cpu = starts, lens
    else:  # the gpu_only path: device tensors, no *_cpu mirrors
        fb.extend_prefix_lens_cpu = fb.extend_seq_lens_cpu = None
        fb.extend_prefix_lens, fb.extend_seq_lens = np.array(starts), np.array(lens)
    return fb, rng.standard_normal((total + 3, 16)).astype(np.float32)  # padded rows after the real ones


MODEL = types.SimpleNamespace(config=types.SimpleNamespace(hc_count=2, hidden_size=8, vocab_size=248320))


def _clamp(ids):
    ids[ids >= 248320] = 248319  # what embed_mm_inputs does to the batch's input ids, in place


@pytest.mark.parametrize("cpu_lens", [True, False])
def test_capture_dumps_every_request_of_an_extend_batch(tmp_path, monkeypatch, cpu_lens):
    monkeypatch.setenv(hd.ENV, str(tmp_path))
    monkeypatch.setenv(hd.ENV_CONTEXT, "3")
    monkeypatch.setattr(hd, "_DUMPER", None)
    a = user([PAD, PAD, 7]) + turn([8, 9])
    b = [5, 6] + turn([10])
    fb, hc = _batch("EXTEND", [("ra", 0, a), ("rb", 40, b)], cpu_lens=cpu_lens)
    args = (fb.input_ids, fb.positions, fb)
    before = hd.input_ids_before(args, {})
    _clamp(fb.input_ids)
    hd.capture(MODEL, args, {}, hc, before)
    ra, rb = hd.load_request(tmp_path, "ra"), hd.load_request(tmp_path, "rb")
    assert ra["token_ids"].tolist() == a and rb["token_ids"].tolist() == b  # pad values, not the clamped ids
    assert rb["positions"].tolist() == list(range(40, 40 + len(b))) and rb["chunks"][0]["missing_prefix"] == 40
    span = ra["spans"][0]
    assert ra["hc_pos"].tolist() == list(range(span[1] - 3, span[2] + 1))
    rows = hd.dequantize(ra["hc"], ra["hc_scale"])
    assert np.allclose(rows, hc[:len(a)][ra["hc_pos"]], rtol=2.0 ** -4, atol=1e-3)
    off = len(a)
    assert np.allclose(hd.dequantize(rb["hc"], rb["hc_scale"]), hc[off + rb["hc_pos"] - 40], rtol=2.0 ** -4, atol=1e-3)
    assert ra["chunks"][0]["mm_embeds"] is True and ra["img_pos"] is None  # the image rows are outside the kept range


def test_capture_ignores_decode_verify_and_odd_batches(tmp_path, monkeypatch):
    monkeypatch.setenv(hd.ENV, str(tmp_path))
    monkeypatch.setattr(hd, "_DUMPER", None)
    monkeypatch.setattr(hd, "_WARNED", set())
    for mode in ("DECODE", "TARGET_VERIFY", "IDLE", "DRAFT_EXTEND_V2"):
        fb, hc = _batch(mode, [("r", 0, turn([1]))])
        args = (fb.input_ids, fb.positions, fb)
        assert hd.input_ids_before(args, {}) is None
        hd.capture(MODEL, args, {}, hc, None)
    fb, hc = _batch("MIXED", [("m", 0, turn([1]))])
    args = (fb.input_ids, fb.positions, fb)
    hd.capture(types.SimpleNamespace(config=types.SimpleNamespace(hc_count=3, hidden_size=8)), args, {}, hc,
               hd.input_ids_before(args, {}))  # 16-wide rows, the model says 24: skipped with a warning
    fb.rids = None
    hd.capture(MODEL, args, {}, hc, hd.input_ids_before(args, {}))  # no request ids: skipped
    assert hd.read_index(tmp_path) == [] and {"width", "shape"} <= hd._WARNED
    hd.capture(MODEL, args, {}, hc, None)  # nothing kept before the forward: nothing to do
    fb.rids = ["m"]
    hd.capture(MODEL, (), {"forward_batch": fb}, hc, hd.input_ids_before((), {"forward_batch": fb}))
    assert [r["rid"] for r in hd.read_index(tmp_path)] == ["m"]


def test_capture_with_torch_tensors_writes_what_numpy_writes(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    ids = user([PAD, 7]) + turn(range(30, 45))
    files = {}
    for kind in ("numpy", "torch"):
        out = tmp_path / kind
        monkeypatch.setenv(hd.ENV, str(out))
        monkeypatch.setattr(hd, "_DUMPER", None)
        fb, hc = _batch("EXTEND", [("r", 0, ids)])
        hc = hd.bf16_value(hd.bf16_bits(hc))
        if kind == "torch":
            fb.input_ids, fb.positions = torch.from_numpy(fb.input_ids), torch.from_numpy(fb.positions)
            fb.mrope_positions = torch.from_numpy(fb.mrope_positions)
            fb.mm_input_embeds = torch.from_numpy(fb.mm_input_embeds).to(torch.bfloat16)
            hc = torch.from_numpy(hc).to(torch.bfloat16)
        else:
            fb.mm_input_embeds = hd.bf16_value(hd.bf16_bits(fb.mm_input_embeds))
        args = (fb.input_ids, fb.positions, fb)
        before = hd.input_ids_before(args, {})
        hd.capture(MODEL, args, {}, hc, before)
        files[kind] = hd.load_request(out, "r")
    for key in ("token_ids", "positions", "mrope_positions", "hc_pos", "hc_role", "hc", "hc_scale", "img_pos",
                "img_embeds"):
        assert np.array_equal(files["numpy"][key], files["torch"][key]), key


def test_input_ids_are_copied_before_the_forward(tmp_path):
    fb, _ = _batch("EXTEND", [("r", 0, [PAD, 5, 6])])
    before = hd.input_ids_before((fb.input_ids, fb.positions, fb), {})
    _clamp(fb.input_ids)
    assert before.tolist() == [PAD, 5, 6] and fb.input_ids.tolist() == [248319, 5, 6]
    assert hd.input_ids_before((None, fb.positions, fb), {}).tolist() == [248319, 5, 6]  # mm routine passes None


# --- the installer ---------------------------------------------------------------------------------------------------------

ORIGINAL_FORWARD = ("    @torch.no_grad()\n" + hd.FORWARD_ANCHOR
                    + "        if hc_hidden_states is not None and isinstance(output, LogitsProcessorOutput):\n"
                      "            output.hidden_states = hc_hidden_states\n"
                      "        return output\n")


def _fake_model_file() -> str:
    """The parts of qwen4_exp.py both patches anchor on (verbatim from the wheel's file)."""
    return ("class Qwen4ExpVLModel(Qwen4ExpModel):\n    pass\n\n\n" + hd.CLASS_ANCHOR
            + "    hf_to_sglang_mapper = None\n\n" + ORIGINAL_FORWARD + "\n" + rp.LOAD_ANCHOR + "        ]\n\n"
            + "    @classmethod\n    def get_model_config_for_expert_location(cls, config):\n" + rp.LOCATION_ANCHOR
            + "        return ModelConfigForExpertLocation(\n        )\n")


def test_patch_text_needs_each_anchor_once_and_reverts_exactly():
    text = _fake_model_file()
    patched = hd.patch_text(text, check_hash=False)
    assert patched.count(hd.MARK) == 3 and patched.count(hd.END) == 3 and hd.unpatch_text(patched) == text
    with pytest.raises(hd.PatchError, match="neither the analysed Pennyroyal file"):
        hd.patch_text(text)
    with pytest.raises(hd.PatchError, match="found 0 times"):
        hd.patch_text(text.replace(hd.FORWARD_ANCHOR, ""), check_hash=False)
    with pytest.raises(hd.PatchError, match="found 2 times"):
        hd.patch_text(text + hd.CLASS_ANCHOR, check_hash=False)
    with pytest.raises(hd.PatchError, match="already patched"):
        hd.patch_text(patched, check_hash=False)
    with pytest.raises(hd.PatchError, match="different or partial"):
        hd.unpatch_text(patched.replace("            _arc3_hc_dump.capture(", "            _arc3_hc_dump.capt("))
    # only lines were added: nothing of the original changes
    diff = list(difflib.ndiff(text.splitlines(), patched.splitlines()))
    added = len(patched.splitlines()) - len(text.splitlines())
    assert not [d for d in diff if d.startswith("- ")] and len([d for d in diff if d.startswith("+ ")]) == added == 20


def test_the_patch_composes_with_reap_in_either_order_on_the_anchored_text():
    text = _fake_model_file()
    reap_then_hc = hd.patch_text(rp.patch_text(text, check_hash=False), check_hash=False)
    hc_then_reap = rp.patch_text(hd.patch_text(text, check_hash=False), check_hash=False)
    assert reap_then_hc == hc_then_reap and reap_then_hc.count(rp.MARK) == 2 and reap_then_hc.count(hd.MARK) == 3
    assert hd.unpatch_text(reap_then_hc) == rp.patch_text(text, check_hash=False)
    compile(reap_then_hc, "qwen4_exp.py", "exec")


def _forward_of(text: str, flag: bool, dump_module=None):
    """Qwen4ExpForConditionalGeneration.forward of TEXT, defined on a stand-in class, with the module flag set."""
    tree = ast.parse(text)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Qwen4ExpForConditionalGeneration")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    assert [ast.unparse(d) for d in fn.decorator_list] == ["torch.no_grad()"]
    source = textwrap.indent(textwrap.dedent(" " * fn.col_offset + ast.get_source_segment(text, fn)), "    ")
    code = ("class Base:\n    def forward(self, *args, **kwargs):\n        return self.base(*args, **kwargs)\n\n\n"
            "class Model(Base):\n" + source)

    class LogitsProcessorOutput:
        hidden_states = "untouched"

    ns = {"LogitsProcessorOutput": LogitsProcessorOutput, "_ARC3_HC_DUMP": flag}
    if dump_module is not None:
        ns["_arc3_hc_dump"] = dump_module
    exec(compile(code, "<forward>", "exec"), ns)  # the method's own source, from the file under test
    return ns["Model"], LogitsProcessorOutput


def _run_forward(model_cls, lpo, hc="HC", calls=None):
    model = model_cls()
    model.model = types.SimpleNamespace(last_hc_hidden_states=hc)
    out = lpo()

    def base(*args, **kwargs):
        if calls is not None:
            calls.append(("forward", args, kwargs))
        return out

    model.base = base
    result = model.forward("ids", "pos", "fb", flag=1)
    return model, result


def _check_forward(original: str, patched: str) -> None:
    for hc in ("HC", None):
        _, before = _run_forward(*_forward_of(original, False), hc=hc)
        _, after = _run_forward(*_forward_of(patched, False), hc=hc)  # _arc3_hc_dump undefined: never touched
        assert after.hidden_states == before.hidden_states == (hc or "untouched")
    calls: list = []
    hooks = types.SimpleNamespace(input_ids_before=lambda a, k: calls.append(("before", a, k)) or "IDS",
                                  capture=lambda *a: calls.append(("capture", *a)))
    model, result = _run_forward(*_forward_of(patched, True, hooks), calls=calls)
    assert [c[0] for c in calls] == ["before", "forward", "capture"] and result.hidden_states == "HC"
    assert calls[0][1:] == (("ids", "pos", "fb"), {"flag": 1})
    assert calls[2][1:] == (model, ("ids", "pos", "fb"), {"flag": 1}, "HC", "IDS")


def test_the_patched_forward_behaves_as_before_unless_the_variable_is_set():
    text = _fake_model_file()
    _check_forward(text, hd.patch_text(text, check_hash=False))


def test_the_module_flag_is_read_once_and_imports_nothing_when_unset(monkeypatch):
    block = hd.CLASS_NEW.replace(hd.CLASS_ANCHOR, "")
    monkeypatch.delenv(hd.ENV, raising=False)
    ns: dict = {}
    exec(compile(block, "<block>", "exec"), ns)  # the patch's own text
    assert ns["_ARC3_HC_DUMP"] is False and "_arc3_hc_dump" not in ns
    fake = types.ModuleType("sglang.srt.arc3_hc_dump")
    srt = types.ModuleType("sglang.srt")
    srt.arc3_hc_dump = fake
    monkeypatch.setitem(sys.modules, "sglang", types.ModuleType("sglang"))
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.arc3_hc_dump", fake)
    monkeypatch.setenv(hd.ENV, "/some/dir")
    ns = {}
    exec(compile(block, "<block>", "exec"), ns)
    assert ns["_ARC3_HC_DUMP"] is True and ns["_arc3_hc_dump"] is fake


def _site_packages(tmp_path: Path, text: str) -> Path:
    site = tmp_path / "site-packages"
    (site / hd.MODEL_FILE).parent.mkdir(parents=True)
    (site / hd.MODEL_FILE).write_text(text)
    cache = site / hd.MODEL_FILE.parent / "__pycache__"
    cache.mkdir()
    (cache / "qwen4_exp.cpython-312.pyc").write_bytes(b"stale")
    return site


def test_apply_and_revert_on_an_installed_tree(tmp_path):
    site = _site_packages(tmp_path, _fake_model_file())
    target = site / hd.MODEL_FILE
    assert hd.main(["apply", "--site-packages", str(site)]) == 1  # not the analysed file
    assert target.read_text() == _fake_model_file() and not (site / hd.MODULE_FILE).exists()
    assert hd.main(["apply", "--site-packages", str(site), "--allow-other-version"]) == 0
    patched = target.read_text()
    assert patched == hd.patch_text(_fake_model_file(), check_hash=False)
    assert (site / hd.MODULE_FILE).read_text() == Path(hd.__file__).read_text()
    assert not list((target.parent / "__pycache__").glob("*.pyc"))
    assert hd.main(["apply", "--site-packages", str(site), "--allow-other-version"]) == 0 and target.read_text() == patched
    target.write_text(patched.replace("        if _ARC3_HC_DUMP:\n            _arc3_hc_dump.capture(", "        if 1:\n"
                                      "            _arc3_hc_dump.capture(", 1))
    assert hd.main(["apply", "--site-packages", str(site), "--allow-other-version"]) == 1  # a different patch
    target.write_text(patched)
    assert hd.main(["revert", "--site-packages", str(site)]) == 0 and target.read_text() == _fake_model_file()
    assert hd.main(["revert", "--site-packages", str(site)]) == 0  # nothing to revert
    assert hd.main(["apply", "--site-packages", str(tmp_path / "nowhere")]) == 1


@needs_wheel
def test_the_real_wheel_file_patches_composes_with_reap_and_refuses_other_versions(tmp_path):
    with zipfile.ZipFile(WHEEL) as wheel:
        original = wheel.read(str(hd.MODEL_FILE)).decode()
    sha = hashlib.sha256
    pinned = {sha(original.encode()).hexdigest(): "pristine", sha(rp.patch_text(original).encode()).hexdigest(): "reap"}
    assert pinned == BASE  # both hashes the patch accepts are these two files
    site = _site_packages(tmp_path / "hc-first", original)
    target = site / hd.MODEL_FILE
    assert hd.main(["apply", "--site-packages", str(site)]) == 0
    patched = target.read_text()
    assert patched == hd.patch_text(original) and not list((target.parent / "__pycache__").glob("*.pyc"))
    _check_forward(original, patched)  # the real method, unset and set
    flag = patched.index("_ARC3_HC_DUMP = bool(")
    assert 0 < patched.index(hd.CLASS_ANCHOR) - flag < 200 and patched.index("_arc3_hc_dump.capture(") > flag
    assert hd.main(["apply", "--site-packages", str(site)]) == 0 and target.read_text() == patched
    # REAP after this patch: its installer accepts only the pristine file and refuses; nothing changes
    assert rp.main(["apply", "--site-packages", str(site), "--kept", str(KEPT)]) == 1 and target.read_text() == patched
    # REAP first, then this: accepted, and the same text as forcing REAP onto the HC-patched file
    site_b = _site_packages(tmp_path / "reap-first", original)
    assert rp.main(["apply", "--site-packages", str(site_b), "--kept", str(KEPT)]) == 0
    assert hd.main(["apply", "--site-packages", str(site_b)]) == 0
    both = (site_b / hd.MODEL_FILE).read_text()
    assert rp.main(["apply", "--site-packages", str(site), "--allow-other-version"]) == 0 and target.read_text() == both
    assert hd.main(["apply", "--site-packages", str(site_b)]) == 0 and (site_b / hd.MODEL_FILE).read_text() == both
    # re-running REAP on the combined file fails (its idempotence check wants the pristine base): revert, REAP, apply
    assert rp.main(["apply", "--site-packages", str(site_b)]) == 1
    assert hd.main(["revert", "--site-packages", str(site_b)]) == 0
    assert (site_b / hd.MODEL_FILE).read_text() == rp.patch_text(original)
    assert rp.main(["apply", "--site-packages", str(site_b)]) == 0 and hd.main(["apply", "--site-packages", str(site_b)]) == 0
    assert (site_b / hd.MODEL_FILE).read_text() == both
    # another version is refused unless asked; the wheel check works without installing
    other = _site_packages(tmp_path / "other", original.replace("Qwen4-Exp VL weights", "Qwen4-Exp weights"))
    assert hd.main(["apply", "--site-packages", str(other)]) == 1
    assert hd.main(["apply", "--site-packages", str(other), "--allow-other-version"]) == 0
    assert hd.main(["check-wheel", str(WHEEL)]) == 0

