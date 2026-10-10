"""scripts/probe_accept_gate.py: the MTP draft's probe gate on two fidelity.json files (synthetic, no GPU)."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import probe_accept_gate as gate  # noqa: E402


def _run(path: Path, arm: str, accept: dict[str, float], *, failed: set[str] = frozenset(), no_spec: set[str] = frozenset()):
    samples = [{"id": rid, "game": rid.split("#")[0], "prompt_tokens": 10000 + 1000 * i}
               for i, rid in enumerate(sorted(accept))]
    passes = {}
    for name in ("seq", "conc"):
        records = []
        for rid, value in sorted(accept.items()):
            rec = {"id": rid, "ok": rid not in failed}
            if rid not in failed and rid not in no_spec:
                rec["spec"] = {"spec_accept_length": value, "spec_verify_ct": 60}
            records.append(rec)
        passes[name] = {"records": records}
    path.write_text(json.dumps({"arm": arm, "samples": samples, "passes": passes}))
    return path


def _accepts(n: int, shift: float = 0.0, seed: int = 0) -> dict[str, float]:
    rng = random.Random(seed)
    return {f"g{i % 3}#{i:03d}": 2.7 + rng.gauss(0, 0.2) + shift for i in range(n)}


def test_a_clear_gain_passes_and_is_reported_per_pass(tmp_path, capsys):
    base = _accepts(40)
    arm = {k: v + 0.15 for k, v in base.items()}
    code = gate.main([str(_run(tmp_path / "b.json", "base", base)), str(_run(tmp_path / "a.json", "draft", arm)),
                      "--bootstrap", "500", "--json", str(tmp_path / "out.json")])
    assert code == 0
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["pass"] and [p["pass"] for p in out["passes"]] == ["seq", "conc"]
    seq = out["passes"][0]
    assert seq["pairs"] == 40 and abs(seq["mean"] - 0.15) < 1e-6 and seq["sd"] < 1e-6
    assert seq["ci95"][0] > 0 and abs(seq["pooled_ratio"] - (seq["pooled"]["arm"] / seq["pooled"]["base"])) < 1e-3
    assert set(seq["by_game"]) == {"g0", "g1", "g2"} and sum(v["n"] for v in seq["by_bucket"].values()) == 40
    assert "verdict: PASS" in capsys.readouterr().out


def test_a_small_or_noisy_gain_fails_with_its_reasons(tmp_path):
    base = _accepts(40)
    small = {k: v + 0.03 for k, v in base.items()}
    code = gate.main([str(_run(tmp_path / "b.json", "base", base)), str(_run(tmp_path / "a.json", "draft", small)),
                      "--bootstrap", "500", "--json", str(tmp_path / "out.json")])
    out = json.loads((tmp_path / "out.json").read_text())
    assert code == 2 and not out["pass"] and any("mean gain +0.030 < +0.080" in r for r in out["reasons"])
    noisy = {k: v + 0.1 + (1.5 if i % 2 else -1.5) for i, (k, v) in enumerate(sorted(base.items()))}
    _run(tmp_path / "n.json", "draft", noisy)
    assert gate.main([str(tmp_path / "b.json"), str(tmp_path / "n.json"), "--bootstrap", "500",
                      "--json", str(tmp_path / "out2.json")]) == 2
    assert any("interval" in r for r in json.loads((tmp_path / "out2.json").read_text())["reasons"])


def test_failed_requests_fail_the_gate_and_unpaired_ones_are_left_out(tmp_path):
    base = _accepts(30)
    arm = {k: v + 0.2 for k, v in base.items()}
    ids = sorted(base)
    _run(tmp_path / "b.json", "base", base, no_spec={ids[0]})
    _run(tmp_path / "a.json", "draft", arm, failed={ids[1]})
    code = gate.main([str(tmp_path / "b.json"), str(tmp_path / "a.json"), "--bootstrap", "200",
                      "--json", str(tmp_path / "out.json")])
    out = json.loads((tmp_path / "out.json").read_text())
    seq = out["passes"][0]
    assert code == 2 and seq["pairs"] == 28 and seq["failed"] == {"base": 0, "arm": 1}
    assert any("failed 1 request" in r for r in out["reasons"])


def test_a_floor_run_is_compared_to_the_baseline_and_bad_input_exits_1(tmp_path, capsys):
    base = _accepts(30)
    _run(tmp_path / "b.json", "base", base)
    _run(tmp_path / "b2.json", "base", _accepts(30, seed=0))
    _run(tmp_path / "a.json", "draft", {k: v + 0.2 for k, v in base.items()})
    assert gate.main([str(tmp_path / "b.json"), str(tmp_path / "a.json"), "--floor", str(tmp_path / "b2.json"),
                      "--bootstrap", "200"]) == 0
    assert "floor seq (second baseline run): mean diff +0.000" in capsys.readouterr().out
    (tmp_path / "junk.json").write_text("{}")
    assert gate.main([str(tmp_path / "b.json"), str(tmp_path / "missing.json")]) == 1
