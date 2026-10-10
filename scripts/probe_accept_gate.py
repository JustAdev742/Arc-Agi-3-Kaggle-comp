#!/usr/bin/env python
"""The MTP draft's probe gate (docs/research/beat-tufa/mtp-drafter-finetune.md 5.3): does an arm's draft accept more
tokens per verify step than the baseline's on the same greedy probe requests?

    .venv/bin/python scripts/probe_accept_gate.py BASE.json ARM.json [--floor BASE2.json] [--json OUT]
        [--bootstrap 10000] [--min-gain 0.08]

Each file is one probe run's ``fidelity.json`` (scripts/fidelity_probe.py: greedy, lossless acceptance, the 154
held-out requests, pass ``seq`` one at a time and ``conc`` several in flight). Per pass and per request answered by
both runs, the difference of SGLang's ``spec_accept_length`` (completion tokens / verify steps): its mean, SD, SE and
a percentile bootstrap 95% interval over requests (seeded), the pooled accept length of each run (all completion
tokens / all verify steps), and the mean difference by prompt length and by game. ``--floor`` adds a second run of
the baseline's configuration, whose difference from BASE is the run-to-run noise (fidelity-base2 vs fidelity-base:
mean +0.025, SD 0.229, SE 0.018).

Verdict (the plan's pass rule, fixed before the data): PASS when in both passes the mean difference is at least
``--min-gain`` (0.08) with the whole interval above 0, and the arm failed no request; otherwise FAIL with the
reasons. Under greedy lossless decoding the outputs do not depend on the draft except through batching, so this
measures speed only. Exit status 0 on PASS, 2 on FAIL, 1 on bad input.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fidelity_compare import CONTEXT_BUCKETS, load

PASSES = ("seq", "conc")


def _accept(rec: dict | None) -> tuple[float, int] | None:
    """(accept length, verify steps) of one answered request, or None."""
    if not rec or not rec.get("ok"):
        return None
    spec = rec.get("spec") or {}
    length, steps = spec.get("spec_accept_length"), spec.get("spec_verify_ct")
    if length is None or not steps:
        return None
    return float(length), int(steps)


def _bucket(tokens) -> str:
    for limit, name in CONTEXT_BUCKETS:
        if limit is None or (tokens or 0) < limit:
            return name
    return CONTEXT_BUCKETS[-1][1]


def _interval(values: list[float], n_boot: int, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * (n_boot - 1))], means[int(0.975 * (n_boot - 1))]


def paired(base: dict, arm: dict, name: str, n_boot: int) -> dict:
    """The per-request accept-length differences arm - base on pass NAME."""
    b, a = base["passes"].get(name) or {}, arm["passes"].get(name) or {}
    diffs, rows, tok = [], [], {"base": [0.0, 0], "arm": [0.0, 0]}
    for rid in sorted(set(b) & set(a)):
        x, y = _accept(b[rid]), _accept(a[rid])
        if x is None or y is None:
            continue
        diffs.append(y[0] - x[0])
        sample = arm["samples"].get(rid) or base["samples"].get(rid) or {}
        rows.append({"id": rid, "game": sample.get("game") or rid.split("#")[0],
                     "bucket": _bucket(sample.get("prompt_tokens")), "base": x[0], "arm": y[0]})
        for key, (length, steps) in (("base", x), ("arm", y)):
            tok[key][0] += length * steps
            tok[key][1] += steps
    out = {"pass": name, "pairs": len(diffs),
           "failed": {"base": sum(1 for r in b.values() if not (r or {}).get("ok")),
                      "arm": sum(1 for r in a.values() if not (r or {}).get("ok"))},
           "missing": {"base": len(set(a) - set(b)), "arm": len(set(b) - set(a))}}
    if len(diffs) < 2:
        return out | {"error": f"only {len(diffs)} paired requests"}
    sd = statistics.stdev(diffs)
    lo, hi = _interval(diffs, n_boot)
    pooled = {k: round(v[0] / v[1], 4) if v[1] else None for k, v in tok.items()}
    out.update(mean=round(statistics.fmean(diffs), 4), sd=round(sd, 4), se=round(sd / len(diffs) ** 0.5, 4),
               ci95=[round(lo, 4), round(hi, 4)], base_mean=round(statistics.fmean(r["base"] for r in rows), 4),
               arm_mean=round(statistics.fmean(r["arm"] for r in rows), 4), pooled=pooled,
               pooled_ratio=round(pooled["arm"] / pooled["base"], 4) if pooled["base"] else None)
    for key in ("bucket", "game"):
        groups: dict[str, list[float]] = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["arm"] - r["base"])
        order = [name for _, name in CONTEXT_BUCKETS] if key == "bucket" else sorted(groups)
        out[f"by_{key}"] = {g: {"n": len(groups[g]), "mean": round(statistics.fmean(groups[g]), 4)}
                            for g in order if g in groups}
    return out


def verdict(results: list[dict], min_gain: float) -> tuple[bool, list[str]]:
    reasons = []
    for r in results:
        if "error" in r:
            reasons.append(f"{r['pass']}: {r['error']}")
            continue
        if r["mean"] < min_gain:
            reasons.append(f"{r['pass']}: mean gain {r['mean']:+.3f} < {min_gain:+.3f}")
        if r["ci95"][0] <= 0:
            reasons.append(f"{r['pass']}: the 95% interval {r['ci95']} reaches 0")
        if r["failed"]["arm"]:
            reasons.append(f"{r['pass']}: the arm failed {r['failed']['arm']} request(s)")
    return not reasons, reasons


def report(base: dict, arm: dict, results: list[dict], floor: list[dict], ok: bool, reasons: list[str]) -> str:
    lines = [f"probe gate: {arm['label']} ({arm['path']}) vs {base['label']} ({base['path']})"]
    for r in results:
        if "error" in r:
            lines.append(f"  {r['pass']}: {r['error']}")
            continue
        lines.append(f"  {r['pass']}: {r['pairs']} paired requests; accept {r['base_mean']:.3f} -> {r['arm_mean']:.3f}, "
                     f"mean diff {r['mean']:+.3f} (SD {r['sd']:.3f}, SE {r['se']:.3f}, 95% {r['ci95'][0]:+.3f} to "
                     f"{r['ci95'][1]:+.3f}); pooled {r['pooled']['base']} -> {r['pooled']['arm']} (x{r['pooled_ratio']}); "
                     f"failed base {r['failed']['base']} arm {r['failed']['arm']}")
        lines.append("    by prompt length: " + ", ".join(f"{g} {v['mean']:+.3f} (n {v['n']})"
                                                         for g, v in r["by_bucket"].items()))
        lines.append("    by game: " + ", ".join(f"{g} {v['mean']:+.3f}" for g, v in r["by_game"].items()))
    for r in floor:
        if "error" not in r:
            lines.append(f"  floor {r['pass']} (second baseline run): mean diff {r['mean']:+.3f} (SD {r['sd']:.3f}, "
                         f"95% {r['ci95'][0]:+.3f} to {r['ci95'][1]:+.3f})")
    lines.append("verdict: " + ("PASS" if ok else "FAIL: " + "; ".join(reasons)))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=Path)
    ap.add_argument("arm", type=Path)
    ap.add_argument("--floor", type=Path, default=None, help="a second run of the baseline's configuration")
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--min-gain", type=float, default=0.08)
    args = ap.parse_args(argv)
    try:
        runs = load([args.base, args.arm] + ([args.floor] if args.floor else []))
    except (OSError, ValueError, KeyError) as exc:
        print(f"probe_accept_gate: {exc}", file=sys.stderr)
        return 1
    base, arm = runs[0], runs[1]
    results = [paired(base, arm, name, args.bootstrap) for name in PASSES]
    floor = [paired(base, runs[2], name, args.bootstrap) for name in PASSES] if args.floor else []
    ok, reasons = verdict(results, args.min_gain)
    print(report(base, arm, results, floor, ok, reasons))
    if args.json:
        args.json.write_text(json.dumps({"base": str(args.base), "arm": str(args.arm), "min_gain": args.min_gain,
                                         "passes": results, "floor": floor, "pass": ok, "reasons": reasons},
                                        indent=1) + "\n")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
