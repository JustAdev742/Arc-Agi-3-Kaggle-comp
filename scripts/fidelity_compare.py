#!/usr/bin/env python
"""Compare fidelity-probe runs: how far a pruned server's greedy outputs and logprobs move, against the floor.

    .venv/bin/python scripts/fidelity_compare.py BASE.json REAP.json [BASE2.json] [--json OUT] [--bootstrap 2000]

Each file is one notebook run's ``fidelity.json`` (scripts/fidelity_probe.py; docs/research/beat-tufa/
fidelity-probe.md); its arm (``base`` or ``reap448``) is read from the file. A series is one pass of one run (``seq``:
one request at a time; ``conc``: several in flight). For every pair of series, per request (both answered):

- ``prefix``: exact-match prefix of the two greedy token sequences (token ids when both runs returned them, else
  token strings); ``identical`` when the sequences are equal; ``prefix share`` = prefix / the longer length, i.e.
  the share of generated positions that agree before the first divergence.
- ``|dlp|``: absolute difference of the chosen tokens' logprobs on the agreed prefix (pooled over tokens; p99, max).
- top-k on the agreed prefix: ``rank-2 flips`` (the runner-up token differs) and ``top-k set changes`` per 1,000
  positions, and the mean |dlp| over tokens present in both top-k lists. The top-1 token cannot flip on the agreed
  prefix (greedy: the chosen token is the top-1), so a top-1 flip is the divergence itself; at that position the
  ``margin`` of each run is its top-1 logprob minus its logprob of the other run's token (when in its top-k), and a
  divergence is a ``near tie`` when both margins are below 0.1 nats.

Pairs are grouped: ``floor`` = base vs base (seq vs conc inside one run: batching and prefix-cache noise; and the
same pass across two base runs when a second base file is given: run-to-run noise), ``reap`` = base vs REAP on the
same pass, ``reap-own`` = REAP seq vs conc (REAP's own batching noise). The headline compares the REAP seq pair with
the floor pair of the most similar structure (across-run seq when available, else within-run), with a paired
bootstrap over requests, and breaks both down by game, context length, images and turn quarter.

Reading (judgment thresholds, see the doc): REAP is "close to the floor" when its pooled |dlp| is at most 2x the
floor's and its mean prefix share is not lower by more than 0.05 (bootstrap 95% interval); it "shifts the model" when
the |dlp| ratio is 3x or more or the prefix share is lower by more than 0.10 with the whole interval below zero;
anything else is "a small, detectable shift" and the breakdowns say where.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

NEAR_TIE = 0.1  # nats: a divergence where both runs had their two candidates this close is a coin flip
CONTEXT_BUCKETS = ((16384, "<16k"), (49152, "16-48k"), (81920, "48-80k"), (None, ">=80k"))
IMAGE_BUCKETS = ((10, "1-9 images"), (30, "10-29 images"), (None, ">=30 images"))


def load(paths: list[Path]) -> list[dict]:
    runs, seen = [], {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        if "passes" not in data or "samples" not in data:
            raise SystemExit(f"{path}: not a fidelity-probe output")
        arm = str(data.get("arm") or "unknown")
        seen[arm] = seen.get(arm, 0) + 1
        label = arm if seen[arm] == 1 else f"{arm}#{seen[arm]}"
        runs.append({"label": label, "arm": arm, "path": str(path), "data": data,
                     "samples": {s["id"]: s for s in data["samples"]},
                     "passes": {name: {r["id"]: r for r in p.get("records") or [] if r}
                                for name, p in data["passes"].items()}})
    return runs


def _seqs(x: dict, y: dict) -> tuple[list, list, bool]:
    if x.get("token_ids") is not None and y.get("token_ids") is not None:
        return x["token_ids"], y["token_ids"], True
    return x["tokens"], y["tokens"], False


def _top_key(entry: list, by_id: bool):
    return entry[0] if by_id and entry[0] is not None else entry[1]


def compare_records(x: dict, y: dict) -> dict:
    """Divergence metrics of two greedy completions of the same request."""
    sx, sy, by_id = _seqs(x, y)
    n = min(len(sx), len(sy))
    prefix = next((j for j in range(n) if sx[j] != sy[j]), n)
    longest = max(len(sx), len(sy))
    out = {"len_x": len(sx), "len_y": len(sy), "prefix": prefix, "identical": prefix == len(sx) == len(sy),
           "prefix_share": prefix / longest if longest else 1.0, "by_ids": by_id,
           "finish_same": x.get("finish_reason") == y.get("finish_reason")}
    dlp = [abs(a - b) for a, b in zip(x["logprobs"][:prefix], y["logprobs"][:prefix])
           if a is not None and b is not None]
    out["dlp"] = dlp
    rank2 = sets = 0
    shared = []
    tx, ty = x.get("top") or [], y.get("top") or []
    positions = min(prefix, len(tx), len(ty))
    for j in range(positions):
        top_by_id = by_id and all(e[0] is not None for e in tx[j] + ty[j])
        kx = [_top_key(e, top_by_id) for e in tx[j]]
        ky = [_top_key(e, top_by_id) for e in ty[j]]
        if len(kx) >= 2 and len(ky) >= 2 and kx[1] != ky[1]:
            rank2 += 1
        if set(kx) != set(ky):
            sets += 1
        lx = {k: e[2] for k, e in zip(kx, tx[j]) if e[2] is not None}
        ly = {k: e[2] for k, e in zip(ky, ty[j]) if e[2] is not None}
        common = set(lx) & set(ly)
        if common:
            shared.append(sum(abs(lx[k] - ly[k]) for k in common) / len(common))
    out.update(top_positions=positions, rank2_flips=rank2, topset_changes=sets, shared_top_dlp=shared)
    out["margin_x"] = out["margin_y"] = None
    if prefix < n:  # the divergence: each run's top-1 against the other run's choice, within its own top-k
        def margin(top_list: list, own, other):
            if prefix >= len(top_list) or not top_list[prefix]:
                return None
            entries = {_top_key(e, by_id): e[2] for e in top_list[prefix] if e[2] is not None}
            if own in entries and other in entries:
                return entries[own] - entries[other]
            return None
        out["margin_x"] = margin(tx, sx[prefix], sy[prefix])
        out["margin_y"] = margin(ty, sy[prefix], sx[prefix])
    return out


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarize(rows: list[dict]) -> dict:
    """Pair-level numbers from per-request comparisons."""
    if not rows:
        return {"n": 0}
    pooled = [d for r in rows for d in r["dlp"]]
    positions = sum(r["top_positions"] for r in rows)
    diverged = [r for r in rows if r["prefix"] < min(r["len_x"], r["len_y"])]
    margins = [r for r in diverged if r["margin_x"] is not None and r["margin_y"] is not None]
    shared = [d for r in rows for d in r["shared_top_dlp"]]
    per_request = [statistics.fmean(r["dlp"]) for r in rows if r["dlp"]]
    return {
        "n": len(rows),
        "identical_share": sum(r["identical"] for r in rows) / len(rows),
        "mean_prefix_share": statistics.fmean(r["prefix_share"] for r in rows),
        "median_prefix_tokens": statistics.median(r["prefix"] for r in rows),
        "median_first_divergence": statistics.median(r["prefix"] for r in diverged) if diverged else None,
        "diverged": len(diverged),
        "length_only_differs": sum(1 for r in rows if not r["identical"] and r["prefix"] == min(r["len_x"], r["len_y"])),
        "finish_differs": sum(1 for r in rows if not r["finish_same"]),
        "agreed_tokens": len(pooled),
        "mean_abs_dlp": statistics.fmean(pooled) if pooled else None,
        "mean_abs_dlp_per_request": statistics.fmean(per_request) if per_request else None,
        "p99_abs_dlp": _quantile(pooled, 0.99),
        "max_abs_dlp": max(pooled) if pooled else None,
        "mean_shared_topk_abs_dlp": statistics.fmean(shared) if shared else None,
        "rank2_flips_per_1k": 1000 * sum(r["rank2_flips"] for r in rows) / positions if positions else None,
        "topset_changes_per_1k": 1000 * sum(r["topset_changes"] for r in rows) / positions if positions else None,
        "near_tie_share": (sum(1 for r in margins if max(r["margin_x"], r["margin_y"]) < NEAR_TIE) / len(margins)
                           if margins else None),
        "median_margin": (statistics.median(max(r["margin_x"], r["margin_y"]) for r in margins) if margins else None),
        "divergences_with_margins": len(margins),
        "compared_by_ids": all(r["by_ids"] for r in rows),
    }


def _bucket(value: float, buckets: tuple) -> str:
    for limit, name in buckets:
        if limit is None or value < limit:
            return name
    return buckets[-1][1]


def groups_of(sample: dict) -> dict[str, str]:
    return {"game": str(sample.get("game")),
            "context": _bucket(int(sample.get("prompt_tokens") or 0), CONTEXT_BUCKETS),
            "images": _bucket(int(sample.get("n_images") or 0), IMAGE_BUCKETS),
            "last message": "fresh frame (user + image)" if sample.get("last_has_image") else "tool result / text",
            "turn quarter": f"Q{int(sample.get('quarter', 0)) + 1}"}


def pair_rows(a: dict, pa: str, b: dict, pb: str) -> dict[str, dict]:
    ra, rb = a["passes"].get(pa, {}), b["passes"].get(pb, {})
    return {i: compare_records(ra[i], rb[i]) for i in ra if i in rb and ra[i].get("ok") and rb[i].get("ok")}


def classify(a: dict, pa: str, b: dict, pb: str) -> str:
    base_a, base_b = a["arm"] == "base", b["arm"] == "base"
    if a is b:
        return "floor" if base_a else "reap-own"
    if base_a and base_b:
        return "floor" if pa == pb else "floor-mixed"
    if base_a != base_b:
        return "reap" if pa == pb else "reap-mixed"
    return "other"


def bootstrap(x: dict[str, dict], y: dict[str, dict], key, n: int, seed: int = 0) -> dict | None:
    """Paired bootstrap over the requests both pairs compared: mean(key(x)) - mean(key(y)), 95% interval."""
    ids = sorted(i for i in set(x) & set(y) if key(x[i]) is not None and key(y[i]) is not None)
    if len(ids) < 2:
        return None
    diffs = [key(x[i]) - key(y[i]) for i in ids]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(diffs) for _ in diffs) for _ in range(n))
    return {"n": len(ids), "mean": statistics.fmean(diffs), "lo": means[int(0.025 * n)], "hi": means[int(0.975 * n) - 1]}


def _mean_dlp(row: dict) -> float | None:
    return statistics.fmean(row["dlp"]) if row["dlp"] else None


def verdict(reap: dict, floor: dict, prefix_ci: dict | None) -> str:
    if not reap.get("n") or not floor.get("n"):
        return "not enough data"
    ratio = (reap["mean_abs_dlp"] / floor["mean_abs_dlp"]
             if reap.get("mean_abs_dlp") is not None and floor.get("mean_abs_dlp") else None)
    drop = prefix_ci["mean"] if prefix_ci else reap["mean_prefix_share"] - floor["mean_prefix_share"]
    lo = prefix_ci["lo"] if prefix_ci else drop
    hi = prefix_ci["hi"] if prefix_ci else drop
    if (ratio is not None and ratio >= 3) or (drop < -0.10 and hi < 0):
        return "REAP SHIFTS THE MODEL beyond serving noise"
    if (ratio is None or ratio <= 2) and lo >= -0.05:
        return "REAP is CLOSE TO THE FLOOR (within serving noise)"
    return "a SMALL, DETECTABLE SHIFT: read the breakdowns for where"


def analyze(paths: list[Path], n_boot: int = 2000) -> dict:
    runs = load(paths)
    files = {r["label"]: (r["data"].get("params") or {}).get("files") for r in runs}
    warnings = []
    if len({json.dumps(v, sort_keys=True) for v in files.values()}) > 1:
        warnings.append(f"the runs replayed different data files: {files}")
    series = [(r, p) for r in runs for p in r["passes"]]
    pairs = []
    for i, (a, pa) in enumerate(series):
        for b, pb in series[i + 1:]:
            rows = pair_rows(a, pa, b, pb)
            pairs.append({"a": f"{a['label']}.{pa}", "b": f"{b['label']}.{pb}", "kind": classify(a, pa, b, pb),
                          "rows": rows, "summary": summarize(list(rows.values()))})
    out = {"runs": [{"label": r["label"], "arm": r["arm"], "path": r["path"],
                     "passes": {name: {k: v for k, v in p.items() if k != "records"}
                                for name, p in r["data"]["passes"].items()},
                     "server": {k: (r["data"].get("server_info") or {}).get(k) for k in
                                ("json_model_override_args", "speculative_accept_threshold_single",
                                 "speculative_accept_threshold_acc", "max_running_requests", "max_total_num_tokens")}}
                    for r in runs],
           "pairs": [{k: v for k, v in p.items() if k != "rows"} for p in pairs], "warnings": warnings}
    base = next((r for r in runs if r["arm"] == "base"), None)
    reap = next((r for r in runs if r["arm"] != "base"), None)
    if base is None or reap is None:
        out["headline"] = None
        return out

    def find(a: str, b: str) -> dict | None:
        return next((p for p in pairs if {p["a"], p["b"]} == {a, b}), None)

    reap_pair = find(f"{base['label']}.seq", f"{reap['label']}.seq")
    base2 = next((r for r in runs if r["arm"] == "base" and r is not base), None)
    floor_pair = (find(f"{base['label']}.seq", f"{base2['label']}.seq") if base2 else None) \
        or find(f"{base['label']}.seq", f"{base['label']}.conc")
    if reap_pair is None or floor_pair is None:
        out["headline"] = None
        return out
    prefix_ci = bootstrap(reap_pair["rows"], floor_pair["rows"], lambda r: r["prefix_share"], n_boot)
    dlp_ci = bootstrap(reap_pair["rows"], floor_pair["rows"], _mean_dlp, n_boot, seed=1)
    breakdowns = {}
    samples = base["samples"]
    for dim in ("game", "context", "images", "last message", "turn quarter"):
        table = {}
        for name, pair in (("reap", reap_pair), ("floor", floor_pair)):
            grouped: dict[str, list] = {}
            for rid, row in pair["rows"].items():
                grouped.setdefault(groups_of(samples.get(rid, {}))[dim], []).append(row)
            for group, rows in grouped.items():
                table.setdefault(group, {})[name] = summarize(rows)
        breakdowns[dim] = dict(sorted(table.items()))
    out["headline"] = {"reap_pair": f"{reap_pair['a']} vs {reap_pair['b']}",
                       "floor_pair": f"{floor_pair['a']} vs {floor_pair['b']}",
                       "reap": reap_pair["summary"], "floor": floor_pair["summary"],
                       "prefix_share_diff_ci": prefix_ci, "mean_dlp_diff_ci": dlp_ci,
                       "verdict": verdict(reap_pair["summary"], floor_pair["summary"], prefix_ci),
                       "breakdowns": breakdowns}
    return out


def _f(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render(report: dict) -> str:
    lines = ["Fidelity probe comparison (scripts/fidelity_compare.py)", ""]
    for r in report["runs"]:
        passes = ", ".join(f"{name}: {p.get('ok')} ok / {p.get('failed')} failed, {p.get('wall_s')} s, accept "
                           f"{p.get('mean_accept_length')}" for name, p in r["passes"].items())
        lines.append(f"run {r['label']:<10} {r['path']}  [{passes}]  server {r['server']}")
    for w in report["warnings"]:
        lines.append(f"WARNING: {w}")
    lines += ["", f"{'pair':<34} {'kind':<11} {'n':>4} {'ident':>6} {'prefix':>7} {'med div':>7} {'|dlp|':>7} "
                  f"{'p99':>6} {'top|dlp|':>8} {'r2/1k':>6} {'set/1k':>6} {'ties':>5}"]
    order = {"floor": 0, "floor-mixed": 1, "reap-own": 2, "reap": 3, "reap-mixed": 4, "other": 5}
    for p in sorted(report["pairs"], key=lambda p: (order.get(p["kind"], 9), p["a"], p["b"])):
        s = p["summary"]
        if not s.get("n"):
            lines.append(f"{p['a'] + ' vs ' + p['b']:<34} {p['kind']:<11}    0")
            continue
        lines.append(f"{p['a'] + ' vs ' + p['b']:<34} {p['kind']:<11} {s['n']:>4} {s['identical_share']:>6.2f} "
                     f"{s['mean_prefix_share']:>7.3f} {_f(s['median_first_divergence'], 0):>7} "
                     f"{_f(s['mean_abs_dlp'], 4):>7} {_f(s['p99_abs_dlp'], 3):>6} "
                     f"{_f(s['mean_shared_topk_abs_dlp'], 4):>8} {_f(s['rank2_flips_per_1k'], 1):>6} "
                     f"{_f(s['topset_changes_per_1k'], 1):>6} {_f(s['near_tie_share'], 2):>5}")
    lines.append("  ident: share identical; prefix: mean share agreeing before the first divergence; med div: median "
                 "first-divergence token (diverged requests); |dlp|: mean |logprob difference| of the chosen tokens "
                 "on the agreed prefix; top|dlp|: same over tokens in both top-k lists; r2/1k, set/1k: runner-up "
                 "flips and top-k set changes per 1,000 agreed positions; ties: share of divergences at a near tie "
                 f"(both margins < {NEAR_TIE} nats)")
    h = report.get("headline")
    if not h:
        lines += ["", "No headline: needs one base run and one REAP run, each with a seq pass."]
        return "\n".join(lines)
    reap, floor = h["reap"], h["floor"]
    lines += ["", f"HEADLINE: {h['reap_pair']} (REAP) against {h['floor_pair']} (floor)"]
    for name, s in (("REAP ", reap), ("floor", floor)):
        lines.append(f"  {name}: identical {s['identical_share']:.2f}, prefix share {s['mean_prefix_share']:.3f}, "
                     f"mean |dlp| {_f(s['mean_abs_dlp'], 4)} (per request {_f(s['mean_abs_dlp_per_request'], 4)}), "
                     f"near-tie divergences {_f(s['near_tie_share'], 2)} of {s['divergences_with_margins']}")
    if reap.get("mean_abs_dlp") and floor.get("mean_abs_dlp"):
        lines.append(f"  |dlp| ratio REAP/floor: {reap['mean_abs_dlp'] / floor['mean_abs_dlp']:.2f}")
    for label, ci in (("prefix share REAP - floor", h["prefix_share_diff_ci"]),
                      ("per-request |dlp| REAP - floor", h["mean_dlp_diff_ci"])):
        if ci:
            lines.append(f"  {label}: {ci['mean']:+.4f} (95% bootstrap {ci['lo']:+.4f} .. {ci['hi']:+.4f}, "
                         f"{ci['n']} requests)")
    lines.append(f"  READING: {h['verdict']}")
    for dim, table in h["breakdowns"].items():
        lines += ["", f"by {dim}: {'group':<28} {'n':>4} {'prefix R/F':>13} {'ident R/F':>11} {'|dlp| R/F':>17}"]
        for group, cell in table.items():
            r, f = cell.get("reap", {"n": 0}), cell.get("floor", {"n": 0})
            if not r.get("n") or not f.get("n"):
                continue
            lines.append(f"   {'':<{len(dim)}} {group:<28} {r['n']:>4} {r['mean_prefix_share']:>6.3f}/"
                         f"{f['mean_prefix_share']:<6.3f} {r['identical_share']:>5.2f}/{f['identical_share']:<5.2f} "
                         f"{_f(r['mean_abs_dlp'], 4):>8}/{_f(f['mean_abs_dlp'], 4):<8}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", type=Path, help="fidelity.json of each run (2 or 3)")
    ap.add_argument("--json", type=Path, default=None, help="also write the numbers here")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    if not 1 <= len(args.files) <= 4:
        raise SystemExit("give 1 to 4 fidelity.json files")
    report = analyze(args.files, args.bootstrap)
    print(render(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    sys.exit(main())
