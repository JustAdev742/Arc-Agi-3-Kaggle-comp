#!/usr/bin/env python
"""Replay model: how much would sharing the model server by game value (patch P21) change the public-25 score?

    .venv/bin/python scripts/sim_call_share.py [--bench-glob GLOB ...] [--draws 400]

Data: the per-game level completion times of stock-cap Duck runs on the Flash-Next server (benchmark.json files:
harvested public runs and ours). Under the stock first-come queue every game gets about the same call rate, so the
minutes a game spent on a level, times that rate, are the model calls the level took. Per (game, level) the hazard
per call is solves / calls spent (censored levels count their calls), with a weak prior for never-solved levels.

Model: each game cycles wait + service; the server finishes MU calls a minute (the measured 10.3 at 25 games). With
the gate's weight x wait rule the waits settle at theta / weight, so a game's call rate is 1 / (theta / w + S0) with
theta set by the server's capacity; S0 is the shortest cycle (in-server queue behind the admitted calls, decode, tool
time). Weight 1 + step x (level - 1), fading after a stalled level. Each draw samples the calls every level needs from
the hazards and plays the same draw under the first-come queue (step 0) and each weighting, so differences are paired.
Level score uses the per-(game, level) mean efficiency observed when solved. Assumes a call is worth the same however
long it waited, and that the server's throughput does not depend on which games it serves.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_BENCH = "/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/harvest_bench/*/benchmark.json"
CAP_MIN = 132.0
MU = 10.3            # calls per minute, whole server (exp-032: 1,356 requests in about 132 min)
FCFS_RATE = MU / 25  # calls per minute per game under the first-come queue with 25 games


def load_runs(patterns: list[str]) -> list[list[dict]]:
    runs = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            if "defiaudit__agi3-duck-qwen38-anim-v2" in f or "qwen3-8-27b" in f or "lb-9" in f:
                continue  # non-stock cap, or the 27B model
            data = json.loads(Path(f).read_text())
            games = data.get("game_runs") or data.get("runs") or []
            if not games:
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict) and "game_id" in v[0]:
                        games = v
                        break
            if len(games) != 25:
                continue
            out = []
            for g in games:
                n = int(g.get("levels_completed", 0))
                apl = list(g.get("actions_per_level") or [])
                base = list(g.get("base_actions_per_level") or [])
                hist = list(g.get("history") or [])
                wall = float(g.get("final_wallclock_seconds") or CAP_MIN * 60)
                if wall > CAP_MIN * 60 * 1.1:
                    break  # a longer cap than stock: not comparable
                done, k = [], 0
                for i in range(n):
                    k += apl[i]
                    done.append(float(hist[k - 1].get("wallclock_seconds") or 0.0) / 60 if 0 < k <= len(hist) else None)
                out.append({"gid": g["game_id"][:4], "apl": apl, "base": base, "done": done, "wall": wall / 60})
            else:
                runs.append(out)
    return runs


def fit(runs: list[list[dict]]):
    solves, exposure, eff = defaultdict(int), defaultdict(float), defaultdict(list)
    nlev = {}
    for run in runs:
        for g in run:
            gid = g["gid"]
            nlev[gid] = len(g["base"])
            prev = 0.0
            for k, t in enumerate(g["done"]):
                if t is None:
                    break
                exposure[(gid, k)] += max(0.05, t - prev) * FCFS_RATE
                solves[(gid, k)] += 1
                a, b = g["apl"][k], g["base"][k]
                eff[(gid, k)].append(min((b / a) ** 2, 1.15) if a else 0.0)
                prev = t
            else:
                k = len(g["done"])
                if k < len(g["base"]):
                    exposure[(gid, k)] += max(0.0, min(g["wall"], CAP_MIN) - prev) * FCFS_RATE
    hazard = {}
    for key, e in exposure.items():
        hazard[key] = (solves[key] + 0.1) / (e + 5.0)  # weak prior: never-solved levels keep a small chance
    mean_eff = {k: sum(v) / len(v) for k, v in eff.items()}
    return nlev, hazard, mean_eff


def game_score(solved: int, n: int, effs: list[float]) -> float:
    total = n * (n + 1) / 2
    s = sum((k + 1) * effs[k] for k in range(solved))
    return 100.0 * min(s, sum(k + 1 for k in range(solved))) / total


def rates(weights: list[float], s0: float, mu: float) -> list[float]:
    """Call rates per game (calls/min) with waits theta / w and cycle theta / w + s0, total capped at mu."""
    if not weights:
        return []
    if sum(1.0 / s0 for _ in weights) <= mu:
        return [1.0 / s0] * len(weights)
    lo, hi = 0.0, 1e4
    for _ in range(60):
        th = (lo + hi) / 2
        if sum(1.0 / (th / w + s0) for w in weights) > mu:
            lo = th
        else:
            hi = th
    return [1.0 / (hi / w + s0) for w in weights]


def play(needs, nlev, effs, step, stall_min, s0, dt=0.5, cap_min=CAP_MIN):
    """Minutes-stepped replay of one draw (one wave); returns the mean game score (percent)."""
    gids = list(needs)
    level = dict.fromkeys(gids, 0)
    work = dict.fromkeys(gids, 0.0)
    since = dict.fromkeys(gids, 0.0)
    t = 0.0
    while t < cap_min - 1e-9:
        active = [g for g in gids if level[g] < nlev[g]]
        ws = []
        for g in active:
            fade = 1.0 if stall_min <= 0 or (t - since[g]) <= stall_min else stall_min / (t - since[g])
            ws.append(1.0 + step * level[g] * fade)
        for g, r in zip(active, rates(ws, s0, MU)):
            work[g] += r * dt
            while level[g] < nlev[g] and work[g] >= needs[g][level[g]]:
                work[g] -= needs[g][level[g]]
                level[g] += 1
                since[g] = t + dt
        t += dt
    return sum(game_score(level[g], nlev[g], effs[g]) for g in gids) / len(gids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-glob", nargs="*", default=[SCRATCH_BENCH, str(ROOT / "runs/exp032-anim-flashnext/kernel-output/benchmark.json"),
                                                          str(ROOT / "runs/pub-*/kernel-output/benchmark.json")])
    ap.add_argument("--draws", type=int, default=400)
    ap.add_argument("--s0", type=float, default=0.85, help="shortest cycle in minutes (in-server queue + decode + tools)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=0, metavar="N",
                    help="model a hidden set of N games (each a random public game) played in waves; see --waves")
    ap.add_argument("--waves", type=int, nargs="*", default=[4, 2, 1], help="wave counts to compare with --hidden")
    ap.add_argument("--hazard-scale", type=float, default=1.0, help="multiply every hazard (below 1: harder games)")
    args = ap.parse_args()
    runs = load_runs(args.bench_glob)
    nlev, hazard, mean_eff = fit(runs)
    hazard = {k: v * args.hazard_scale for k, v in hazard.items()}
    print(f"{len(runs)} stock-cap runs; {len(nlev)} games; levels with a solve: {sum(1 for k in mean_eff)}")
    rng = random.Random(args.seed)
    if args.hidden:
        hidden_waves(args, rng, nlev, hazard, mean_eff)
        return
    arms = [("first-come (control)", 0.0, 0.0), ("step 0.5", 0.5, 0.0), ("step 1", 1.0, 0.0), ("step 1, fade 45", 1.0, 45.0),
            ("step 2", 2.0, 0.0), ("step 2, fade 45", 2.0, 45.0), ("step 4, fade 45", 4.0, 45.0)]
    totals = {a[0]: [] for a in arms}
    for _ in range(args.draws):
        needs = {g: [rng.expovariate(hazard.get((g, k), 0.1 / 5.0)) for k in range(n)] for g, n in nlev.items()}
        effs = {g: [mean_eff.get((g, k), 0.8) for k in range(n)] for g, n in nlev.items()}
        for name, step, fade in arms:
            totals[name].append(play(needs, nlev, effs, step, fade, args.s0))
    base = totals[arms[0][0]]
    for name, _, _ in arms:
        v = totals[name]
        diff = [a - b for a, b in zip(v, base)]
        m = sum(v) / len(v)
        md = sum(diff) / len(diff)
        sd = math.sqrt(sum((d - md) ** 2 for d in diff) / max(1, len(diff) - 1))
        print(f"{name:22s} mean {m:5.2f}  paired diff {md:+.2f} (sd {sd:.2f}, P(diff>0) {sum(d > 0 for d in diff) / len(diff):.0%})")


def hidden_waves(args, rng, nlev, hazard, mean_eff) -> None:
    """N games in W sequential waves of N/W concurrent games, each wave given 528 / W minutes (the Duck's 4 x 132)."""
    publics = list(nlev)
    arms = [("first-come", 0.0, 0.0), ("P21 step 2, fade 45", 2.0, 45.0)]
    out = {(w, a[0]): [] for w in args.waves for a in arms}
    for _ in range(args.draws):
        picks = [rng.choice(publics) for _ in range(args.hidden)]
        needs = {f"{g}#{i}": [rng.expovariate(hazard.get((g, k), 0.02)) for k in range(nlev[g])]
                 for i, g in enumerate(picks)}
        levs = {f"{g}#{i}": nlev[g] for i, g in enumerate(picks)}
        effs = {f"{g}#{i}": [mean_eff.get((g, k), 0.8) for k in range(nlev[g])] for i, g in enumerate(picks)}
        keys = list(needs)
        for w in args.waves:
            size = -(-len(keys) // w)
            for name, step, fade in arms:
                total = 0.0
                for j in range(w):
                    wave = keys[j * size:(j + 1) * size]
                    if not wave:
                        continue
                    sub = {k: needs[k] for k in wave}
                    total += play(sub, {k: levs[k] for k in wave}, {k: effs[k] for k in wave}, step, fade, args.s0,
                                  cap_min=CAP_MIN * 4 / w) * len(wave)
                out[(w, name)].append(total / len(keys))
    for w in args.waves:
        for name, _, _ in arms:
            v = out[(w, name)]
            print(f"{w} wave(s) of {-(-args.hidden // w)}: {name:22s} mean {sum(v) / len(v):5.2f}  vs first-come 4 waves "
                  f"{sum(a - b for a, b in zip(v, out[(4, arms[0][0])])) / len(v):+.2f}" if (4, arms[0][0]) in out else "")


if __name__ == "__main__":
    sys.exit(main())
