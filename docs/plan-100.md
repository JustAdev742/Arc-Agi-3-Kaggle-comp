# Plan for 100% RHAE (design only; not implemented; written 2026-09-16)

Status: a design, requested by the owner, for how this system could in principle reach 100% on the
hidden set. Written before the first model run; the measured state since then is in `docs/research_log.md`
and `docs/status.md` (best dev run 1.37 RHAE on 2026-09-16, public leaderboard leader 18.81). Every
mechanism below is a hypothesis to be measured with the discipline in CLAUDE.md, in the order given in
section 6. Items 1-4 of section 6 have been built in some form (perception, entity tracker, rule DSL,
verified execution, learning memory, level-boundary consolidation); see lesson 0013 for the external
reference point (Tycho) and what was ported from it.
`docs/research/road-to-100.md` (2026-09-17) is the evidence survey that ranks the next steps.

## 1. What 100% means, exactly

From the toolkit scorer (`arc_agi/scorecard.py`, parity-tested in `tests/test_scoring.py`):

- Level score `S_l = min((h_l / a_l)^2, 1.15)`, where `h_l` is the upper-median best human action count on
  first exposure and `a_l` ours (RESET included).
- Game score `E = min(sum_l l * S_l / sum_l l, fraction_of_level_weight_completed)`; total = mean over games.

So 100% requires, for every hidden game: every level completed, and the level-weighted mean of `S_l`
at or above 1.0. The 1.15 cap gives a little slack: a level played at 90% efficiency (0.81) can be covered
by a heavier level played 8% better than the human. There is no slack for an unfinished level, a crash, a
timeout, or a game whose mechanics we never infer. The human baseline already contains human exploration,
so "human efficiency" means: explore at least as cheaply as a good human, then execute optimally.

Consequences that shape everything else:

1. Actions are the only cost; compute is free until the 9-hour wall clock binds. Spend compute lavishly
   (simulation, search, verification) to save single actions.
2. Later levels weigh more and are reached only through earlier ones, so knowledge must transfer across
   levels within a game; the levels where humans are most efficient (they have learned the game) are the
   ones where a memoryless agent loses most.
3. A wrong action is never free; an informative action is only worth taking if surviving hypotheses
   disagree about its outcome.

## 2. Why systems plateau today (failure taxonomy -> mechanism)

Public write-ups (Duck, 2nd/3rd place) and our own two baselines point at the same failures. Each maps to a
mechanism in section 3.

| Failure category (docs/ARC-AGI-3_Research_Vision.md §20) | What happens | Mechanism |
|---|---|---|
| exploration | actions spent on probes that no hypothesis needed | 3.3 experiment selection |
| mechanics inference | rules guessed from one transition, never checked | 3.2 executable world model + 3.4 falsification |
| goal inference | wrong target chased for hundreds of actions | 3.5 goal hypotheses verified by level completion |
| planning / action selection | greedy stepwise play instead of a searched plan | 3.6 search in the model |
| efficiency | correct but long paths; repeated no-op actions | 3.6 + 3.7 budget controller + no-op memory |
| state tracking / perception | HUD bars mistaken for objects, avatar misidentified | 3.1 exact perception + entity tracking |
| memory | level k+1 restarted from zero | 3.8 within-game transfer |
| timeout / infrastructure | games unfinished, servers dead | 3.9 scheduler + fallbacks (exists today) |
| model capability | the model cannot write the program the game needs | 4.x mechanic library, training, stronger base model |

## 3. Architecture: a research organism, made concrete

Only the final step (3.6/3.7) sends actions. Everything else runs in the sandbox against exact data.

### 3.1 Exact perception and entity tracking (exists in part: `arc3/perception.py`)
- Logical grid via upscale detection; connected components with shape hashes; diffs; moved-object matching.
- Entity tracker across frames: persistent ids, motion vectors, appearance/disappearance events.
- Avatar detection as a statistic: the entity whose displacement is best explained by the last actions.
- HUD detection: edge strips that change monotonically with step count are excluded from goal hypotheses.
- Everything is code; the model reads summaries, never re-derives them.

### 3.2 Executable world model (EWM) by program synthesis (the central mechanism)

Status 2026-09-16: `set_model(predict)` (per-action verification, batch stop on mismatch, stats every turn) and
`verify_model(predict)` (replay of the level's full transition log with counter-examples) exist in `arc3/sandbox.py`.
Not yet built: the hypothesis ensemble with survival tracking and the confidence signal; not yet measured.
- The agent writes `predict(state, action) -> state'` as Python over the entity representation, not over
  raw pixels. Proposals come from the model; the harness verifies each proposal against the *entire*
  transition log of the level (and the game) and returns counter-examples (step, predicted vs actual).
- Keep an ensemble of surviving candidate programs (competing hypotheses). A hypothesis dies on its first
  counter-example. Confidence = number of verified transitions and agreement among survivors.
- Prediction is checked after every real action; a mismatch is an event that stops execution and triggers
  revision. The system never continues on a falsified model (Vision §5).
- Model class: a library of ARCEngine-native mechanics (sprites with masks, blocking modes, pushing,
  toggles, teleports, counters, keys/doors, colour matching, rotation, gravity, timers, camera/levels). The
  hidden games are built with the same engine, so this library is the strongest prior available. The
  synthesizer composes library pieces before writing bespoke code (section 4.2).

### 3.3 Active experiment selection
- Candidate probes are enumerated from perception (each legal key, each distinct click target by entity).
- Each probe's value = disagreement among surviving hypotheses about its outcome (expected information
  gain), minus its action cost and risk (irreversibility, game over). Probes whose outcome all survivors
  agree on are never taken for information.
- Cheap-first ordering: single keys before clicks, reversible before irreversible, near-avatar before far.
- Stop exploring when the survivors agree on everything reachable by the current plan (Vision §9).

### 3.4 Falsification and self-verification passes
- A falsifier pass re-runs every surviving program on the full log after each change and hunts for
  contradictions between the model and the goal hypotheses. Implemented first as a prompt role on the same
  served model (prefix-cached, cheap), later as the council's specialist model if it measures better.
- A "replay checker" replays the recorded actions through the EWM offline and diffs frame by frame.

### 3.5 Goal inference as competing hypotheses
- Candidate goals from structure: reach/touch/align entities, match colours or patterns, fill counters,
  remove all of a class, reach a state seen in a "target" panel. Candidates are ranked by evidence and
  tested by cheap partial progress (a counter moving, an entity changing state).
- The first `levels_completed` increment is ground truth: the transition that completed level 1 is mined
  for the win condition, which becomes the prior for every later level of that game.

### 3.6 Planning by search in the EWM
- Once the EWM is trusted (all survivors agree on the plan's transitions), search (BFS/A*/beam/IDA*) for
  the shortest action sequence to the inferred goal. Execute the plan with per-step prediction checks.
- Cost model includes RESET (1 action) so "reset and replay the known optimal path" is chosen when it
  beats correcting from a bad state.
- Plans are batched into one `act([...])` call: many actions per model call keeps seconds per action low.

### 3.7 Efficiency controller (the RHAE-aware budget)
- The controller knows the scoring formula and the level index. Before each non-plan action it estimates
  the marginal score change: information now vs actions later, weighted by remaining levels. It caps
  exploration per level at a fraction of the expected human count for that level type (calibrated on the
  public games) and raises the cap only when the model is still wrong.
- No-op memory: (state hash, action) pairs that changed nothing are never retried in the same state.
- Stagnation recovery: after N actions without new information, run the strategic reset of Vision §10
  (summarise evidence, list hidden assumptions, regenerate hypotheses, run one discriminating experiment).

### 3.8 Within-game transfer, then cross-game skills
- The EWM, entity classes, goal prior, and no-op memory persist across levels of a game. A new level starts
  with the previous model as the leading hypothesis; only layout is re-perceived. This is where the heavy
  late-level weights are won.
- Cross-game: a versioned skill store (Vision §6, §8) with trigger, procedure, evidence, successes and
  failures; a skill is promoted only after it helps on more than one game, and demoted on contradiction.

### 3.9 Scheduling and time safety (exists in part: governor, fallbacks)
- All hidden games run concurrently on one server. A scheduler allocates model calls to the games with the
  highest expected marginal score (level weight x probability of completion x time left) and parks games
  that are stuck after a strategic reset, revisiting them if time remains.
- Hard caps everywhere: sandbox time, model call time, per-game deadline, reserve before the 9-hour limit.
  A crash in any component degrades to the explorer, never to an idle game.

### 3.10 Council of specialists (the owner's 6 + 1 design)
- Roles: perception, mechanics, explorer, goal analyst, falsifier, planner; one coordinator acts.
- First as prompts on the coordinator's model (prefix caching makes this nearly free in wall-clock), then
  with Qwen3-VL-8B-NVFP4 serving the six roles concurrently on a second server. Kept only where it beats
  the single-model arm on dev and holds on val at acceptable seconds per action. Already implemented as an
  ablation arm (`arc3/agents/council.py`); not yet measured.

## 4. Data, learning and generation

### 4.1 Human replays (published for the 25 public games)
- Calibrate 3.3 and 3.7: what good humans probe first, how many actions they spend before the first
  goal-directed sequence, per game type (click / keyboard / mixed).
- Train small components: a click-target prior and an action prior conditioned on the entity summary,
  used to order probes (Vision §6; CLAUDE.md item 6). Never as the policy itself.

### 4.2 Mechanic library from the engine and the public games
- Read ARCEngine (`arcengine` package) and the 25 public game sources to enumerate the mechanics actually
  used (movement, blocking, pushing, merging/detaching, toggles, level transitions, UI displays).
- Encode each as a parameterised EWM component with a verification test; the synthesizer composes them.
- Guard against overfitting: the val split (6 games) is never used to write library components.

### 4.3 Procedural generator and self-play (CLAUDE.md item 7, only if public games stop giving signal)
- Generate new games from the mechanic library with the real engine; reject trivially solvable, unsolvable
  and ambiguous ones; the optimal solver's action count stands in for the human baseline.
- Use: stress tests of the synthesis loop, trajectories for the small learned components, and an unbounded
  dev set. Own generated games are never a holdout.

### 4.4 Base model
- Qwen3.8-27B (FP8 primary, NVFP4 A/B) as specified. The plan does not depend on a specific model, but
  the synthesis loop's ceiling is the model's ability to write correct simulators from few examples;
  a stronger model or a code-specialised fine-tune on generated (game, program) pairs is the escalation.

## 5. Budget arithmetic (9 hours, all games concurrent)

Assume N hidden games (unknown; plan for 50), ~8 levels each, ~60 actions per level at human
efficiency: ~480 real actions per game, ~24,000 in total. In 9 hours that is 1.35 s per action on
average across the fleet, or about 60 s per action per game if all run in parallel. A plan-driven agent
takes most actions in batched sequences from search, so model calls per action fall well below one.
Throughput needed from the server: measured, not assumed (diag v4 and exp-003 will give the first
numbers). Speculative decoding (MTP), prefix caching, FP8 KV and short reasoning budgets for routine calls
are the levers; the NVFP4 A/B trades accuracy for throughput and is decided by RHAE, not tokens/s.

## 6. Roadmap with gates (each gate is a measured number on dev and val)

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| 0 (now) | first real action; exp-003 control arm (single 27B REPL) | any level solved by the model; seconds per action known |
| 1 | EWM synthesis loop with counter-example verification (3.2), replay checker (3.4) | actions per solved level on dev fall vs exp-003; no regression on val |
| 2 | experiment selection + efficiency controller + no-op memory (3.3, 3.7) | fewer probing actions per level; more levels reached in the same time |
| 3 | within-game transfer (3.8) | later-level efficiency approaches early-level efficiency |
| 4 | council roles as prompts, then VL-8B specialists (3.10) | beats the phase-3 arm on dev, holds on val, fits time |
| 5 | mechanic library + replay-calibrated priors (4.1, 4.2) | fewer hypotheses to test per level; faster convergence |
| 6 | generator + learned components (4.3, 4.1) only if dev signal is exhausted | gains on generated games transfer to public games |
| Milestone 2 (2026-09-30) | best measured arm, notebook made public under an open-source license | owner's OK to publish |
| Final (2026-11-02) | best measured arm, one clean 9-hour run | validated end to end on Kaggle before the last day |

## 7. What would have to be true, and what would falsify the plan

- Every hidden game's mechanics are expressible in the model class the agent can synthesise. If a game
  needs rules the model cannot write from evidence, that game is lost (partial credit only).
- Goals are inferable from cheap partial progress or the level-1 completion. Games whose goals are only
  discoverable by expensive exploration cap efficiency below human.
- The server sustains the required actions per second with all games concurrent. If not, the scheduler
  must sacrifice games deliberately rather than time out on all of them.
- Falsifier: if phase 1 does not reduce actions per solved level on dev while holding val, the central
  mechanism is wrong for this benchmark and the plan reverts to improving the single-model harness.

## 8. Mapping to the owner's ideas

- The models: Qwen3.8-27B (FP8 primary, NVIDIA NVFP4 A/B), Qwen3-VL-8B-NVFP4 specialists: sections 3.10, 4.4.
- Six specialists + coordinator: 3.10 (built as `council`, unmeasured).
- Research-organism loop observe -> hypothesise -> test -> model -> falsify -> plan -> act -> learn: 3.2 to 3.7.
- Memory (episodic, semantic, procedural), skill consolidation, post-mortems: 3.8, docs/lessons, docs/postmortems.
- Stagnation recovery and long-runtime reasoning: 3.7, 3.9.
- Procedural environment and human calibration: 4.3, deferred by the brief's own ordering.
- Non-negotiable principle: the architecture is discovered by the gates in section 6, not assumed.

---

# Part II: build specification (written 2026-09-16 after exp-003, the first measured control arm)

Part I says what 100% requires and which mechanisms address which failures. This part says what to build, in
what order, with interfaces, gates and dates. It is the working plan; each item is still a hypothesis.

## II.1 The four facts the design rests on

1. **The engine is known.** Every game is an `ARCBaseGame` (we have `arcengine` and 25 public games): sprites with
   pixel masks and colours, levels as sprite lists, a camera with integer upscale, discrete moves (`try_move`),
   bounding-box or pixel-perfect blocking, and `next_level()` fired by a predicate in `step()`. The mechanics of a
   hidden game are therefore drawn from a structured space, not an open one.
2. **Entities are the right state.** Rules in these games are statements about sprites: "the avatar moves one tile
   per key unless a wall blocks", "touching a key removes it and opens the door", "clicking a tile toggles it".
   A model over tracked entities is small, verifiable and searchable; a model over pixels is none of those.
3. **Compute is nearly free, actions are not.** In the competition every game runs for the full 9 hours in
   parallel. Measured aggregate generation is ~350 tok/s at 8 concurrent (exp-003) and rises with batch size; that is
   hundreds of model calls per game, 15-40x what the 20-minute dev runs allow. RHAE never sees compute. So: replay,
   verify, search and simulate as much as needed; probe the environment only when surviving hypotheses disagree.
4. **The model is weakest where code is strongest.** Pixel-precise geometry, bookkeeping across hundreds of steps,
   exhaustive search. The harness must own those; the model proposes rules and goals and writes code for the rest.

## II.2 Components, interfaces, and where they run

All of this runs inside the sandbox as variables and functions the model can call, so the REPL stays the single
interface and every piece is also usable by hand for debugging.

### A. Entity perception and tracking (`arc3/entities.py`)  <- build first
- `tile(grid) -> k`: the logical cell size (engine upscale x sprite grid), from component sizes and movement steps.
- `entities(grid) -> [Entity]`: connected components merged into multi-colour sprites when they move together
  (co-movement learned from the transition log); each has id, mask, colours, bbox, tile position.
- `track(before, after) -> events`: persistent ids across frames (shape hash + proximity + motion consistency);
  events: moved(id, dx, dy), appeared, disappeared, recoloured, reshaped. This is the symbolic transition.
- `roles()`: static (never changes: walls, floor), dynamic, HUD (edge strip with monotone change), avatar (the
  entity whose displacement correlates with key actions), candidate targets (unique colour/shape, referenced by a
  panel).
- Exposed in the REPL: `ents()`, `events()` (last transition), `event_log()` (level), `avatar()`, `hud()`.
- Gate (exp-005): with only this added to the prompt, actions per solved level on dev fall vs exp-003c and the
  model's first probe is a key when an avatar exists (measured from traces).

### B. Rule language and verifier (`arc3/dsl.py`)
- Rule types, parameterised (small finite parameter spaces so code can search them):
  `Move(cls, keymap, step, blocked_by, wrap)`, `Push(mover, pushable, chain)`, `OnOverlap(a, b, effect)`,
  `Toggle(cls, trigger)`, `Drift(cls, dx, dy)`, `ClickEffect(cls, effect)`, `Counter(hud, event)`,
  `Win(predicate)` with predicates `on(a, b)`, `none_left(cls)`, `count(cls) == n`, `matches(reference)`,
  `aligned(cls)`.
- `simulate(state, action, rules) -> state'` and `render(state) -> grid` (entities pasted back on the static layer).
- `verify(rules) -> counter-examples` over the level's symbolic transition log (entity-level diffs, then pixels).
- `fit(rule_type, log) -> [consistent parameterisations]`: code enumerates parameters; the model only chooses types.
- Hypothesis manager: alive rule-sets, survival on every real action (extends today's `set_models`).
- Gate (exp-006): on the public games, the fraction of levels whose full transition log is explained by a rule-set
  the model + fit() can produce within N calls. This number is the coverage estimate for the hidden set.

### C. Experiment selection and planning (`arc3/planner.py`)
- `probe_value(action)`: number of distinct predicted outcomes across alive rule-sets, minus cost and risk
  (irreversible effects, game over, RESET); `best_probe()`.
- `plan(goal_predicate) -> actions` by BFS/A* over simulated states under the surviving rule-set; execute with
  per-step verification (already in `act()`); replan on mismatch.
- `goal_candidates()`: enumerated predicates over the entity set ranked by priors; `confirm_goal()` from the
  level-completion transition; carried to the next level.
- Gate (exp-007): fewer probing actions per level and fewer RESETs than exp-005 on dev; val holds.

### D. Controller and memory (`arc3/agents/repl_agent.py` extensions)
- RHAE-aware budget: expected human count per level type (calibrated on the public games) sets an exploration
  cap; stagnation reset after N no-information actions; no-op memory.
- Cross-level carry: rules, entity classes, goal predicate, no-op memory survive level changes in the sandbox.
- Council roles (proposer, goal analyst, falsifier, planner) as prompts; VL-8B specialists as the A/B.
- Gate (exp-008/9): later-level efficiency approaches level-1 efficiency; council beats single-model or is dropped.

### E. Data and learning (`tools/`, only after A-C exist)
- Mechanic catalogue from the 25 public game sources: list every mechanic and check the DSL expresses it.
- Human replays: probing patterns, actions before first goal-directed move, click targets -> priors for C.
- Procedural generator (ARCEngine + DSL): unlimited (game, rules) pairs to stress A-C and, if the model's rule
  proposals are the bottleneck, to fine-tune a rule-induction LoRA on the local RTX PRO 6000.

### F. Scheduler (Kaggle adapter)
- All games concurrent; model calls allocated by expected marginal score (level weight x completion probability x
  time left); stuck games parked after a strategic reset and revisited.

## II.3 Order of work and dates

| When | Build | Measure |
|---|---|---|
| Sep 17-19 | A (entities, tracking, roles) in the REPL; exp-004 thinking ablation | exp-005 vs exp-003c (queued on Kaggle 09-16) |
| Sep 20-23 | B (DSL, verify, fit, hypothesis manager) — BUILT 09-16 (`arc3/dsl.py`) | exp-006a code-only coverage 0.43 mean over 25 games (research log); dev/val on the model: exp-007 |
| Sep 24-27 | C (probe selection, planner, goal candidates, carry-over) — planner + goal candidates/hints built 09-16; probe selection and carry-over open | exp-007 (running 09-16) |
| Sep 28-30 | Milestone 2 candidate: best arm, Save & Run All on RTX, public copy (owner OK) | one clean 9 h-style run |
| Oct 1-12 | D (controller, council A/B, NVFP4 A/B), E1-E2 (catalogue, replays) | exp-008..011 |
| Oct 13-24 | E3 (generator, LoRA if warranted), F (scheduler); freeze by Oct 26 | full-length validation runs |
| Oct 27-Nov 1 | final validation runs, submission | |

## II.4 What would make me abandon parts of this

- A gives no gain (exp-005 flat): the model already sees enough; the bottleneck is reasoning, go to B anyway.
- B's coverage on public games < 60%: the DSL is too narrow; keep verify/fit but route more mechanics to bespoke
  Python, and move the fine-tune (E3) earlier.
- C does not cut probing actions: hypotheses are not converging; invest in priors from replays (E2) and the
  falsifier role.
- Later-level efficiency stays poor after carry-over: goals change more than mechanics; invest in goal inference.

## II.5 Where the 100% could still fail even if everything above works

- Hidden games with mechanics outside the DSL and beyond what the model can write from few examples.
- Goals only discoverable by long exploration (humans would also score badly; the baseline reflects that, but our
  cap is 1.15 so we cannot compensate elsewhere).
- Concurrency-induced latency in the 9-hour run making calls per game too few for the search-heavy loop.
- Any infrastructure failure on the one competition run.
