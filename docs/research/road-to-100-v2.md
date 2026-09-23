# Road to 100 percent, part II: the scoring lever, optimal-play headroom and an engineered route (research, 2026-09-23)

Requested by the owner on 2026-09-23 ("research ways to get 100% ... find ways and engineer your own ways").
Part I (`docs/research/road-to-100.md`, 2026-09-17) surveyed the field and ranked model and harness experiments
that need the GPU. This part measures, game by game, what 100 percent actually requires, and builds and measures on
CPU the first pieces of a route that uses the model only where code cannot do the job. Every number points at a run
in `runs/` or a command in section 9; the research log has one entry per experiment.

## 0. Outcome first

- **100 percent does not mean "no mistakes". It means "learn the game on level 1, then play every later level with
  slightly fewer actions than a human".** Two measured facts make that concrete:
  1. **Level 1 is nearly free.** If every level from 2 on uses at most 95 to 99 percent of the human action count
     (95 for a 4-level game, 98 for 7 levels, 99 for 9 or more), level 1 may take any number of actions and the game
     still scores 100 (section 1; checked with the toolkit-parity scorer).
  2. **Optimal play is far below the human count.** An exact search inside the real engine (a research instrument;
     the agent never has the engine) finished 32 dev levels of 13 games: the optimum is a **median 41 percent of the
     human action count** (mean 46, range 9 to 100). 30 of the 32 levels leave room for the 115 cap; the median
     level leaves **21.5 actions of slack** between the optimum and the human count (section 2).
  So a system that knows a game's rules and goal can afford about twenty wasted actions per level (experiments,
  mistakes) and still score 100.
- **The games are more regular than they look** (census of the 19 dev games' source, section 3): the kind of win
  condition never changes between a game's levels (19 of 19), the target is drawn on screen (17 of 19), five
  relational kinds cover almost everything, and every game has a per-attempt budget drawn as an edge bar.
- **Built and measured this cycle, all without a model, on CPU:**
  - exp-027: the explorer ignores the budget bar when it keys states. Dev levels 6 → 15; level 1 cleared on 9 of 19
    games in at most 90 s of CPU each.
  - exp-028: goal induction by contrast (the winning frame against every non-winning state the explorer visited).
    [filled below]
  - exp-029: search steered by the induced goal. [filled below]
  - A harness bug fixed: the REPL agent archived the wrong frame as a level's winning frame on animated wins (3 of
    15 explorer-solved levels; the new rule is exact on 15 of 15).
- **Where we stand:** the submitted agent scores about 1 on dev; the Kaggle leader is about 19; the best verified
  system on hidden games (GPT-6 Astra) is 62.7 with unlimited compute. The route below decomposes "100" into parts
  that can each be measured; the parts that need a strong model are named as such (section 4.4).

## 1. The scoring lever: level 1 is nearly free

Game score = min(Σ l·S_l / Σ l, completed-weight fraction × 100), S_l = min((human/agent)² × 100, 115), weights =
level index (`arc3/scoring.py`, parity-tested against `arc_agi.scorecard`). With level 1 at a score near 0 and levels
2..n all at score S, the game reaches 100 when (W − 1)·S ≥ 100·W, W = n(n + 1)/2:

| Levels in the game | Score needed on levels 2..n | Max fraction of human actions on levels 2..n |
|---:|---:|---:|
| 2 | 150 | impossible (cap 115) |
| 3 | 120 | impossible (cap 115) |
| 4 | 111.1 | 0.949 |
| 5 | 107.1 | 0.966 |
| 6 | 105.0 | 0.976 |
| 7 | 103.7 | 0.982 |
| 8 | 102.9 | 0.986 |
| 9 | 102.3 | 0.989 |
| 10 | 101.9 | 0.991 |

Checked with `arc3.scoring.game_score`: 7 levels of baseline 100, level 1 at 10,000 actions, levels 2 to 7 at 94
actions each → 100.0; at 99 actions each → 98.4. The dev games have 6 to 9 levels. Consequences:

- Everything learned on level 1 is paid for by the later levels' margin, so level 1 is where to explore, test
  hypotheses and induce the goal, with no action budget beyond the game's own attempt budget.
- The lever only pays when every later level is completed below the human count. Until an agent can do that, level
  1 is worth what it scores (weight 1 of 21 to 45 per game). Today that is most of our score: in the three champion
  dev runs level 1 alone gives 84, 79 and 100 percent of the summed game scores (`runs/kaggle-repl-dev-011`, `-011b`,
  `-011c`), and the model solves those levels at or below the human count (ar25 in 19 actions against 32, tn36 in 14
  against 32, lp85 in 10 against 17), where the explorer needs 52 to 984. Explore-first would lower today's score;
  it becomes the right trade only once levels 2+ are won (section 7, item 2).
- It cannot be gamed with a second play: the scorer keeps the best play, but competition mode refuses a second play
  and a full reset (`arc_agi/api.py`; lesson 0001), and `arc3/env.py` mirrors that locally.

## 2. The headroom: how short can a level be?

`scripts/oracle_bfs.py` runs breadth-first search inside the real engine (deep copies of the game object, about
12 ms each). Keys: the board without UI overlays, the game's hidden state, and the full render with the budget bar
masked (a selection shown only in the HUD is otherwise merged away). Actions: the engine's valid actions, plus one
click per visible object when a game tags no clickable sprite (bp35, cd82, ft09, s5i5, su15 and others). Budget:
240 s or 60,000 nodes per level; levels are solved in sequence. Dev split only (`runs/oracle-bfs/summary.json`; the
first version, which also ran the validation games, is superseded: `runs/oracle-bfs-v1`).

| Game | Levels | Solved levels: optimal / human (ratio) | First unsolved level: optimum at least |
|---|---:|---|---|
| ar25 | 8 | L1 15/32 (0.47); L2 11/50 (0.22) | L3 ≥ 8 (human 75) |
| bp35 | 9 | none (82 candidate clicks per state) | L1 ≥ 5 (human 21); a code-derived solution takes 15 |
| cd82 | 6 | L1 5/55 (0.09); L2 6/8 (0.75) | L3 ≥ 7 (human 41) |
| dc22 | 6 | L1 20/59 (0.34); L2 42/102 (0.41); L3 45/67 (0.67); L4 62/98 (0.63) | L5 ≥ 8 (human 324) |
| ft09 | 6 | L1 4/43 (0.09) | L2 ≥ 6 (human 12) |
| ka59 | 7 | L1 11/28 (0.39) | L2 ≥ 7 (human 109) |
| lp85 | 8 | L1 5/17 (0.29) | L2 ≥ 6 (human 38) |
| ls20 | 7 | L1 13/22 (0.59); L2 45/123 (0.37); L3 39/73 (0.53) | L4 ≥ 38 (human 84) |
| m0r0 | 6 | L1 15/30 (0.50); L2 23/111 (0.21) | L3 ≥ 8 (human 203) |
| re86 | 8 | none | L1 ≥ 14 (human 26) |
| s5i5 | 8 | L1 13/20 (0.65) | L2 ≥ 14 (human 89) |
| sb26 | 8 | none | L1 ≥ 7 (human 18); a code-derived solution takes 9 |
| sk48 | 8 | L1 14/61 (0.23) | L2 ≥ 15 (human 177) |
| su15 | 9 | L1 7/22 (0.32) | L2 ≥ 9 (human 42) |
| tn36 | 7 | none | L1 ≥ 6 (human 32) |
| tr87 | 6 | none | L1 ≥ 9 (human 54); a code-derived solution takes 15 |
| tu93 | 9 | all nine: 18/19, 10/16, 19/34, 17/42, 29/123, 28/80, 14/14, 21/23, 29/111 | none |
| vc33 | 7 | L1 3/7 (0.43); L2 7/18 (0.39); L3 23/44 (0.52); L4 21/61 (0.34) | L5 ≥ 20 (human 131) |
| wa30 | 9 | none | L1 ≥ 12 (human 71) |

Over the 32 solved levels: optimum / human median 0.41, mean 0.46, minimum 0.09 (cd82 L1, ft09 L1), maximum 1.00
(tu93 L7, where the human was optimal). Level 1: median 0.39 over 13; levels 2+: median 0.41 over 19. Ratio ≤ 0.93
(the 115 cap reachable): 30 of 32. The "code-derived solution" entries are the census agents' level-1 solutions,
checked in the engine (`win-conditions-dev.md`): upper bounds on the optimum.

Caveats: object clicks are one click per visible component, so a level that needs a click between objects (su15
pulls toward the click point) is solved by a restricted alphabet and the value is an upper bound; the search keys
ignore the step counter, which cannot make a path shorter; the unsolved rows are search budget, not impossibility.
The later, harder levels are under-represented (the search stops at the first unsolved level), so the median ratio
is for the levels a 4-minute search can finish.

What it means: a human's first-exposure count includes their exploration and mistakes (the census found baselines
above the per-attempt budget on 8 of 19 games, so some include failed attempts). Planning in a correct model beats it
by a factor of about 2.4 on the median level. The slack is the budget for learning the new mechanics each level adds.

## 3. What the games are made of

`docs/research/win-conditions-dev.md` (source census of the 19 dev games; lesson 0016). In short: place pieces in
slots (11 games), make a region equal a shown reference (10), bring a sprite to a marker (7), align (1), count (1),
per-tile rules (1); a game can carry two. The kind never changes between levels; later levels add mechanics. The
goal is drawn in 17 of 19. The win check is gated by action kind in at least 5 games. Every game has a per-attempt
budget bar. Multi-target games need every target, which level 1 (one target) cannot distinguish from "some target":
hence the universal goal kinds added to `arc3/dsl.py` this cycle (`every_<relation>`, opt-in).

## 4. An engineered route: learn on level 1, plan the rest

The route follows from sections 1 to 3. Each component is listed with what exists, what was measured, and what is
missing. Components 4.1 to 4.3 and 4.5 are code; 4.4 is where a model is required.

### 4.1 Exact perception of the frame (built)
- Budget bars masked from state keys by volatility (exp-027, kept).
- The winning frame of a level is `perception.terminal_layer(layers, before)`: the layer before the level switch
  when the step ends with a jump, else the last layer. Exact on 15 of 15 explorer-solved levels; the old rule
  (`layers[0]`) was wrong on 3 (cd82's 16-frame pour twice, tu93's 9-frame move) and fed the REPL agent's level
  archive the board before the winning move.

### 4.2 Level 1 by exploration (built; 9 of 19 dev games)
The novelty explorer with masked keys clears level 1 of ar25, cd82, ft09, lp85, ls20, m0r0, s5i5, tu93 and vc33 within
20,000 actions and 90 s of CPU (exp-027). It fails on games with deep level-1 solutions or wide click alphabets
(bp35, dc22, ka59, re86, sb26, sk48, su15, tn36, tr87, wa30). In the full system the model takes over level 1 where the
explorer stalls, with the explorer's effect statistics as its starting knowledge.

### 4.3 The goal from level 1, by contrast (exp-028, measured)
[filled below]

### 4.4 The mechanics of each new level (needs a model)
Each level adds one or two mechanics (a changer tile, a crane, bombs, enemies). The route budgets about 20 actions
of slack per level to learn them. A fixed rule library covers about half the entity events on public games
(exp-006a: mean 0.50), so the new mechanics have to be written as code by a model: the REPL harness's world-model
loop (the model writes a simulator, the harness checks it against every recorded transition and returns the first
mismatch). Part I's evidence is that this step is where open 27B-class models fail and frontier models succeed;
it is the component that the model A/B (Part I, section 5) and any fine-tuning must target.

### 4.5 Planning at below-human cost (built as code; measured with exp-029's model-free proxy)
With a model of the mechanics and a goal predicate, breadth-first or A* search in the model (`dsl.plan`, the
sandbox's `plan_rules`) produces the optimal plan; section 2 shows the optimum leaves a median of 21 actions of
slack. [exp-029 filled below]

### 4.6 Time
Level-1 exploration costs seconds of CPU per game, not model calls; the model's 9-hour budget moves to levels 2+.

## 5. The arithmetic of 100 for one game

Take a 7-level game with the dev median ratio: every later level's optimum is 0.41 of the human count, and the
game needs levels 2 to 7 at 0.98 or less. The per-level budget for learning that level's new mechanics and for
mistakes is then 0.98 − 0.41 = 0.57 of the human count, about 35 actions on a 60-action level. Level 1 has no
budget beyond the game's own attempt budget. A single unfinished level caps the game at the completed weight, so
reliability (finishing every level) matters more than polish.

## 6. Ideas examined

| Idea | Status | Evidence |
|---|---|---|
| Explore freely, then replay the solution in a second play | closed | competition mode refuses a second play and full resets (lesson 0001) |
| Level 1 as a free learning level | open, measured | section 1; needs every later level below the human count to pay |
| Mask the budget bar in state keys | kept | exp-027: 6 → 15 dev levels |
| Goal induction by contrast from exploration data | measured | exp-028 |
| Universal ("every target") goal kinds | built, opt-in | census; `dsl.forall_distance`, tests |
| Search steered by the induced goal, without a mechanics model | measured | exp-029 |
| Colour-free goal templates (the same kind on different colours per level) | proposed | exp-028: vc33's goal is "aligned" on every level, on a different colour each time |
| Reach goals as "cover the marker" with a distance from the mover | proposed | tu93, dc22: the induced goal is none_left(marker colour), which has no gradient |
| Oracle trajectories as training data (optimal action sequences from the engine search) | proposed, not started | 32 optimal level solutions exist; useful for a click-target or action ranker (CLAUDE.md item 6); overfitting risk: the hidden games differ |
| Procedural level variants of the public games for training | proposed, not started | CLAUDE.md item 7; only when the public games stop giving signal |
| Human replay priors | blocked (data) | Part I, section 8 |

## 7. Ranked next experiments

1. **Colour-free goal templates and a mover-relative reach distance** (code, CPU). Gate: exp-028's transfer rate on
   games with two or more solved levels rises, and exp-029's steered explorer stops losing on ls20.
2. **Explorer-first hand-off in the REPL harness** (config flag; GPU): the explorer plays level 1 for a bounded budget,
   the model starts level 2 with the level-1 transition log, the winning frame and the induced goal candidates.
   Gate: levels 2+ solved on dev and actions per level-2+ below the champion's; level-1 score loss accepted only if
   levels 2+ gain more (section 1).
3. **Model challengers and the effort arm** (Part I, items 1 and 2): unchanged; mechanics inference (4.4) is the
   binding component and depends on the model.
4. **Grammar gaps from the census:** masked region equality (cd82, sk48), bracket slots as one target (lp85), merge
   counts (m0r0, su15), per-tile constraints read from clue tiles (ft09). Each is a goal kind the contrastive
   induction can then find.

## 8. Risks

- Overfitting the public set: the census and the oracle table are public-game evidence. Everything built from them
  is an engine-level prior (budget bars, goal kinds, quantifiers), not a per-game rule, and is measured on dev with
  validation held out.
- The optimum table covers the levels a short search can finish; harder, later levels may have less slack.
- Level 1 as free exploration lowers today's score until the later levels are won (section 1).

## 9. Reproduce

```
.venv/bin/python scripts/oracle_bfs.py --split dev --max-nodes 60000 --max-s 240 --workers 3 --out runs/oracle-bfs/summary.json
.venv/bin/python scripts/eval.py --agent explorer --split dev --time-per-game 90 --max-actions 20000 --config '{"mask_volatile": false}' --run-name exp027-explorer-mask-false
.venv/bin/python scripts/eval.py --agent explorer --split dev --time-per-game 90 --max-actions 20000 --run-name exp027-explorer-mask-true
.venv/bin/python scripts/goal_induction.py collect --source explorer --split dev --workers 3
.venv/bin/python scripts/goal_induction.py collect --source oracle --games dc22,vc33,tu93,ls20,ar25,cd82,m0r0 --workers 2
.venv/bin/python scripts/goal_induction.py analyze
.venv/bin/python scripts/eval.py --agent explorer --split dev --time-per-game 300 --max-actions 20000 --run-name exp029-explorer-control
.venv/bin/python scripts/eval.py --agent explorer --split dev --time-per-game 300 --max-actions 20000 --config '{"goal_directed": true}' --run-name exp029-explorer-goal
```
