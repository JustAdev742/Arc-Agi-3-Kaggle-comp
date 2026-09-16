# ARC-AGI-3 Research Vision

This document describes the long-term research vision for the ARC-AGI-3 agent. It is supplementary to `CLAUDE.md`.

`CLAUDE.md` is the authoritative engineering and competition document.

This document explains the broader idea, desired capabilities, research direction, and architectural possibilities. Nothing in this file should override the experimental discipline, competition requirements, safety constraints, or validation rules in `CLAUDE.md`.

---

# 1. LONG-TERM OBJECTIVE

Build a general-purpose interactive reasoning system that can learn unfamiliar environments from interaction rather than relying on memorized solutions.

The long-term research goal is an extremely high ARC-AGI-3 score, potentially 100%, but 100% is not assumed or guaranteed.

The system should become progressively better at:

- perception
- exploration
- mechanics discovery
- goal inference
- world modeling
- hypothesis formation
- hypothesis falsification
- planning
- action efficiency
- recovery from mistakes
- learning reusable skills
- transferring validated knowledge across environments

The central philosophy is:

> Observe → hypothesize → test → model → falsify → plan → act → learn → improve.

Do not optimize for architectural complexity.

Optimize for measurable generalization.

---

# 2. CENTRAL IDEA

The system should behave more like a group of researchers investigating an unknown environment than like a chatbot answering a puzzle.

It should be capable of saying:

- “I don't know.”
- “These are the two most plausible explanations.”
- “This experiment would distinguish them.”
- “Our previous model predicted X, but reality produced Y.”
- “That invalidates part of the current model.”
- “We learned something reusable from this mistake.”
- “The current plan is inefficient.”
- “We should gather more evidence before acting.”

Uncertainty should be explicit and useful.

---

# 3. MULTI-AGENT RESEARCH ARCHITECTURE

One candidate architecture is:

- six Qwen3-VL-8B specialist agents
- one Qwen3.8-27B senior reasoning/coordinator agent

This is an experimental branch, not a permanent requirement.

The six specialists are conceptually:

## Perception

Focus only on accurate state extraction.

Responsibilities:

- object identification
- object localization
- color and shape information
- relationships
- state changes
- frame differences
- uncertainty

Do not ask the model to perform calculations that deterministic code can perform exactly.

---

## Mechanics / Rule Discovery

Determine how the environment behaves.

Responsibilities:

- action → outcome relationships
- object interactions
- movement rules
- transformations
- hidden mechanics
- delayed effects
- dependencies

Maintain competing hypotheses rather than prematurely committing to one.

---

## Explorer

Treat interaction as experimentation.

Responsibilities:

- identify informative actions
- estimate information gain
- identify low-risk tests
- distinguish hypotheses
- avoid redundant experiments
- recognize when exploration has become unproductive

The explorer should optimize information gained per action rather than blindly trying to win immediately.

---

## Goal Analyst

Infer what the environment is asking the agent to achieve.

Maintain multiple candidate goal hypotheses when necessary.

Distinguish:

- immediate objectives
- intermediate objectives
- final objective

Update confidence as evidence accumulates.

---

## Falsifier / Critic

Actively attempt to prove the system wrong.

Responsibilities:

- identify assumptions
- find contradictions
- construct counterexamples
- test predictions
- attack the world model
- detect confirmation bias
- recommend experiments that discriminate between competing explanations

This agent should be rewarded for discovering that the current model is wrong.

A useful internal question is:

> “What observation would prove our current explanation false?”

---

## Planner / Efficiency

Use the current world model to construct candidate action sequences.

Responsibilities:

- action sequence search
- simulation
- shortest-path reasoning
- risk assessment
- recovery plans
- action-efficiency optimization

Prefer plans that achieve the goal with fewer actions when confidence is comparable.

---

# 4. SENIOR REASONER

The Qwen3.8-27B model should act as the senior reasoner rather than merely another independent agent.

It should receive structured information from the specialists, including:

- current state
- perception result
- mechanics hypotheses
- exploration results
- goal hypotheses
- falsification results
- world model
- candidate plans
- action history
- memory
- known failures

The senior reasoner should determine:

1. What is known?
2. What is uncertain?
3. Which explanation best fits all evidence?
4. What evidence contradicts it?
5. Should the system explore or execute?
6. Which experiment has the highest expected value?
7. Which action is currently best?
8. Can the action be performed more efficiently?
9. Should the current strategy be abandoned?

The senior reasoner is the final strategic authority before an environment action is committed.

---

# 5. EXECUTABLE WORLD MODEL

The system should maintain a programmatic model of the environment.

The model should represent:

- objects
- positions
- relationships
- state
- mechanics
- transitions
- goals
- constraints
- predictions
- uncertainty

It should support:

`state + action → predicted next state`

Candidate plans should be simulated against this model whenever possible before consuming real environment actions.

The model must also compare:

`predicted state vs actual state`

Any discrepancy should create an explicit model error.

The system must not silently continue as though its prediction was correct.

---

# 6. MEMORY

Memory should exist at multiple levels.

## Episodic memory

Store specific experiences.

Example:

- Game 12
- action 8
- hypothesis
- prediction
- actual result
- failure reason

## Semantic memory

Store general knowledge.

Example:

> Similar visual appearance does not imply identical interaction behavior.

## Procedural / skill memory

Store reusable strategies.

Example:

> Before treating an unknown object as an obstacle, test whether it is interactable using a safe reversible action.

Skills should contain:

- name
- trigger
- procedure
- evidence
- confidence
- successes
- failures
- environments tested
- whether the skill is general or game-specific

---

# 7. LEARNING FROM FAILURE

After significant failures, create a structured post-mortem.

The core question should literally be:

> WHAT DID WE LEARN FROM THIS MISTAKE?

Record:

- what we believed
- what actually happened
- which assumption was wrong
- evidence that exposed the mistake
- what cheaper experiment could have detected it earlier
- revised hypothesis
- whether the lesson is generalizable
- whether it should enter long-term memory

Do not convert every failure into a skill.

A single unusual event is an observation.

Repeated evidence can become a hypothesis.

Repeated cross-environment evidence can justify a reusable skill.

---

# 8. SKILL CONSOLIDATION

The system should gradually transform experience into reusable knowledge.

Use this distinction:

## Experience

“This happened in Game 18.”

## Hypothesis

“This may happen in other games.”

## Validated skill

“This strategy has successfully worked across multiple independent environments.”

## Deprecated skill

“This previously looked useful but has repeatedly failed under new evidence.”

Skills should therefore be editable, versioned, and capable of being invalidated.

This prevents the memory system from accumulating incorrect permanent rules.

---

# 9. ACTIVE EXPLORATION

Every possible exploratory action should be considered as an experiment.

Conceptually estimate:

`utility = information gain + progress value - action cost - risk`

The exact mathematical implementation should be determined experimentally.

Exploration should stop when:

- useful information is no longer being gained
- hypotheses have converged
- the world model predicts outcomes reliably
- a sufficiently confident plan exists

The system should then switch from exploration to execution/search.

---

# 10. STAGNATION RECOVERY

Long runtimes should be used for learning, not repetition.

If the system repeatedly:

- makes the same type of mistake
- gains no new information
- produces no useful new hypothesis
- makes no world-model improvement
- repeatedly executes equivalent plans

trigger a strategic reset.

Possible recovery steps:

1. summarize current evidence
2. identify hidden assumptions
3. ask specialists for independent reconsideration
4. generate alternative models
5. run the most informative discriminating experiment
6. update the world model
7. re-plan

Do not let a long time budget become an excuse for endless repetition.

---

# 11. LONG-RUNTIME REASONING

Large compute budgets may be available for research.

The system should be capable of spending substantial time on a difficult environment when worthwhile.

However, long runtime should produce progressively better understanding.

A long run should look like:

`explore → fail → diagnose → learn → revise → simulate → retry`

rather than:

`try → fail → try same thing → fail`

Use checkpoints so that long experiments can be resumed after crashes or interruptions.

---

# 12. COMPUTE ALLOCATION

Do not assume every specialist deserves equal compute.

The system should eventually learn where additional reasoning is useful.

For example:

- perception may need short fast calls
- exploration may require several candidate analyses
- falsification may require deeper analysis
- planning may require search plus model simulation
- senior reasoning may require the largest reasoning budget

Benchmark different allocations.

Optimize for final score and efficiency, not raw tokens per second.

---

# 13. SPECIALIST INDEPENDENCE

If separate specialist model instances are used, encourage genuine diversity.

Different agents should:

- have different prompts
- maintain different internal contexts
- prioritize different evidence
- be able to disagree

Do not artificially force consensus.

However, disagreement should eventually be resolved through evidence and the world model rather than popularity voting.

---

# 14. WORLD MODEL AS THE SHARED TRUTH

The specialists should not have separate incompatible versions of reality indefinitely.

They can disagree about interpretation, but the shared state should eventually contain:

- observations
- competing hypotheses
- evidence
- predictions
- actual outcomes
- confidence

This provides a common evidence base for the senior reasoner.

---

# 15. MODEL ARCHITECTURE SHOULD BE DISCOVERED

Do not assume:

6 × 8B + 1 × 27B is optimal.

Compare:

- 27B alone
- 27B + world model
- 27B + memory
- 27B + three specialists
- 27B + six specialists
- shared 8B specialists
- independent 8B specialists
- alternative agent roles
- different reasoning budgets
- different model quantizations

Keep the architecture that produces the best measured result.

The strongest architecture may be simpler than the initial design.

---

# 16. PROCEDURAL RESEARCH ENVIRONMENT

Eventually create a large procedural environment capable of generating many novel interactive games.

The purpose is not to replace ARC-AGI-3.

Its purposes are:

- stress testing
- discovering failure modes
- testing generalization
- producing training trajectories
- testing new mechanics
- testing unusual combinations of mechanics

The generator should be capable of producing effectively unbounded variations.

Possible dimensions:

- grid size
- object count
- object types
- movement
- interactions
- state transitions
- transformations
- delayed effects
- dependencies
- constraints
- goals
- irreversible actions
- mechanic composition
- difficulty

Automatically reject:

- impossible games
- trivially solvable games
- broken games
- accidental shortcuts
- games with ambiguous objectives that humans cannot reasonably infer

---

# 17. HUMAN CALIBRATION

Generated environments should be tested with humans.

Record:

- completion rate
- actions
- time
- mistakes
- exploration behavior
- whether the environment is learnable
- whether the intended mechanics can be discovered

Use human testing to detect bad benchmark design.

A difficult environment should be difficult because its underlying reasoning is difficult, not because it is broken.

---

# 18. PRIVATE GENERALIZATION BENCHMARK

Maintain a private internal evaluation set separate from development.

Use:

- development games
- validation games
- locked holdout games

The holdout should remain inaccessible to tuning and architecture decisions.

Do not repeatedly train directly against the holdout.

When the architecture stabilizes, create a fresh holdout.

The goal is to determine whether the system has learned general mechanisms rather than memorized specific tasks.

---

# 19. SELF-IMPROVEMENT LOOP

The complete research loop should eventually look like:

`run → measure → classify failure → propose improvement → implement → benchmark → keep/revert`

Each experiment should answer:

- What problem are we solving?
- Why should this change help?
- What changed?
- Did it help?
- Did it generalize?
- What did it cost in time/compute?
- Should it remain?

No component should remain solely because it sounds sophisticated.

---

# 20. FAILURE TAXONOMY

Maintain a controlled failure taxonomy.

Suggested categories:

- perception
- state tracking
- mechanics inference
- goal inference
- exploration
- world model
- falsification
- planning
- action selection
- efficiency
- recovery
- memory
- coordination
- timeout
- infrastructure
- model capability

Every major failure should receive a primary category.

This allows the research process to discover the actual bottleneck.

---

# 21. RESEARCH LOG

Keep an experiment history containing:

- experiment ID
- hypothesis
- implementation change
- expected effect
- development score
- validation score
- wall-clock cost
- action cost
- VRAM
- tokens
- failure distribution
- decision
- lessons learned

The research log should make it possible to reconstruct why the final architecture exists.

---

# 22. FINAL SYSTEM VISION

The finished system should feel less like:

“a model answering ARC puzzles”

and more like:

“a research organism investigating an unfamiliar world.”

It should:

- perceive
- experiment
- build theories
- challenge those theories
- construct predictive models
- discover goals
- plan
- execute
- notice when predictions fail
- learn from those failures
- preserve useful skills
- transfer validated knowledge
- continuously improve its procedures

The long-term goal is not simply to create a large collection of prompts.

The goal is to create a system capable of **learning how to solve new problems**.

---

# 23. NON-NEGOTIABLE RESEARCH PRINCIPLE

Do not assume the original idea is correct.

The architecture described in this document is a hypothesis.

If a simpler system wins, use the simpler system.

If a specialist is useless, remove it.

If a new component improves the result, keep it.

If the system discovers an architecture nobody expected, follow the evidence.

The benchmark score and generalization results are the authority.

Build the system that the experiments prove is strongest.