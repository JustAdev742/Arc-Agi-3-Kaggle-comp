Engine facts: 64x64 grid, integer upscaling of small cameras, one action may yield several frames, available_actions is authoritative.

- Games render a small camera (e.g. 16x16 or 32x32) upscaled by an integer factor into 64x64; `perception.detect_scale`
  finds it and `downscale` gives the logical grid. Reason on the logical grid, click in the 64x64 frame.
- An action can return several frames (animation); the last frame is the state to reason on.
- `available_actions` in each frame lists the legal action ids; ACTION6 legality is reported without coordinates.
- GAME_OVER accepts only RESET. Levels are sequential; completing level k increments `levels_completed`.
- The local engine runs at ~1 ms per action on CPU: time is entirely the model's, never the environment's.
Source: arcengine 0.9.x sources, `docs.arcprize.org/actions`, local timing on 2026-09-15.
