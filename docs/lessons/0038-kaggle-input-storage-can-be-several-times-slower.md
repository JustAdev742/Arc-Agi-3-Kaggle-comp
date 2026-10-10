Summary: Kaggle's /kaggle/input can read 4-5x slower than usual for a whole session (weights 26-53 s per shard instead of 5-9), so every wait on a model server must tolerate a ~30-minute boot and should key on progress, not a fixed clock.

# Kaggle input storage can be several times slower than usual (2026-10-10)

What happened: MTP session A v1 (Sat 2026-10-10, 00:14 UTC, right after the weekly quota reset) read the model at
26-53 s per safetensors shard; all 22 earlier D' runs read 5.3-9.2 s per shard (full load 4-6.5 min). The slowness
was there before any server started: the 6.7 GB wheel precache took 31.5 s (others 7-10 s). The session's fixed
20-minute health wait stopped the server at 29 of 38 shards. Nothing in the notebook explained it.
The probe pushed 40 minutes later (00:55) was still about 2x slow (weights 10:14, healthy 15.3 min after the start,
past the 12-minute release), so the slowness lasted at least an hour after the Saturday reset.

Where a slow boot bites:
- Our own waits: the session-A health wait (now: stops only a server whose log stops growing, up to 50 min), the
  fidelity probe and the --fail-fast watchdog (now 55 min from the notebook start).
- The submission (D', Franzen's notebook): the benchmark is released 12 min after the notebook starts whether or
  not the server is up (`SERVER_STARTUP_TIMEOUT`). Normal readiness in our runs is 8.9-11.7 min (exp-076g hit the
  12-min release). Only the 14 games holding gate slots call the model; each one's first request waits up to
  `ARC3_HTTP_RETRY_INITIAL_SECONDS` (900 s), then 10 consecutive failures (~27 s each) end that game. So a server
  ready later than about 31 minutes after the start costs the first 14 games outright (~13% of a draw); session A
  v1's storage would have put readiness near 29 minutes.

Again on 2026-10-10 (MTP session A2, 11:49-13:09 UTC): the first server loaded in 20.4 min (~28 s per shard) while
the notebook's precache read the inputs sequentially (~181 MB/s); a second server started 30 min later found the page
cache cold and read ~2.3 min per shard (~8-9 MB/s, the loader's range reads), so it was at 22 of 38 shards after the
50-min limit. Kaggle also showed that kernel as RUNNING for ~5 h before its first cell executed.

How to apply:
- Size every boot wait for ~30 minutes of weight loading and prefer "still making progress" over a fixed deadline.
- Before restarting a server in the same session, re-read its weight shards sequentially (a few parallel `cat`s; 45 GB
  in ~4 min even on a slow day) so the loader hits the page cache instead of the network storage.
- Next time a submission candidate is rebuilt, add `--env ARC3_HTTP_RETRY_INITIAL_SECONDS=2400`: the grace covers
  only each game's first request, so it costs nothing when the server is up in time.
- A run with a slow boot is visible in its notebook log ("DEADLINE at ...s from notebook start", or "Precaching:
  loaded 211 files (6.70 GB)" taking well over 10 s); read it before blaming a config for a low score.
