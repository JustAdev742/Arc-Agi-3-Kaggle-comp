#!/usr/bin/env python
"""Replace vendor/ARC-AGI-3-Agents/agents/__init__.py with a minimal registry (no langgraph/smolagents deps).
Same trick as the Kaggle starter and the official sample notebook."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INIT = ROOT / "vendor" / "ARC-AGI-3-Agents" / "agents" / "__init__.py"
SLIM = '''"""Slimmed by scripts/slim_framework.py: only the random template is registered."""
from typing import Type
from dotenv import load_dotenv
from .agent import Agent, Playback
from .swarm import Swarm
from .templates.random_agent import Random

load_dotenv()

AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
    "random": Random,
}
'''

if __name__ == "__main__":
    if not INIT.exists():
        raise SystemExit(f"framework not found at {INIT}; run `make setup`")
    INIT.write_text(SLIM)
    print(f"[slim_framework] wrote {INIT.relative_to(ROOT)}")
