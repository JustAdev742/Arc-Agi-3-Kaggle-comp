"""Fixed dev/validation split of the 25 public games.

Derived once with ``random.Random(2026)`` stratified by input tag (2 click, 1 keyboard,
3 keyboard_click) and then hardcoded so it never drifts. Rules (CLAUDE.md):
never tune on VAL; report DEV and VAL separately in every research-log entry.
"""
from __future__ import annotations

TAGS: dict[str, str] = {
    "s5i5": "click",
    "tn36": "click",
    "lp85": "click",
    "su15": "click",
    "lf52": "click",
    "r11l": "click",
    "vc33": "click",
    "ls20": "keyboard",
    "tr87": "keyboard",
    "wa30": "keyboard",
    "g50t": "keyboard",
    "dc22": "keyboard_click",
    "sb26": "keyboard_click",
    "m0r0": "keyboard_click",
    "bp35": "keyboard_click",
    "cd82": "keyboard_click",
    "ka59": "keyboard_click",
    "re86": "keyboard_click",
    "ar25": "keyboard_click",
    "sp80": "keyboard_click",
    "cn04": "keyboard_click",
    "sc25": "keyboard_click",
    "tu93": "keyboard_click",
    "sk48": "keyboard_click",
    "ft09": "none"
}

VAL_GAMES: list[str] = ['cn04', 'g50t', 'lf52', 'r11l', 'sc25', 'sp80']
DEV_GAMES: list[str] = ['ar25', 'bp35', 'cd82', 'dc22', 'ft09', 'ka59', 'lp85', 'ls20', 'm0r0', 're86', 's5i5', 'sb26', 'sk48', 'su15', 'tn36', 'tr87', 'tu93', 'vc33', 'wa30']
ALL_GAMES: list[str] = sorted(TAGS)

# Two-game smoke set used by ``make verify`` (fast, one keyboard + one click game).
SMOKE_GAMES: list[str] = ["ls20", "vc33"]


def resolve(split: str) -> list[str]:
    s = split.strip().lower()
    if s == "dev":
        return list(DEV_GAMES)
    if s == "val":
        return list(VAL_GAMES)
    if s == "all":
        return list(ALL_GAMES)
    if s == "smoke":
        return list(SMOKE_GAMES)
    games = [g.strip().split("-")[0] for g in split.split(",") if g.strip()]
    unknown = [g for g in games if g not in TAGS]
    if unknown:
        raise ValueError(f"unknown games {unknown}; known: {ALL_GAMES}")
    return games
