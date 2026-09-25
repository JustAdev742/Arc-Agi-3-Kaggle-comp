"""List same-shape groups (rotation-invariant) on a game's first frame, ignoring colour."""
import sys, logging
sys.path.insert(0, '/home/user/Arc-Agi-3-Kaggle-comp')
sys.path.insert(0, '/tmp/claude-0/-home-user-Arc-Agi-3-Kaggle-comp/d342458e-03bd-545b-8a6d-06bca061963e/scratchpad/research/hg')
logging.disable(logging.INFO)
import numpy as np
from collections import defaultdict
from arc3.env import LocalEnv, make_arcade
from tr87_check import comps, bitmap, rot_key  # noqa (re-runs tr87 print, fine)
