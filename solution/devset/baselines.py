"""The reference baseline, baseline_smooth50, re-implemented on a Tracking.

starter/baseline.py streams the raw ATD file; a dev match has no such file, only arrays. The
nearest-player step is redone here on the arrays, exactly as the starter does it: every slot with
its XML team, keeper slots included and never split, NOBODY where the ball is not seen. The
majority vote is the starter's own smooth(), imported rather than copied, so the two cannot drift.

On the real match this reproduces the published file frame for frame (tests/test_devset.py checks
it when the match data is present).
"""

import numpy as np

from possession.repo import baseline, pm

WINDOW = 50  # frames either side: baseline_smooth50, the published floor


def nearest_team(tracking):
    """Per frame, the team of the tracked slot nearest the ball; NOBODY without ball or players."""
    gap = np.hypot(*(tracking.players - tracking.ball[:, None, :]).transpose(2, 0, 1))
    gap = np.where(np.isnan(gap), np.inf, gap)
    slot = np.argmin(gap, axis=1)
    found = np.isfinite(gap[np.arange(len(gap)), slot])
    return np.where(found, tracking.team_of_slot[slot], baseline.NOBODY).astype(np.int8)


def smooth50(tracking, blocks, window=WINDOW):
    """possession_metrics codes per frame: the starter's labels inside the blocks, X outside."""
    letters = baseline.smooth(nearest_team(tracking), window)
    codes = pm.label_codes(letters)
    inside = np.zeros(len(codes), dtype=bool)
    for start, stop in blocks:
        inside[start:stop] = True
    codes[~inside] = pm.X
    return codes
