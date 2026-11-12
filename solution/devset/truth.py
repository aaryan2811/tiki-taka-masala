"""Ground truth for a DFL dev match, in possession_metrics' codes (H 0, A 1, D 2, X 3).

The competition's truth is D where the corrected tracking marks the ball dead, and otherwise the
team it names as owning (CHALLENGE.md). DFL carries both flags on every frame, so the dev truth is
the same construction: D where BallStatus is 0, else H or A from BallPossession. The two halves are
the graded blocks, laid out as degrade.layout lays out the tracking, with X between them.

The two differ in one point of definition. The challenge's possession changes only when the other
team gains *controlled* possession: "a simple touch of the ball without control" and "an aerial duel
won without subsequent control of the ball by that team" do not count (CHALLENGE.md). DFL's flag
credits nearly every touch, including one the other team wins straight back: on the two calibration
matches, 82-85% of single touches lost back within 2 s are credited (the DFL feasibility audit, a
research note kept out of this line), and 37% / 30% of their spells are under 2 s. So the merged
variants absorb a short spell sandwiched between two spells of the other team (no dead ball between)
into them, at 2 s and 3 s.

The threshold is a modelling assumption, not a measurement. The feasibility audit first compared
the merged DFL spells with the public window's possessions; nothing rests on that now. M4 trains on
the 2 s variant because the pre-registered leave-one-calibration-match-out gate chose it over the
3 s one (DESIGN.md section 4, M4). A rule that itself merges short spells cannot be validated against
a truth that was merged the same way -- so read every variant, and trust a change only if it holds
under all of them.
"""

import numpy as np

from possession.repo import pm

from . import degrade

VARIANTS = (("raw", 0.0), ("merged2s", 2.0), ("merged3s", 3.0))  # (name, merge threshold in s)


def half_codes(half):
    """Codes for one half: D where the ball is dead, else the team BallPossession names."""
    codes = np.where(half.possession == 1, pm.H, pm.A).astype(np.int8)
    codes[half.status == 0] = pm.D
    return codes


def merge_lost_back(codes, seconds, fps=25):
    """Copy of codes with every H or A spell shorter than `seconds` absorbed into the spells of the
    other team on both sides of it.

    The audit's rule (merge_flipflops in its script), done on the run list rather than the frames:
    scanning left to right, a short spell with the same other-team spell on each side is joined with
    both, and the joined spell is checked again, since joining can make it the middle of a new
    sandwich. A spell next to a dead ball is never merged. After one, the challenge's definition
    settles it: play restarts with a clear possessor (CHALLENGE.md). Before one, absorbing a short
    spell into the other team's was tested as M4's training target by leaving one calibration match
    out, and it lost in all six cells (DESIGN.md section 4, "Re-checked on DFL";
    experiments/target_dead_ball.py).
    """
    if seconds <= 0:
        return codes.copy()
    limit = seconds * fps
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    runs = [[int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])]
            for i in range(len(bounds) - 1)]
    i = 1
    while i < len(runs) - 1:
        before, middle, after = runs[i - 1], runs[i], runs[i + 1]
        if (middle[2] in pm.POSSESSION_LABELS and before[2] == after[2]
                and before[2] in pm.POSSESSION_LABELS and before[2] != middle[2]
                and middle[1] - middle[0] < limit):
            runs[i - 1:i + 2] = [[before[0], after[1], before[2]]]
            i = max(1, i - 1)
        else:
            i += 1
    out = np.empty_like(codes)
    for start, stop, code in runs:
        out[start:stop] = code
    return out


def match_codes(match, merge_seconds=0.0):
    """(codes, blocks) for a whole match on degrade.layout's frame axis, X outside the halves."""
    n_frames, blocks = degrade.layout(match)
    codes = np.full(n_frames, pm.X, dtype=np.int8)
    for half, (start, stop) in zip(match.halves, blocks):
        codes[start:stop] = merge_lost_back(half_codes(half), merge_seconds)
    return codes, blocks


def variants(match):
    """{name: codes} for every truth variant of one match."""
    return {name: match_codes(match, seconds)[0] for name, seconds in VARIANTS}
