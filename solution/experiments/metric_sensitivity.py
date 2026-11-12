"""How much each kind of error costs on spellF1. Reproduces the table in DESIGN.md section 1.

    python solution/experiments/metric_sensitivity.py

Takes the public ground truth, damages it in one controlled way at a time, and scores the result
against the undamaged original with the official metric. No tracking data needed.

Seventeen possessions make the exact numbers noisy; the ordering between kinds of error is the point.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import repo  # noqa: E402  (needs the path above)
from possession.repo import pm  # noqa: E402

FPS = pm.FPS


def late(gt, runs, frames):
    """Every boundary moved later by `frames`: the whole window shifted right."""
    out = gt.copy()
    for start, stop in runs:
        out[start + frames:stop] = gt[start:stop - frames]
    return out


def absorb_short(gt, runs, seconds):
    """Possessions shorter than `seconds` taken over by the label before them."""
    out = gt.copy()
    for start, stop, code in pm.segments(gt, runs):
        if code in pm.POSSESSION_LABELS and stop - start < seconds * FPS and start > runs[0][0]:
            out[start:stop] = out[start - 1]
    return out


def no_dead_ball(gt, runs):
    """Every dead-ball stretch replaced by the team before it."""
    out = gt.copy()
    for start, stop, code in pm.segments(gt, runs):
        if code == pm.D and start > runs[0][0]:
            out[start:stop] = out[start - 1]
    return out


def false_dead_ball(gt, runs, seconds=1, longer_than=10):
    """A dead ball of `seconds` in the middle of every possession longer than `longer_than` s."""
    out = gt.copy()
    for start, stop, code in pm.segments(gt, runs):
        if code in pm.POSSESSION_LABELS and stop - start > longer_than * FPS:
            middle = (start + stop) // 2
            out[middle:middle + seconds * FPS] = pm.D
    return out


def main():
    gt = pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT))
    runs = pm.valid_runs(gt)

    cases = [("perfect", gt)]
    cases += [("all possession changes {} frames late".format(k), late(gt, runs, k))
              for k in (5, 12, 25, 50)]
    cases += [("possessions under {} s merged into the previous one".format(s),
               absorb_short(gt, runs, s)) for s in (1, 2, 3, 5)]
    cases += [("no dead balls at all", no_dead_ball(gt, runs)),
              ("false 1 s dead ball inside every long possession", false_dead_ball(gt, runs))]

    print("{:<52} {:>8} {:>8}".format("error put into a perfect answer", "spellF1", "macroF1"))
    for name, codes in cases:
        result = pm.evaluate(gt, codes)
        print("{:<52} {:>8.3f} {:>8.3f}".format(name, result["spell_f1_mean"], result["macro_f1"]))


if __name__ == '__main__':
    main()
