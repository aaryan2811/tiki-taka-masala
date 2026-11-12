"""Stage 1: repair the ball track.

The ball is the strongest signal and the least reliable one: missing on 41% of graded frames, and
disproportionately on the contested ones. This stage only removes what is clearly wrong and fills
what is clearly safe to fill. Longer gaps are left as gaps, for the possession logic to bridge.
"""

import numpy as np

from . import config
from .runs import true_runs


def jumps(ball, fps=config.FPS):
    """Speed in m/s implied by each frame's step from the previous one. NaN without both ends."""
    step = np.full(len(ball), np.nan, dtype=np.float32)
    step[1:] = np.hypot(*(ball[1:] - ball[:-1]).T) * fps
    return step


def clean(ball, max_speed=config.BALL_MAX_SPEED, min_run=config.BALL_MIN_RUN, fps=config.FPS):
    """Copy of the ball track with bad detections set to NaN.

    Two rules, both conservative:
      - a detection reached by an impossible jump and left by another is a spike, not a shot
        (a real shot is one impossible-looking jump at most, never a there-and-back);
      - a visible run shorter than min_run frames is dropped as spurious.
    """
    out = ball.copy()
    step = jumps(out, fps)
    spike = np.zeros(len(out), dtype=bool)
    spike[:-1] = (step[:-1] > max_speed) & (step[1:] > max_speed)
    out[spike] = np.nan

    starts, stops = true_runs(~np.isnan(out[:, 0]))
    for start, stop in zip(starts, stops):
        if stop - start < min_run:
            out[start:stop] = np.nan
    return out


def interpolate(ball, max_gap=config.BALL_MAX_INTERP_GAP):
    """(filled, interpolated): short gaps filled with a straight line between their ends.

    Only interior gaps with a detection on both sides are filled; a gap at the start or end of the
    array has nothing to interpolate towards.
    """
    filled = ball.copy()
    interpolated = np.zeros(len(ball), dtype=bool)
    starts, stops = true_runs(np.isnan(ball[:, 0]))
    for start, stop in zip(starts, stops):
        if start == 0 or stop == len(ball) or stop - start > max_gap:
            continue
        before, after = ball[start - 1], ball[stop]
        weight = (np.arange(start, stop) - (start - 1)) / float(stop - start + 1)
        filled[start:stop] = before + weight[:, None] * (after - before)
        interpolated[start:stop] = True
    return filled, interpolated


def repair(ball):
    """clean() then interpolate(), with the default thresholds. Returns (ball, interpolated)."""
    return interpolate(clean(ball))
