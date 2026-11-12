"""Stage 5: put each change of possession on the touch that won the ball (milestone M3).

M1 changes possession on the new team's first *controlled* touch. That is the right evidence that the
ball has changed hands, but often the wrong moment. The public ground truth stamps a change a few
frames after the decisive touch -- the tackle that knocks the ball loose, the interception, the
header, the one-touch clearance -- not once the new team is visibly in control (DESIGN.md section 4,
M3). In open play the two can be seconds apart: a tackle pokes the ball away, it runs loose, and a
team-mate picks it up 40 frames later.

So detection stays where it is and only the boundary moves. For each change between the teams, walk
back from the controlled touch to the new team's first *contact* since the old team was last near the
ball:

  contact     a run of frames with one team nearest the ball within CONTROL_RADIUS (no speed
              condition), across which the ball's velocity changes by CONTACT_MIN_CHANGE or more.
              The change is what tells a touch from a pass rolling past a player.
  window      after the old team's last frame near the ball, and after the last frame the repaired
              ball is missing -- a contact seen before the ball vanished for more than a second says
              nothing about who has it when it comes back, the same rule M1 follows. Never before the
              old team's spell began. It ends where the run of frames holding the controlled touch
              begins: walking into that run would only find the ball arriving, which on the smoothed
              track is already early.
  no contact  the change stays on the controlled touch.

A change at a restart is left alone: the restart team takes the ball at the kick (deadball.py).
"""

import bisect

import numpy as np

from . import config
from .features import velocity
from .runs import value_runs

NOBODY = -1


def velocity_change(ball_velocity, start, stop, gap=config.CONTACT_GAP, span=config.CONTACT_SPAN):
    """How much the ball's velocity changed across [start, stop), in m/s: the mean velocity over the
    frames gap..span after the run, minus the mean over gap..span before it. NaN unless the velocity
    is known on all of those frames -- an unseen ball is no evidence of a touch.
    """
    before = ball_velocity[max(0, start - span):max(0, start - gap + 1)]
    after = ball_velocity[stop - 1 + gap:stop + span]
    if len(before) < span - gap + 1 or len(after) < span - gap + 1:
        return np.nan
    if np.isnan(before).any() or np.isnan(after).any():
        return np.nan
    return float(np.hypot(*(after.mean(axis=0) - before.mean(axis=0))))


def contacts(near, ball_velocity, min_change=config.CONTACT_MIN_CHANGE):
    """[(start, stop, team)] runs of one team nearest the ball (control.on_ball_team without a speed
    condition) across which the ball's velocity changed by min_change or more, in frame order."""
    return [(start, stop, team) for start, stop, team in value_runs(near)
            if team != NOBODY and velocity_change(ball_velocity, start, stop) >= min_change]


def place_changes(owner, near, ball, stoppages=()):
    """(owner, moved): a copy of owner with every change between the teams moved back to the decisive
    touch, and [(old_frame, new_frame)] for each change that moved.

    near is the per-frame team nearest the ball within CONTROL_RADIUS, with no speed condition; ball
    the repaired track (stage 1), NaN where it is still missing. stoppages are deadball.DeadBall.
    """
    n_frames = len(owner)
    dead = np.zeros(n_frames, dtype=bool)
    for stoppage in stoppages:
        dead[stoppage.start:stoppage.stop] = True
    restarts = set(stoppage.stop for stoppage in stoppages)
    unseen = np.isnan(ball[:, 0])

    found = contacts(near, velocity(ball, config.FPS))
    found_starts = [start for start, _, _ in found]
    breaks = np.concatenate(([True], near[1:] != near[:-1]))
    run_start = np.maximum.accumulate(np.where(breaks, np.arange(n_frames), 0))

    out = owner.copy()
    moved = []
    changes = np.flatnonzero((owner[1:] != owner[:-1]) & (owner[1:] != NOBODY)
                             & (owner[:-1] != NOBODY)) + 1
    for change in changes:
        if dead[change] or dead[change - 1] or change in restarts:
            continue
        new, old = owner[change], owner[change - 1]
        stop = run_start[change] if near[change] == new else change

        spell = change - 1
        while (spell > 0 and owner[spell - 1] == old and not dead[spell - 1]
               and spell not in restarts):
            spell -= 1
        lost = np.flatnonzero((near[spell:stop] == old) | unseen[spell:stop])
        begin = spell + (int(lost[-1]) + 1 if len(lost) else 0)

        for index in range(bisect.bisect_left(found_starts, begin), len(found)):
            start, _, team = found[index]
            if start >= stop:
                break
            if team == new:
                out[start:change] = new
                moved.append((int(change), int(start)))
                break
    return out, moved
