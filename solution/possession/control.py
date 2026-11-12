"""Stage 2: controlled touches, and the possession they imply (milestone M1).

The baseline gives the ball to the nearest player on every frame and then smooths. That flickers on
every deflection and writes D through every long occlusion. This module does what CHALLENGE.md
describes instead: possession changes only when the other team gains *controlled* possession, and
stays put through everything else -- loose balls, single touches, and frames with no ball at all.

Each change is placed on the first frame of the touch that caused it, which is the boundary
precision DESIGN.md section 1 says spellF1 pays for.
"""

import numpy as np

from . import config
from .runs import value_runs

NOBODY = -1


def on_ball_team(distance, relative=None, radius=config.CONTROL_RADIUS,
                 max_relative=config.CONTROL_MAX_RELATIVE_SPEED):
    """Per frame, the team whose nearest player is within radius of the ball and nearer than any
    opponent -- and, given relative (features.relative_speed), whose player the ball is moving with.
    NOBODY with no ball, nobody close, the ball going past, or an exact tie.
    """
    gap = np.where(np.isnan(distance), np.inf, distance)
    team = np.argmin(gap, axis=1)
    rows = np.arange(len(gap))
    on_ball = (gap[rows, team] <= radius) & (gap[:, 0] != gap[:, 1])
    if relative is not None:
        with np.errstate(invalid="ignore"):  # unknown relative speed is not evidence of control
            on_ball &= relative[rows, team] <= max_relative
    return np.where(on_ball, team, NOBODY).astype(np.int8)


def touches(on_ball, min_frames=config.CONTROL_MIN_FRAMES):
    """[(start, stop, team)] controlled touches: runs of one team on the ball of min_frames or more.

    A run is broken by any frame without the ball, so a touch is evidence seen, never inferred.
    """
    return [(start, stop, team) for start, stop, team in value_runs(on_ball)
            if team != NOBODY and stop - start >= min_frames]


def possession(touch_list, n_frames, blocks, stoppages=()):
    """Per-frame team in possession, 0 home or 1 away; NOBODY outside blocks.

    Inside each half-open block the owner is the team of the most recent touch that began in it. A
    touch already running on the block's first frame counts as beginning there -- the kick-off is
    taken by a player who was standing on the ball before the half was graded. Frames before the
    block's first touch take that touch's team, the same back-fill the scorer applies to
    abstentions. A block with no touch at all stays NOBODY.

    stoppages (deadball.DeadBall) change two things. Touches during a dead ball count for nothing --
    players pick the ball up, hand it back, place it -- except from the restart onwards if they are
    still going. And a stoppage's restart team, when known, takes possession at the restart.
    """
    dead = np.zeros(n_frames, dtype=bool)
    for stoppage in stoppages:
        dead[stoppage.start:stoppage.stop] = True

    marks = np.full(n_frames, NOBODY, dtype=np.int8)
    for start, stop, team in touch_list:
        alive = np.flatnonzero(~dead[start:stop])
        if not len(alive):
            continue
        first = start + int(alive[0])
        marks[first] = team
        for block_start, _ in blocks:
            if first < block_start < stop:
                marks[block_start] = team
    for stoppage in stoppages:
        if stoppage.team != NOBODY and stoppage.stop < n_frames:
            marks[stoppage.stop] = stoppage.team

    owner = np.full(n_frames, NOBODY, dtype=np.int8)
    for start, stop in blocks:
        window = marks[start:stop]
        known = np.flatnonzero(window != NOBODY)
        if not len(known):
            continue
        carry = np.maximum.accumulate(np.where(window != NOBODY, np.arange(len(window)), -1))
        carry[carry < 0] = known[0]
        owner[start:stop] = window[carry]
    return owner
