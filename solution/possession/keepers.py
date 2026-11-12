"""Stage 1: a goalkeeper slot belongs to the goal it stands in (milestone M3b).

The XML flags one slot per team as the goalkeeper, and the tracker does not keep the two keepers
apart. It hands each keeper slot to whichever keeper it currently sees: of the graded frames, the
away keeper slot is in Barcelona's half on 16,392 and the home slot in Athletic's half on 1,610.
When the slot changes keeper it either reappears at the other goal after a gap or sweeps across the
pitch in a second, at speeds no player reaches. Read with a fixed team, each of those frames gives
the ball to the wrong side whenever a keeper has it -- a catch, a back-pass, a goal kick -- which is
two of the touches the public truth refuses to credit (DESIGN.md section 4, M3b).

The repair: each keeper slot becomes two, one per team, and on every frame the position goes to the
team that defends the half it is in (from attack_direction_first_half, switched at half time).
Frames where the slot moves faster than KEEPER_MAX_SPEED are dropped: that is the sweep between the
goals, not a player.
"""

import numpy as np

from . import config
from .ball import jumps
from .features import AWAY, HOME


def home_defends_low_x(n_frames, blocks, home_attacks_right_first_half):
    """Per frame, True where the home team defends the x = 0 goal. The ends change at the start of
    the second graded block, the second half."""
    out = np.full(n_frames, bool(home_attacks_right_first_half))
    if len(blocks) > 1:
        out[blocks[1][0]:] = not home_attacks_right_first_half
    return out


def split(tracking, blocks, max_speed=config.KEEPER_MAX_SPEED):
    """A copy of tracking with every keeper slot replaced by a home slot and an away slot, appended
    after the other slots; each holds the keeper's position only on the frames it is in that team's
    half. keeper_slots is empty on the result, so splitting twice changes nothing.
    """
    keepers = list(tracking.keeper_slots)
    if not keepers:
        return tracking
    others = [slot for slot in range(len(tracking.team_of_slot)) if slot not in keepers]
    players, speed = [tracking.players[:, others]], [tracking.speed[:, others]]
    teams = [tracking.team_of_slot[others]]

    home_low = home_defends_low_x(tracking.n_frames, blocks, tracking.home_attacks_right_first_half)
    middle = tracking.pitch[0] / 2.0
    for slot in keepers:
        position = tracking.players[:, slot].copy()
        with np.errstate(invalid="ignore"):  # NaN steps are not fast
            step = jumps(position) > max_speed
        sweep = step | np.concatenate((step[1:], [False]))  # both ends of a too-fast step
        position[sweep] = np.nan
        seen = ~np.isnan(position[:, 0])
        with np.errstate(invalid="ignore"):
            low = position[:, 0] < middle
        for team in (HOME, AWAY):
            own = seen & (low == (home_low if team == HOME else ~home_low))
            players.append(np.where(own[:, None], position, np.nan)[:, None])
            speed.append(np.where(own, tracking.speed[:, slot], np.nan)[:, None])
            teams.append(np.array([team], dtype=tracking.team_of_slot.dtype))

    return tracking.__class__(ball=tracking.ball, ball_fields=tracking.ball_fields,
                              players=np.concatenate(players, axis=1),
                              speed=np.concatenate(speed, axis=1),
                              team_of_slot=np.concatenate(teams), team_names=tracking.team_names,
                              pitch=tracking.pitch, keeper_slots=(),
                              home_attacks_right_first_half=tracking.home_attacks_right_first_half)
