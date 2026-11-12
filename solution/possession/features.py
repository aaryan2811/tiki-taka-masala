"""Stage 2 inputs: per-frame geometry of the ball and both teams.

Collective features only. A track slot is a tracklet, not a person, so nothing here follows a player
through time -- per frame, each team is just the set of its tracked positions.
"""

import warnings

import numpy as np

HOME, AWAY = 0, 1
TEAMS = (HOME, AWAY)


def nearest_by_team(ball, players, team_of_slot):
    """(distance, slot), each (n, 2): each team's nearest tracked player to the ball, per frame.

    distance is in metres and NaN where the ball or that whole team is untracked; slot is -1 there.
    """
    gap = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))  # (n, s)
    distance = np.full((len(ball), 2), np.nan, dtype=np.float32)
    slot = np.full((len(ball), 2), -1, dtype=np.int16)
    for team in TEAMS:
        columns = np.flatnonzero(team_of_slot == team)
        team_gap = np.where(np.isnan(gap[:, columns]), np.inf, gap[:, columns])
        best = np.argmin(team_gap, axis=1)
        nearest = team_gap[np.arange(len(ball)), best]
        found = np.isfinite(nearest)
        distance[found, team] = nearest[found]
        slot[found, team] = columns[best[found]]
    return distance, slot


def velocity(positions, fps=25):
    """Velocity in m/s along the first axis: the mean of the steps in and out of each frame, or the
    one step that exists at the edge of a tracked run. NaN with neither neighbour tracked.

    Works on (n, 2) and (n, s, 2) alike. A slot that swaps identity shows up as one huge step, which
    only ever makes a touch look less controlled -- the safe direction.
    """
    back = np.full_like(positions, np.nan)
    back[1:] = positions[1:] - positions[:-1]
    ahead = np.full_like(positions, np.nan)
    ahead[:-1] = back[1:]
    both = (back + ahead) / 2
    return np.where(np.isnan(back), ahead, np.where(np.isnan(ahead), back, both)) * fps


def relative_speed(ball, players, slot, fps=25):
    """(n, 2): speed of the ball relative to each team's nearest player, from nearest_by_team's slots.

    Near zero when the ball moves with the player (a dribble, a ball at the feet); large when it is
    going past them (a pass, a shot, a clearance). NaN where either velocity is unknown.
    """
    ball_velocity = velocity(ball, fps)
    player_velocity = velocity(players, fps)
    out = np.full((len(ball), 2), np.nan, dtype=np.float32)
    for team in TEAMS:
        rows = np.flatnonzero(slot[:, team] >= 0)
        gap = ball_velocity[rows] - player_velocity[rows, slot[rows, team]]
        out[rows, team] = np.hypot(gap[:, 0], gap[:, 1])
    return out


def team_shape(players, speed, team_of_slot):
    """Per-team shape per frame, as a dict of (n, 2) arrays (or (n, 2, 2) for x/y pairs).

    centroid    mean position
    spread      standard deviation along x and along y -- a team in possession spreads out
    mean_speed  mean tracked speed
    n_tracked   tracked players
    """
    n = len(players)
    out = {
        "centroid": np.full((n, 2, 2), np.nan, dtype=np.float32),
        "spread": np.full((n, 2, 2), np.nan, dtype=np.float32),
        "mean_speed": np.full((n, 2), np.nan, dtype=np.float32),
        "n_tracked": np.zeros((n, 2), dtype=np.int8),
    }
    for team in TEAMS:
        columns = team_of_slot == team
        positions = players[:, columns]
        tracked = ~np.isnan(positions[:, :, 0])
        count = tracked.sum(axis=1)
        seen = count > 0
        out["n_tracked"][:, team] = count
        # nanmean warns on all-NaN rows; computing only where something is tracked avoids that.
        out["centroid"][seen, team] = np.nanmean(positions[seen], axis=1)
        out["spread"][seen, team] = np.nanstd(positions[seen], axis=1)
        # A team can be tracked on a frame with no speed for any of its players (speed is missing
        # for each tracklet's first frames); that mean stays NaN, without nanmean's warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out["mean_speed"][seen, team] = np.nanmean(speed[seen][:, columns], axis=1)
    return out
