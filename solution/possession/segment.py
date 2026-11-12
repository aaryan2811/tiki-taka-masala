"""Stage 4: possessions from per-frame evidence (milestone M4).

A per-frame P(home) flickers: any frame-level threshold invents a possession wherever the evidence
wavers for a moment, and section 1 of DESIGN.md says an invented possession costs more than a missed
one. So inside each stretch of play between two stoppages, the team in possession is decoded as a
whole: a two-state Viterbi path that collects each frame's log-odds for the team it gives the ball
to, pays switch_cost for every change of team, and keeps every possession but the first and last of
a stretch at least min_frames long. A short spell survives only if its evidence outweighs the two
changes it needs.

The stoppages are the rules' (deadball.py) and stay as they are, and so does a known restart team:
it has the ball on the stretch's first frame, and can lose it straight away.
"""

import numpy as np

from . import control
from .learned import play_stretches


def viterbi(log_odds, switch_cost, min_frames=1, first=None):
    """Per-frame team, 0 home or 1 away, maximising the sum over frames of half the log-odds for
    home (half its negative for away) minus switch_cost per change, with every possession that both
    begins and ends inside the span at least min_frames long. first forces the team of frame 0."""
    n = len(log_odds)
    if not n:
        return np.zeros(0, dtype=np.int8)
    half = np.asarray(log_odds, dtype=np.float64) / 2.0
    if min_frames <= 1:
        return _two_state(half, switch_cost, first)

    # score[team, age]: the best path ending here with the team holding the ball for age + 1
    # frames, the last age standing for min_frames or more. A change enters age 0 from the other
    # team's last age only, so the only choice to remember is how the last age was reached.
    oldest = int(min_frames) - 1
    score = np.full((2, oldest + 1), -np.inf)
    for team in ((0, 1) if first is None else (first,)):
        score[team, oldest] = half[0] if team == 0 else -half[0]
    stayed = np.zeros((n, 2), dtype=bool)
    gain = np.column_stack([half, -half])
    for t in range(1, n):
        new = np.empty_like(score)
        new[:, 0] = score[::-1, oldest] - switch_cost
        new[:, 1:oldest] = score[:, :oldest - 1]
        stay = score[:, oldest] >= score[:, oldest - 1]
        new[:, oldest] = np.where(stay, score[:, oldest], score[:, oldest - 1])
        stayed[t] = stay
        score = new + gain[t][:, None]

    team, age = np.unravel_index(int(np.argmax(score)), score.shape)
    out = np.empty(n, dtype=np.int8)
    out[n - 1] = team
    for t in range(n - 1, 0, -1):
        if age == 0:
            team, age = 1 - team, oldest
        elif age < oldest:
            age -= 1
        elif not stayed[t, team]:
            age = oldest - 1
        out[t - 1] = team
    return out


def _two_state(half, switch_cost, first):
    """viterbi() with no minimum length, in plain floats: a change is allowed on any frame."""
    n = len(half)
    home = half[0] if first in (None, 0) else -np.inf
    away = -half[0] if first in (None, 1) else -np.inf
    home_from_away = np.zeros(n, dtype=bool)
    away_from_home = np.zeros(n, dtype=bool)
    for t in range(1, n):
        value = half[t]
        switch_home, switch_away = away - switch_cost, home - switch_cost
        home_from_away[t] = switch_home > home
        away_from_home[t] = switch_away > away
        home, away = max(home, switch_home) + value, max(away, switch_away) - value
    team = 0 if home >= away else 1
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = team
        if t and (home_from_away[t] if team == 0 else away_from_home[t]):
            team = 1 - team
    return out


def decode(log_odds, owner, stoppages, blocks, switch_cost, min_frames=1):
    """A copy of owner with every stretch of play between the stoppages decoded by viterbi() from
    the per-frame log-odds; dead frames and frames outside the blocks keep owner's value."""
    out = owner.copy()
    for start, stop, team in play_stretches(len(owner), stoppages, blocks):
        first = None if team == control.NOBODY else int(team)
        out[start:stop] = viterbi(log_odds[start:stop], switch_cost, min_frames, first)
    return out
