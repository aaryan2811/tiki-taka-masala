"""Stage 2: a duel belongs to the team that comes out of it with the ball (milestone M3b).

M1 changes possession on any controlled touch by the other team. In a duel that is the wrong
question: two players of opposite teams are at the ball together, their tracked body centres a metre
or two from a heavily smoothed ball, and which of them is nearest flips back and forth on noise. The
public truth credits the team that comes out of the duel with the ball (DESIGN.md section 4, M3b):
an Athletic player shielding the ball with a Barcelona player on him keeps it, and an Athletic
dribble under pressure that ends with Barcelona taking the ball is never Athletic's.

So a controlled touch is void -- it never changes possession -- when both hold:

  contested   the other team is at the ball throughout it: its nearest player within
              CONTEST_RADIUS of the ball on every frame of the touch;
  taken back  the other team is on the ball again straight away: the next controlled touch is
              theirs and begins within CONTEST_GAP frames of this one ending.

The ball never leaves the two players, so the touch is a flip of the nearest team on tracking noise
rather than a change of hands. A tackle that knocks the ball loose is a different thing: the truth
credits the tackling team even when the other team recovers the loose ball (15346, 15766), and this
rule never touches it, because after a loose ball the other team is not back "straight away".

What the rule does not do: void a touch after which the ball travels, however soon the other team
recovers it. Over the whole match those waits spread out over several seconds with no gap to put a
threshold in, and the public truth credits some of them (a clearance recovered three seconds later)
and not others (a ball held under pressure and lost after 1.4 s) -- DESIGN.md section 4, M3b.
"""

import numpy as np

from . import config


def contested(distance, start, stop, team, radius=config.CONTEST_RADIUS):
    """True if the other team's nearest player is within radius of the ball on every frame of
    [start, stop). An untracked opponent is not evidence of a duel."""
    other = distance[start:stop, 1 - team]
    with np.errstate(invalid="ignore"):
        return bool(len(other)) and bool(np.all(other <= radius))


def resolve(touch_list, distance, gap=config.CONTEST_GAP, radius=config.CONTEST_RADIUS):
    """(kept, void): the controlled touches with every lost duel taken out, and the ones taken out.

    touch_list is control.touches() output in frame order; distance is features.nearest_by_team's,
    on the same ball track.
    """
    kept, void = [], []
    for index, (start, stop, team) in enumerate(touch_list):
        following = touch_list[index + 1] if index + 1 < len(touch_list) else None
        taken_back = (following is not None and following[2] != team
                      and following[0] - stop <= gap)
        if taken_back and contested(distance, start, stop, team, radius):
            void.append((start, stop, team))
        else:
            kept.append((start, stop, team))
    return kept, void
