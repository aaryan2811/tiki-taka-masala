"""Stage 3: dead ball, and who restarts play (milestone M2).

A false dead ball inside a possession splits it in two and costs more than a missed one (DESIGN.md
section 1), so everything here is built for precision:

  1. candidates   the players as a whole slow down (collective_speed below SLOW_SPEED for
                  SLOW_MIN_FRAMES). Every stoppage has one, but so do quiet spells of play.
  2. confirmation a candidate becomes a dead ball only with independent evidence: the ball crossing
                  a line just before or during it, a Smart Tagging set piece starting around it, or
                  a slowdown deeper or longer than open play gets (milestone M2b, mostly fouls).
  3. boundaries   start at the line crossing when there is one, else SLOW_LAG before the slowdown;
                  end at the restart kick, else where the slowdown ends.
  4. restart team the player nearest the ball as it is kicked, else the set-piece tag's team.

Milestone M2c adds two steps (detect's stoppage_starts and restart_laws, on by default):

  start           a stoppage starts at an earlier crossing when the ball has not been seen back on
                  the pitch since and the players never ran as in open play (out_since); one with no
                  crossing at all starts where the players' slowdown began (whistle).
  restart team    the laws of the game say who restarts, from where the players stand: a corner's
                  attackers crowd the penalty area, a goal kick's stay out of it (goal_line_team);
                  a throw-in's taker stands on the touchline (thrower_team). The nearest player and
                  the tag stay as fallbacks.

Short, shallow stoppages with neither a crossing nor a tag -- quick free kicks -- are still missed:
on labelled DFL matches the players behave exactly as in open play, and the ball signals that might
tell them apart cannot be checked anywhere (DESIGN.md section 4, M2b).
"""

import numpy as np

from . import config
from .ball import jumps
from .keepers import home_defends_low_x
from .runs import true_runs

NOBODY = -1
SET_PIECES = "set pieces"
TOUCHLINE, GOAL_LINE = "touchline", "goal line"


class DeadBall(object):
    """One stoppage: dead on [start, stop), play restarts at stop, taken by `team` (0, 1 or NOBODY)."""

    def __init__(self, start, stop, team, evidence):
        self.start, self.stop, self.team, self.evidence = int(start), int(stop), int(team), evidence

    def __repr__(self):
        return "DeadBall({}, {}, team={}, {})".format(self.start, self.stop, self.team, self.evidence)


def known_dead(n_frames, spans=config.KNOWN_DEAD_SPANS):
    """Mask of the published dead-ball spans (the second half graded before its kickoff)."""
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in spans:
        mask[start:stop] = True
    return mask


def dead_mask(n_frames, stoppages):
    """Boolean mask of every stoppage's [start, stop)."""
    mask = np.zeros(n_frames, dtype=bool)
    for stoppage in stoppages:
        mask[stoppage.start:stoppage.stop] = True
    return mask


# -- signals -----------------------------------------------------------------------------------------

def out_of_bounds(ball, pitch, margin=config.OUT_OF_BOUNDS_MARGIN):
    """Frames where the tracked ball is beyond a touchline or goal line by more than margin.

    The origin is a pitch corner -- checked in stage 0: positions span 0-1 with the ball straying
    past the lines -- so the field is [0, length] x [0, width] in metres.
    """
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid="ignore"):  # NaN compares False, which is the right answer
        return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def line_crossings(ball, pitch, min_frames=config.OUT_OF_BOUNDS_MIN_FRAMES):
    """Frames where the ball crossed a line, for every out-of-bounds stretch of min_frames or more.

    The crossing is walked back from the first frame past the margin to the first frame past the
    line itself, so the stoppage starts when the ball went out rather than when it was sure to be.
    """
    beyond = out_of_bounds(ball, pitch, margin=0.0)
    starts, stops = true_runs(out_of_bounds(ball, pitch))
    crossings = []
    for start, stop in zip(starts, stops):
        if stop - start < min_frames:
            continue
        while start > 0 and beyond[start - 1]:
            start -= 1
        crossings.append(start)
    return np.array(crossings, dtype=np.int64)


def collective_speed(speed, window=config.SLOWDOWN_WINDOW):
    """Mean speed of every tracked player, averaged over a centred window of +/- window frames.

    When play stops, everyone slows down together and stays slow until the restart; during play
    someone is usually running. NaN where nobody is tracked across the whole window.
    """
    per_frame_sum = np.nansum(speed, axis=1)
    per_frame_count = np.sum(~np.isnan(speed), axis=1)
    total = np.concatenate(([0.0], np.cumsum(per_frame_sum)))
    count = np.concatenate(([0], np.cumsum(per_frame_count)))

    index = np.arange(len(speed))
    low = np.maximum(0, index - window)
    high = np.minimum(len(speed), index + window + 1)
    seen = count[high] - count[low]
    out = np.full(len(speed), np.nan, dtype=np.float32)
    out[seen > 0] = (total[high] - total[low])[seen > 0] / seen[seen > 0]
    return out


def slow_runs(collective, graded, threshold=config.SLOW_SPEED, min_frames=config.SLOW_MIN_FRAMES):
    """[(start, stop)] graded stretches where collective speed stays below threshold long enough."""
    with np.errstate(invalid="ignore"):
        starts, stops = true_runs(graded & (collective < threshold))
    return [(int(a), int(b)) for a, b in zip(starts, stops) if b - a >= min_frames]


def beyond_open_play(collective, start, stop, deep=config.DEEP_SLOW_SPEED,
                     long_frames=config.LONG_SLOW_FRAMES):
    """True if the slowdown [start, stop) goes deeper or lasts longer than any in open play.

    In open play someone is always moving: the team on the ball looks for a pass, the other presses.
    Everyone walking, or a slowdown that goes on and on, happens when the referee has stopped play.
    """
    with np.errstate(invalid="ignore"):
        deepest = np.nanmin(collective[start:stop]) if stop > start else np.nan
    return bool(stop - start >= long_frames or deepest < deep)


def kicks(ball, speed=config.KICK_SPEED, min_frames=config.KICK_MIN_FRAMES,
          rest_frames=config.KICK_REST_FRAMES, fps=config.FPS):
    """Frames where the ball starts moving fast after resting: still, held, or unseen.

    A kick is the first frame of a run of min_frames faster than speed, with no frame that fast in
    the rest_frames before it. An unseen ball counts as resting, so a restart the tracker only picks
    up in flight still registers -- a frame or two late.
    """
    with np.errstate(invalid="ignore"):
        moving = jumps(ball, fps) > speed
    starts, stops = true_runs(moving)
    long_enough = starts[(stops - starts) >= min_frames]
    count = np.concatenate(([0], np.cumsum(moving)))
    before = count[long_enough] - count[np.maximum(0, long_enough - rest_frames)]
    return long_enough[(before == 0) & (long_enough >= rest_frames)]


def nearest_team(ball, players, team_of_slot, frame, radius=config.RESTART_TEAM_RADIUS,
                 search=config.RESTART_TEAM_SEARCH):
    """Team of the player nearest the ball on the first frame from `frame` where the ball is seen,
    if within radius. NOBODY when the ball is not seen within `search` frames or nobody is close."""
    for index in range(frame, min(frame + search, len(ball))):
        if np.isnan(ball[index, 0]):
            continue
        gap = np.hypot(*(players[index] - ball[index]).T)
        if np.all(np.isnan(gap)):
            return NOBODY
        slot = int(np.nanargmin(gap))
        return int(team_of_slot[slot]) if gap[slot] <= radius else NOBODY
    return NOBODY


# -- M2c: when play stopped, and who restarts it by the laws of the game ---------------------------

def crossed_line(ball, pitch, frame, margin=config.OUT_OF_BOUNDS_MARGIN):
    """(line, end, x) for the crossing at `frame`: TOUCHLINE or GOAL_LINE, 0 for the line at y = 0
    (a touchline) or x = 0 (a goal line) and 1 for the other one, and the ball's x as it went out.

    Read on the first frame from `frame` that is past the margin, where line_crossings found the
    stretch, and on the line the ball is furthest beyond -- near a corner it is past both.
    """
    length, width = pitch
    beyond = np.flatnonzero(out_of_bounds(ball[frame:], pitch, margin))
    x, y = ball[frame + int(beyond[0])] if len(beyond) else ball[frame]
    past_x, past_y = max(-x, x - length), max(-y, y - width)
    if past_x > past_y:
        return GOAL_LINE, int(x > length / 2), float(x)
    return TOUCHLINE, int(y > width / 2), float(x)


def out_since(ball, pitch, collective, crossings, start, back=config.OUT_SINCE_BACK,
              open_play=config.OPEN_PLAY_SPEED):
    """The earliest crossing before `start` since which play has stayed dead, else None.

    Play stays dead from a crossing while the ball is never seen back inside the lines and the
    players never move as they do in open play (collective speed below open_play throughout). Each
    crossing further back has to pass the test up to `start`, so the walk stops at the first one
    that fails. Players can jog on for 15-40 s after the ball goes out for a goal kick or a goal,
    longer than M2's crossing window, so a stoppage confirmed by its slowdown or a tag can start that
    long before the slowdown.
    """
    inside = ~np.isnan(ball[:, 0]) & ~out_of_bounds(ball, pitch, margin=0.0)
    found = None
    for crossing in crossings[(crossings >= start - back) & (crossings < start)][::-1]:
        with np.errstate(invalid="ignore"):
            if inside[crossing:start].any() or np.nanmax(collective[crossing:start]) >= open_play:
                break
        found = int(crossing)
    return found


def whistle(collective, slow_start, level=config.STOP_SPEED, back=config.STOP_BACK):
    """Where the slowdown that begins at slow_start set in: one past the last frame in the `back`
    frames before it with collective speed at or above `level`; slow_start if there is none.

    The candidate starts once the players are down to SLOW_SPEED, a second or more after they
    react to the whistle. Play that was already slow before the stoppage has no such frame, and the
    start stays put -- the early side is the expensive one (config.SLOW_LAG).
    """
    low = max(0, slow_start - back)
    with np.errstate(invalid="ignore"):
        above = np.flatnonzero(collective[low:slow_start] >= level)
    return low + int(above[-1]) + 1 if len(above) else slow_start


def thrower_team(players, team_of_slot, pitch, frame, end=None, x=None,
                 window=config.THROW_WINDOW, line=config.THROW_LINE, reach=config.THROW_REACH):
    """Team of the player standing on or behind a touchline in the `window` frames up to `frame`,
    else NOBODY.

    Law 15: the thrower stands on or behind the touchline, and opponents keep 2 m from the point of
    the throw. At free kicks and corners they keep 9.15 m from the ball (Laws 13 and 17), so whoever
    stands on a touchline at a restart is its taker or a team-mate. `end` (0 for y = 0) and `x`,
    from crossed_line, limit the search to the line the ball went out over and to `reach` metres
    along it from where it went out; without them both touchlines count.
    """
    width = pitch[1]
    span = players[max(0, frame - window):frame + 1]
    x_now, y_now = span[..., 0], span[..., 1]
    inside = np.minimum(y_now, width - y_now)
    with np.errstate(invalid="ignore"):
        if end is not None:
            inside = np.where((y_now > width / 2) == bool(end), inside, np.nan)
        if x is not None:
            inside = np.where(np.abs(x_now - x) <= reach, inside, np.nan)
    if np.isnan(inside).all():
        return NOBODY
    _, slot = np.unravel_index(np.nanargmin(inside), inside.shape)
    return int(team_of_slot[slot]) if np.nanmin(inside) <= line else NOBODY


def goal_line_team(players, team_of_slot, pitch, frame, end, home_defends_low,
                   min_attackers=config.CORNER_MIN_ATTACKERS):
    """Who restarts after the ball went out over the goal line at `end` (0 at x = 0): the attacking
    team for a corner, the defending team for a goal kick.

    Law 17 at a corner: the attackers crowd the penalty area for the cross. Law 16 at a goal kick:
    the attackers stay outside the penalty area until the ball is in play. So the attackers in that
    penalty area at the restart tell the two apart, which needs no ball and no knowing who touched it
    last. home_defends_low is keepers.home_defends_low_x on `frame`.
    """
    length, width = pitch
    defending = 0 if bool(home_defends_low) == (end == 0) else 1
    x, y = players[frame, :, 0], players[frame, :, 1]
    with np.errstate(invalid="ignore"):
        in_box = ((np.abs(x - (length if end else 0.0)) <= config.PENALTY_AREA_DEPTH)
                  & (np.abs(y - width / 2) <= config.PENALTY_AREA_HALF_WIDTH))
    attackers = int(np.sum(in_box & (team_of_slot == 1 - defending)))
    return 1 - defending if attackers >= min_attackers else defending


# -- stoppages ---------------------------------------------------------------------------------------

def tag_team(tag, team_names):
    """0 or 1 for a tag whose Team names one of the clubs, NOBODY for 'N/A' or anything else."""
    name = (tag["tags"].get("Team") or "").strip().lower()
    names = [team.strip().lower() for team in team_names]
    return names.index(name) if name in names else NOBODY


def detect(tracking, ball, blocks, tags, foul_stoppages=True, stoppage_starts=True,
           restart_laws=True):
    """[DeadBall] for the whole recording, sorted and non-overlapping.

    ball is the repaired ball track (stage 1); tags the Smart Tagging instances as returned by
    starter/load_smart_tagging.load(). The published known spans are always included.
    foul_stoppages=False confirms candidates by crossings and tags alone, as M2 did, to measure what
    the slowdown test adds. stoppage_starts moves starts back to an earlier crossing the ball never
    came back from (out_since), or to where the slowdown set in when there is no crossing
    (whistle). restart_laws takes the restart team from where the players stand (thrower_team,
    goal_line_team) before the nearest player and the tag. Both are M2c's; False measures what
    each adds.
    """
    n_frames = tracking.n_frames
    graded = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True

    crossings = line_crossings(ball, tracking.pitch)
    kick_frames = kicks(ball)
    pieces = [tag for tag in tags if (tag["code"] or "").strip().lower() == SET_PIECES]
    collective = collective_speed(tracking.speed)
    candidates = slow_runs(collective, graded)
    home_low = home_defends_low_x(n_frames, blocks, tracking.home_attacks_right_first_half)

    def law_team(frame, crossing):
        if crossing is None:
            return thrower_team(tracking.players, tracking.team_of_slot, tracking.pitch, frame)
        line, end, x = crossed_line(ball, tracking.pitch, crossing)
        if line == GOAL_LINE:
            return goal_line_team(tracking.players, tracking.team_of_slot, tracking.pitch, frame,
                                  end, home_low[crossing])
        return thrower_team(tracking.players, tracking.team_of_slot, tracking.pitch, frame, end, x)

    def restart_team(frame, tag=None, crossing=None, laws=False):
        team = law_team(min(frame, n_frames - 1), crossing) if laws else NOBODY
        if team == NOBODY:
            team = nearest_team(ball, tracking.players, tracking.team_of_slot, frame)
        if team == NOBODY and tag is not None:
            team = tag_team(tag, tracking.team_names)
        return team

    found = []
    for slow_start, slow_stop in candidates:
        crossed = crossings[(crossings >= slow_start - config.OOB_BEFORE) & (crossings < slow_stop)]
        tagged = [tag for tag in pieces
                  if slow_start - config.TAG_BEFORE <= tag["start_frame"]
                  <= slow_stop + config.TAG_AFTER]
        slow = foul_stoppages and beyond_open_play(collective, slow_start, slow_stop)
        if not len(crossed) and not tagged and not slow:
            continue

        crossing = int(crossed[0]) if len(crossed) else None
        start = crossing if crossing is not None else slow_start - config.SLOW_LAG
        if stoppage_starts:
            earlier = out_since(ball, tracking.pitch, collective, crossings, start)
            if earlier is not None:
                start = crossing = earlier
            elif crossing is None:
                start = whistle(collective, slow_start)
        window = kick_frames[(kick_frames >= max(start + config.MIN_DEAD_FRAMES,
                                                 slow_stop - config.RESTART_BACK))
                             & (kick_frames < slow_stop + config.RESTART_AHEAD)]
        stop = int(window[-1]) if len(window) else slow_stop
        if stop - start < config.MIN_DEAD_FRAMES:
            continue
        evidence = "+".join(name for name, seen in (("oob", len(crossed)), ("slow", slow),
                                                     ("tag", tagged)) if seen)  # as merge() sorts
        found.append(DeadBall(start, stop, restart_team(stop, tagged[0] if tagged else None,
                                                        crossing, restart_laws), evidence))

    for start, stop in config.KNOWN_DEAD_SPANS:
        found.append(DeadBall(start, stop, restart_team(stop), "known"))
    return clip(join(merge(found), kick_frames), blocks)


def merge(stoppages):
    """Overlapping stoppages joined; the later restart (and its team) wins."""
    out = []
    for stoppage in sorted(stoppages, key=lambda s: s.start):
        if out and stoppage.start < out[-1].stop:
            last = out[-1]
            later = stoppage if stoppage.stop >= last.stop else last
            out[-1] = DeadBall(last.start, later.stop, later.team,
                               "+".join(sorted(set(last.evidence.split("+"))
                                               | set(stoppage.evidence.split("+")))))
        else:
            out.append(stoppage)
    return out


def join(stoppages, kick_frames, gap=config.JOIN_GAP):
    """Sorted, non-overlapping stoppages with each one that has no restart kick joined to the next
    when that begins within gap frames and nothing is kicked in between: the ball never came back
    into play. The later restart (and its team) wins, as in merge()."""
    kick_frames = np.asarray(kick_frames)
    out = []
    for stoppage in stoppages:
        if out:
            last = out[-1]
            kicked = np.any((kick_frames >= last.stop) & (kick_frames < stoppage.start))
            if not kicked and stoppage.start - last.stop < gap:
                out[-1] = DeadBall(last.start, stoppage.stop, stoppage.team,
                                   "+".join(sorted(set(last.evidence.split("+"))
                                                   | set(stoppage.evidence.split("+")))))
                continue
        out.append(stoppage)
    return out


def clip(stoppages, blocks):
    """Stoppages cut to the graded blocks; anything left entirely outside them is dropped."""
    out = []
    for stoppage in stoppages:
        for block_start, block_stop in blocks:
            start, stop = max(stoppage.start, block_start), min(stoppage.stop, block_stop)
            if start < stop:
                out.append(DeadBall(start, stop, stoppage.team, stoppage.evidence))
    return out
