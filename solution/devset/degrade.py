"""Make a clean DFL match look like the competition's tracking, and hand it over as an atd.Tracking.

The DFL positions come from a multi-camera system with every person identified on every frame and
the ball present throughout. The competition's ATD has none of that (DESIGN.md section 4, DATA.md):
the ball is missing on 41% of graded frames, in long stretches, and heavily smoothed; players are
46 anonymous team-level track slots that start and end; one keeper slot per team, which the tracker
sometimes hands to the other keeper. Scoring the pipeline on clean DFL tracking would measure a
problem we do not have, so each of those is reproduced here, in the order the tracker would have
produced them:

  1. dead-ball surgery  the DFL ball is frozen during every dead ball and jumps to the restart spot
                        when play resumes, which gives every label away. It is replaced by a ball
                        that rolls on for a moment, is carried to the restart spot and rests there
                        until the kick, which stays where DFL has it (dead_ball_track).
  2. smoothing          ball and players, Gaussian, widths fitted so that a kick ramps up and a
                        player's track wobbles as much as in the competition (fit.py).
  3. ball offset        a slowly varying misplacement of the ball, fitted on how near the nearest
                        player is (fit.py). It comes out small: smoothing alone gets most of it.
  4. ball dropout       the ball is lost and found again at the competition's per-frame rates,
                        which differ between play and dead balls and around the changes between
                        them: there it is seen on 83% of open-play frames, 56% of the three seconds
                        before a stoppage and 20% of a long stoppage.
  5. track slots        each person's track is cut into tracklets with competition lifetimes and
                        re-acquisition gaps, dealt into 23 anonymous slots per team; the keeper
                        keeps one flagged slot, which sometimes shows the other keeper instead.
  6. speed              from the degraded positions (the competition's speed channel agrees with its
                        positions, median ratio 0.99), missing where the competition's is.

Every number comes from params.json, written by fit.py. Every random draw comes from one seeded
generator per match, so a degraded match is reproducible.

What this does not reproduce, because nothing in the competition tracking measures it without a
ground truth: identity switches inside a slot, the ball stuck on the centre spot, the ball lagging
the real one by 15-30 frames, and any positional bias of the tracker.
"""

import json
import os

import numpy as np

from possession import atd
from possession.features import velocity
from possession.runs import true_runs

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "params.json")

# Probabilities at which fit.py stores every fitted length distribution. 21 points from ~75-1000
# runs: fine enough to keep the shape, too coarse to say anything about any one run.
PROBS = tuple(np.round(np.linspace(0.0, 1.0, 21), 2).tolist())

# Frames of X between the two halves of a dev match, so the scorer's valid_runs sees two blocks.
HALF_GAP = 250

SLOTS_PER_TEAM = 23  # 46 slots for 22 players, as in the competition (24 + 22 there)

# Ball dropout phases. The competition's ball is seen less in the last seconds before a stoppage
# (56%, the ball flying out or lost in a crowd at a foul) than in open play (84%), is lost over
# the first seconds of a stoppage (44%, then 20%) and found again in the first seconds after the
# restart.
# A chain with one rate per state cannot do that -- on the competition's own stoppages it gave 34%
# seen in them and 72% in play, against 23% and 83%; with these six phases 27% and 76%.
PHASES = ("alive_late", "alive_early", "alive_pre", "dead_late", "dead_early", "dead_pre")
PHASE_EDGE = 75  # frames: the "early" and "pre" phases are the first and last 3 s of a state

# Dead-ball surgery. Neither number is a degradation of the tracker, so neither can be fitted from
# the competition tracking: they describe what the real ball does, which DFL does not record.
#   RUN_OUT  the ball keeps the velocity it had when play stopped and slows to rest over this long
#            -- it rolls on past the line, or on after the whistle -- before anyone picks it up.
#   REST     it is placed (or held for a throw-in) at the restart spot this long before the kick.
#            The pipeline's kick detector wants 1 s without a fast move before a restart
#            (config.KICK_REST_FRAMES), less than this, so this choice barely matters to it. It
#            also means the dev set cannot say how long the rest should be.
RUN_OUT = 25  # frames
REST = 50  # frames


def load_params(path=PARAMS_PATH):
    with open(path) as handle:
        return json.load(handle)


def sample(table, rng, size=None):
    """Draws from a distribution stored as quantiles at PROBS (inverse CDF, linear between them)."""
    return np.interp(rng.random(size), PROBS, table)


def sample_frames(table, rng):
    """One length in whole frames, at least 1."""
    return max(1, int(round(float(sample(table, rng)))))


# -- 1. dead-ball surgery --------------------------------------------------------------------------

def dead_ball_track(ball, status, run_out=RUN_OUT, rest=REST):
    """Copy of a half's (n, 2) ball with every dead-ball stretch replaced, as described above.

    On [a, b) with status 0: frames a.. roll on from the last live position with the last live
    velocity, slowing linearly to rest over run_out frames; the last `rest` frames before b sit at
    the restart spot; the frames between carry the ball in a straight line from one to the other.
    The restart spot is where DFL has the ball one frame before its first live frame b -- on b it is
    already moving. A dead ball still running when the half ends just rolls out and stays.
    """
    out = ball.copy()
    n = len(ball)
    starts, stops = true_runs(status == 0)
    for a, b in zip(starts, stops):
        if a < 3:  # a half starts on its kick-off, so there is no live play to roll on from
            last, step = (ball[b] if b < n else ball[a]), np.zeros(2, dtype=ball.dtype)
        else:
            last, step = ball[a - 1], (ball[a - 1] - ball[a - 3]) / 2.0
        length = b - a
        k = min(run_out, length // 3) if b < n else min(run_out, length)
        frames = np.arange(1, k + 1, dtype=np.float32)
        out[a:a + k] = last + step * (frames - frames ** 2 / (2.0 * (k + 1)))[:, None]
        if b >= n:
            out[a + k:b] = out[a + k - 1] if k else last
            continue
        spot = ball[b] - (ball[b + 1] - ball[b]) if b + 1 < n else ball[b]
        hold = min(rest, (length - k) // 2)
        carry = length - k - hold
        origin = out[a + k - 1] if k else last
        weight = (np.arange(1, carry + 1, dtype=np.float32) / (carry + 1))[:, None]
        out[a + k:a + k + carry] = origin + weight * (spot - origin)
        out[b - hold:b] = spot
    return out


# -- 2. smoothing, 3. offset -----------------------------------------------------------------------

def gaussian_kernel(sigma):
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / float(sigma)) ** 2)
    return kernel / kernel.sum()


def smooth(track, sigma):
    """Centred Gaussian smoothing of an (n, d) track, edges held, NaN rows left NaN.

    Centred, not causal: the competition ball moves *before* the touch that moves it (DESIGN.md:
    the smoothed ball arrives about 11 frames early), which only a two-sided filter does. NaN rows
    are bridged by holding the last value, smoothed through, and put back -- callers smooth whole
    on-pitch stretches, so this only matters at their edges.
    """
    if sigma <= 0 or len(track) < 2:
        return track.copy()
    missing = np.isnan(track[:, 0])
    if missing.all():
        return track.copy()
    filled = track.copy()
    if missing.any():
        index = np.where(~missing, np.arange(len(track)), 0)
        index = np.maximum.accumulate(index)
        first = int(np.flatnonzero(~missing)[0])
        index[:first] = first
        filled = track[index]
    kernel = gaussian_kernel(sigma)
    radius = len(kernel) // 2
    padded = np.pad(filled, ((radius, radius), (0, 0)), mode="edge")
    out = np.column_stack([np.convolve(padded[:, d], kernel, mode="valid")
                           for d in range(track.shape[1])]).astype(track.dtype)
    out[missing] = np.nan
    return out


def smooth_stretches(track, sigma):
    """smooth() applied to each run of non-NaN rows separately, so nothing leaks across a gap."""
    out = np.full_like(track, np.nan)
    starts, stops = true_runs(~np.isnan(track[:, 0]))
    for a, b in zip(starts, stops):
        out[a:b] = smooth(track[a:b], sigma)
    return out


def slow_offset(n, sd, tau, rng):
    """(n, 2) AR(1) offset with stationary standard deviation sd per axis and time constant tau."""
    if sd <= 0:
        return np.zeros((n, 2), dtype=np.float32)
    keep = np.exp(-1.0 / tau)
    shocks = rng.normal(0.0, sd * np.sqrt(1 - keep ** 2), size=(n, 2))
    out = np.empty((n, 2))
    out[0] = rng.normal(0.0, sd, size=2)
    for i in range(1, n):
        out[i] = keep * out[i - 1] + shocks[i]
    return out.astype(np.float32)


# -- 4. dropout ------------------------------------------------------------------------------------

def phases(dead, edge=PHASE_EDGE):
    """Index into PHASES per frame of one half, from its dead-ball mask. "pre" wins over "early" in
    a state shorter than two edges; the last state of a half has no "pre", since nothing follows."""
    out = np.zeros(len(dead), dtype=np.int8)
    breaks = np.flatnonzero(dead[1:] != dead[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(dead)]))
    for start, stop in zip(bounds[:-1], bounds[1:]):
        base = 3 if dead[start] else 0
        out[start:stop] = base
        out[start:min(stop, start + edge)] = base + 1
        if stop < len(dead):
            out[max(start, stop - edge):stop] = base + 2
    return out


def dropout_mask(dead, ball_params, rng):
    """Boolean (n,) mask of frames where the ball is seen, for one half.

    A two-state chain: a seen ball is lost, and a lost one found, with the competition's per-frame
    rates for the frame's phase (PHASES). So the ball is seen about as often as there in play, in
    dead balls and around the changes between them. Stretch lengths come out geometric, which
    keeps their means but not their exact shape (README.md compares them).

    Drawing whole stretch lengths per state was tried first: it ties each stretch to the state it
    began in, and left the ball seen as often in dead balls as in play.
    """
    phase = phases(dead)
    rates = ball_params["rates"]
    lose = np.array([rates[name][0] for name in PHASES])[phase]
    find = np.array([rates[name][1] for name in PHASES])[phase]
    draws = rng.random(len(dead))
    seen = np.zeros(len(dead), dtype=bool)
    state = draws[0] < ball_params["seen_share"]["dead" if dead[0] else "alive"]
    for frame in range(len(dead)):
        if frame:
            state = draws[frame] >= lose[frame] if state else draws[frame] < find[frame]
        seen[frame] = state
    return seen


def degrade_ball(ball, status, params, rng):
    """(n, 2) degraded ball for one half, from the clean DFL ball and status."""
    track = dead_ball_track(ball, status)
    track = smooth(track, params["ball"]["smoothing_sigma"])
    track = track + slow_offset(len(track), params["ball"]["offset_sd"],
                                params["ball"]["offset_tau"], rng)
    seen = dropout_mask(status == 0, params["ball"], rng)
    return np.where(seen[:, None], track, np.nan).astype(np.float32)


# -- 5. track slots --------------------------------------------------------------------------------

def reacquire_gap(slot_params, rng):
    """Frames a person stays untracked between two tracklets."""
    if rng.random() < slot_params["reacquired_share"]:
        return sample_frames(slot_params["reacquire_frames"], rng)
    return slot_params["reacquire_window"] + 1 + int(rng.exponential(
        slot_params["late_reacquire_mean"]))


def tracklets(on_pitch, slot_params, rng):
    """[(start, stop)] tracked stretches of one person, inside the rows where they are on the pitch.

    Tracked and untracked stretches alternate with the competition's slot lifetimes and
    re-acquisition gaps. The first stretch is tracked, cut at a uniform point of its drawn lifetime,
    so that a person is not always picked up exactly when they come on.
    """
    out = []
    starts, stops = true_runs(on_pitch)
    for a, b in zip(starts, stops):
        frame = a
        life = int(sample_frames(slot_params["lifetime_frames"], rng) * rng.random()) + 1
        while frame < b:
            out.append((frame, min(b, frame + life)))
            frame += life + reacquire_gap(slot_params, rng)
            life = sample_frames(slot_params["lifetime_frames"], rng)
    return out


def deal_slots(pieces, n_slots):
    """[(slot, start, stop, person)] from [(start, stop, person)]: each tracklet goes to the slot
    that has been free the longest, the way a tracker reuses an identifier it has let go."""
    free_since = np.full(n_slots, -10 ** 9)
    busy_until = np.zeros(n_slots, dtype=np.int64)
    out = []
    for start, stop, person in sorted(pieces):
        free = np.flatnonzero(busy_until <= start)
        if not len(free):
            raise ValueError("more than {} tracklets at frame {}".format(n_slots, start))
        slot = int(free[np.argmin(free_since[free])])
        out.append((slot, start, stop, person))
        busy_until[slot] = stop
        free_since[slot] = stop
    return out


def keeper_swaps(n, keeper_params, rng):
    """Boolean (n,) mask of frames where a keeper slot shows the other team's keeper.

    The competition's away keeper slot stands in Barcelona's half on 16,392 graded frames, in a
    handful of long stretches, while Athletic's own keeper is not tracked at all. Normal stretches
    are exponential with the mean that gives the fitted share; swapped ones are drawn from the
    fitted lengths.
    """
    mask = np.zeros(n, dtype=bool)
    share = keeper_params["swap_share"]
    if share <= 0:
        return mask
    mean_swap = float(np.mean(keeper_params["swap_frames"]))
    mean_normal = mean_swap * (1 - share) / share
    frame = int(rng.exponential(mean_normal))
    while frame < n:
        length = sample_frames(keeper_params["swap_frames"], rng)
        mask[frame:frame + length] = True
        frame += length + int(rng.exponential(mean_normal)) + 1
    return mask


def speed_missing(starts, stops, speed_params, rng, n):
    """Boolean (n,) mask of a slot's tracked frames whose speed is missing: the first frames of
    some tracklets, while the tracker has too few positions to differentiate, and short stretches
    elsewhere."""
    mask = np.zeros(n, dtype=bool)
    for a, b in zip(starts, stops):
        if rng.random() < speed_params["start_share"]:
            mask[a:min(b, a + sample_frames(speed_params["start_frames"], rng))] = True
        for _ in range(rng.poisson(speed_params["interior_rate"] * (b - a))):
            at = int(rng.integers(a, b))
            mask[at:min(b, at + sample_frames(speed_params["interior_frames"], rng))] = True
    return mask


def degrade_players(half, params, rng):
    """(players (n, 46, 2), team_of_slot (46,), keeper_slots) for one half, in pitch-corner metres
    (the caller shifts them), from the clean DFL players."""
    n = half.n_frames
    sigma = params["players"]["smoothing_sigma"]
    clean = np.stack([smooth_stretches(half.players[:, person], sigma)
                      for person in range(half.players.shape[1])], axis=1)
    on_pitch = ~np.isnan(clean[:, :, 0])

    players = np.full((n, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    team_of_slot = np.repeat(np.array([0, 1], dtype=np.int8), SLOTS_PER_TEAM)
    keeper_slots = (SLOTS_PER_TEAM - 1, 2 * SLOTS_PER_TEAM - 1)
    keeper_track = {}
    for team in (0, 1):
        offset = team * SLOTS_PER_TEAM
        outfield, keeper = [], np.full((n, 2), np.nan, dtype=np.float32)
        for person in np.flatnonzero(half.team == team):
            pieces = tracklets(on_pitch[:, person], params["slots"], rng)
            if half.keeper[person]:
                for a, b in pieces:
                    keeper[a:b] = clean[a:b, person]
            else:
                outfield += [(a, b, int(person)) for a, b in pieces]
        for slot, a, b, person in deal_slots(outfield, SLOTS_PER_TEAM - 1):
            players[a:b, offset + slot] = clean[a:b, person]
        keeper_track[team] = keeper

    # The keepers' true positions, for a swapped slot to show. A team's keeper slot shows the
    # other keeper during its swaps, and its own keeper is then not tracked anywhere.
    truth = {}
    for team in (0, 1):
        truth[team] = np.full((n, 2), np.nan, dtype=np.float32)
        for person in np.flatnonzero((half.team == team) & half.keeper):  # a substitute keeper
            truth[team] = np.where(np.isnan(truth[team]), clean[:, person], truth[team])
    for team in (0, 1):
        swapped = keeper_swaps(n, params["keepers"], rng)
        players[:, keeper_slots[team]] = np.where(swapped[:, None], truth[1 - team],
                                                  keeper_track[team])
    return players, team_of_slot, keeper_slots


def alternating(n, on_table, off_table, on_share, rng):
    """Boolean (n,) mask of alternating on and off stretches with the given length distributions,
    starting on with probability on_share."""
    mask = np.zeros(n, dtype=bool)
    frame, on = 0, rng.random() < on_share
    while frame < n:
        length = sample_frames(on_table if on else off_table, rng)
        mask[frame:frame + length] = on
        frame += length
        on = not on
    return mask


def slot_speed(players, keeper_slots, params, rng):
    """(n, s) speed in m/s from the slot positions, NaN where the competition's would be: the
    outfield pattern on outfield slots, and the keeper slots' own, much sparser, channel."""
    v = velocity(players)
    speed = np.hypot(v[..., 0], v[..., 1]).astype(np.float32)
    keeper_params = params["keepers"]
    for slot in range(players.shape[1]):
        if slot in keeper_slots:
            missing = alternating(len(players), keeper_params["speed_missing_frames"],
                                  keeper_params["speed_present_frames"],
                                  keeper_params["speed_missing_share"], rng)
        else:
            starts, stops = true_runs(~np.isnan(players[:, slot, 0]))
            missing = speed_missing(starts, stops, params["slots"]["speed_missing"], rng,
                                    len(players))
        speed[missing, slot] = np.nan
    return speed


# -- the whole match -------------------------------------------------------------------------------

def home_attacks_right_first_half(match):
    """True if the home team kicks off the first half from the low-x side: at a kick-off every
    player stands in their own half, so the home players' mean x on the first frame says it."""
    half = match.halves[0]
    home = half.players[:5, half.team == 0, 0]
    return bool(np.nanmean(home) < 0)


def layout(match, gap=HALF_GAP):
    """(n_frames, blocks): the halves laid end to end on one frame axis with `gap` frames between,
    so the dev match looks like the competition's -- one array, two graded blocks."""
    blocks, start = [], 0
    for half in match.halves:
        blocks.append((start, start + half.n_frames))
        start += half.n_frames + gap
    return blocks[-1][1], blocks


def to_tracking(match, params, seed):
    """(tracking, blocks) for one DFL match: degraded, in pitch-corner metres, ready for
    possession.pipeline.label_frames."""
    rng = np.random.default_rng(seed)
    n_frames, blocks = layout(match)
    length, width = match.pitch
    shift = np.array([length / 2.0, width / 2.0], dtype=np.float32)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, 2 * SLOTS_PER_TEAM, 2), np.nan, dtype=np.float32)
    team_of_slot, keepers = None, ()
    for half, (start, stop) in zip(match.halves, blocks):
        ball[start:stop] = degrade_ball(half.ball[:, :2], half.status, params, rng) + shift
        half_players, team_of_slot, keepers = degrade_players(half, params, rng)
        players[start:stop] = half_players + shift
    speed = slot_speed(players, keepers, params, rng)

    scale = np.array(match.pitch, dtype=np.float32)
    ball_fields = np.column_stack([ball / scale, np.full(n_frames, np.nan, dtype=np.float32)])
    tracking = atd.Tracking(ball=ball, ball_fields=ball_fields, players=players, speed=speed,
                            team_of_slot=team_of_slot, team_names=match.team_names,
                            pitch=match.pitch, keeper_slots=keepers,
                            home_attacks_right_first_half=home_attacks_right_first_half(match))
    return tracking, blocks
