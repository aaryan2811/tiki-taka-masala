"""Stage 2b: a learned P(home has the ball) on every frame (milestone M4).

The rules of M1-M3b each read one kind of evidence: a controlled touch changes possession, a contact
moves the change back, a duel taken straight back is void. Where the evidence is weak or
contradictory -- a pass rolling among opponents, the ball unseen for seconds, two teams at the ball
-- each rule decides alone. This stage weighs all of it together, frame by frame, and stage 4
(segment.py) turns the result into possessions.

Every feature describes one team from its own side of the pitch: positions along the direction it
attacks (keepers.home_defends_low_x), never along x. So the features of the home team and of the
away team are the same quantities, and swapping them describes the same frame with the teams' roles
exchanged. The model is made symmetric under that swap -- nothing it learns can be about being the
home or the away team:

  trees     a gradient-boosted tree ensemble trained on every frame and its swapped copy; its
            log-odds are the half-difference of the raw scores of the frame and its swap
  logistic  a linear model on the home-minus-away difference of each team feature, and on the
            shared features times the difference in M2c's owner; no intercept

Per team, at each frame:
  distance, relative     the nearest player's distance to the ball and the ball's speed past him,
                         as control.on_ball_team reads them; 30 m and 20 m/s when unknown
  near_*, control_*      the share of frames in the second or three seconds before and after on
                         which the team is nearest the ball within CONTROL_RADIUS, without and with
                         the speed condition
  near_run               how long the run of frames nearest the ball holding this frame lasts
  since_*, until_*       how long since the team's last controlled touch (control.touches after
                         the duel rule), contact (boundary.contacts) and frame nearest the ball, and
                         until its next one
  owner                  M2c names the team
  restart                the stoppage this stretch of play follows was restarted by the team
  centroid, spread_x, spread_y, speed, tracked, advance
                         team shape (features.team_shape): centroid along the attack from halfway,
                         spread along and across, mean speed, tracked players, and how fast the
                         centroid moves towards the other goal
  ball_x, ball_to_centroid
                         where the ball was last seen, along the attack from halfway, and how far
                         from the team's centroid
Shared:
  seen, since_seen, until_seen      the repaired ball, and how long since and until it is seen
  ball_speed, ball_turn             its speed, and how much its velocity changes across the frame
                                    (boundary.velocity_change over one frame)
  since_restart, until_stop         how long since this stretch of play began, and until it ends
  spell_age, spell_left             how long since M2c's current possession began, and until it ends

Times are seconds, capped at CAP_SECONDS, as log(1 + s). Every one of them is a whole number of
frames underneath, and the trees compare them as such (tree_scores): many of their thresholds are
exactly log(1 + k / FPS) for some frame count k, so a float comparison would let one rounding step
of log1p on another platform move a frame across a split. The model's parameters live in model.json
next to this file, written by devset/train_model.py from the DFL dev set alone.
"""

import json
import os

import numpy as np

from . import config, control
from .boundary import contacts
from .features import AWAY, HOME, team_shape, velocity
from .keepers import home_defends_low_x
from .runs import true_runs, value_runs

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
FPS = config.FPS
CAP_SECONDS = 60.0
FAR, FAST = 30.0, 20.0  # distance (m) and relative speed (m/s) when unknown, and their caps
WINDOWS = (("back_1s", -FPS, 0), ("ahead_1s", 0, FPS), ("back_3s", -3 * FPS, 0),
           ("ahead_3s", 0, 3 * FPS))
ADVANCE_HALF = 12  # frames either side for the centroid's speed along the attack
CLIP = 12.0  # log-odds beyond this are clipped

# A tree threshold on a time feature within ON_FRAME frames of a whole frame count stands on that
# count. In the shipped model 395 of the 415 such thresholds are within 1e-9 frames of one, and the
# other 20 (midpoints between two counts) are at least 0.498 frames from any; expm1 is off by
# about 1e-12 frames at the largest threshold (4596 frames).
ON_FRAME = 1e-6  # frames

TEAM_FEATURES = (("distance", "relative")
                 + tuple("near_" + name for name, _, _ in WINDOWS)
                 + tuple("control_" + name for name, _, _ in WINDOWS)
                 + ("near_run", "since_touch", "until_touch", "since_contact", "until_contact",
                    "since_near", "until_near", "owner", "restart", "centroid", "spread_x",
                    "spread_y", "speed", "tracked", "advance", "ball_x", "ball_to_centroid"))
SHARED_FEATURES = ("seen", "since_seen", "until_seen", "ball_speed", "ball_turn", "since_restart",
                   "until_stop", "spell_age", "spell_left")
# The features seconds() makes from whole frame counts.
TEAM_TIMES = ("near_run", "since_touch", "until_touch", "since_contact", "until_contact",
              "since_near", "until_near")
SHARED_TIMES = ("since_seen", "until_seen", "since_restart", "until_stop", "spell_age", "spell_left")


def names():
    """Column names of evidence(): every team feature for home, then for away, then the shared."""
    return ([name + "_home" for name in TEAM_FEATURES] + [name + "_away" for name in TEAM_FEATURES]
            + list(SHARED_FEATURES))


def time_columns():
    """Boolean mask over names(): True on the features seconds() makes from whole frame counts."""
    return np.array([name.rsplit("_", 1)[0] in TEAM_TIMES or name in SHARED_TIMES
                     for name in names()])


def swap(features):
    """The same frames with the two teams' columns exchanged."""
    k = len(TEAM_FEATURES)
    return np.concatenate([features[:, k:2 * k], features[:, :k], features[:, 2 * k:]], axis=1)


# -- building blocks ---------------------------------------------------------------------------------

def since(mask, cap=int(CAP_SECONDS * FPS)):
    """Frames since mask was last True, at or before each frame (0 where it is True); cap where it
    never was or longer ago."""
    index = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, index, -1))
    return np.where(last >= 0, np.minimum(index - last, cap), cap)


def until(mask, cap=int(CAP_SECONDS * FPS)):
    """Frames until mask is next True, at or after each frame; cap where it never is again."""
    return since(np.asarray(mask)[::-1], cap)[::-1]


def seconds(frames):
    return np.log1p(np.asarray(frames, dtype=np.float64) / FPS)


def window_mean(values, low, high):
    """Mean of values over the frames [t + low, t + high) that exist, per frame t."""
    n = len(values)
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(n)
    start, stop = np.clip(index + low, 0, n), np.clip(index + high, 0, n)
    count = stop - start
    return np.where(count > 0, (total[stop] - total[start]) / np.maximum(count, 1), 0.0)


def span_mask(n, spans):
    mask = np.zeros(n, dtype=bool)
    for start, stop in spans:
        mask[start:stop] = True
    return mask


def run_length(mask):
    """Per frame, the length of the run of True holding it, 0 where False."""
    out = np.zeros(len(mask))
    for start, stop in zip(*true_runs(mask)):
        out[start:stop] = stop - start
    return out


def turn(ball_velocity, gap=config.CONTACT_GAP, span=config.CONTACT_SPAN):
    """Per frame, |mean velocity over frames gap..span after it - the same before it|, in m/s: the
    contact rule's velocity change, over a single frame. 0 unless the ball is seen on all of
    them."""
    known = ~np.isnan(ball_velocity[:, 0])
    filled = np.nan_to_num(ball_velocity)
    width = span - gap + 1
    before = [window_mean(filled[:, axis], -span, -gap + 1) for axis in (0, 1)]
    after = [window_mean(filled[:, axis], gap, span + 1) for axis in (0, 1)]
    complete = ((window_mean(known, -span, -gap + 1) * width > width - 0.5)
                & (window_mean(known, gap, span + 1) * width > width - 0.5))
    change = np.hypot(after[0] - before[0], after[1] - before[1])
    return np.where(complete, change, 0.0)


def play_stretches(n_frames, stoppages, blocks):
    """[(start, stop, restart team)] runs of graded frames outside every stoppage; the team is the
    restart team of the stoppage the stretch follows, control.NOBODY if unknown or none."""
    dead = span_mask(n_frames, [(s.start, s.stop) for s in stoppages])
    restart = {s.stop: s.team for s in stoppages}
    out = []
    for block_start, block_stop in blocks:
        starts, stops = true_runs(~dead[block_start:block_stop])
        for start, stop in zip(block_start + starts, block_start + stops):
            out.append((int(start), int(stop), restart.get(int(start), control.NOBODY)))
    return out


# -- the features ------------------------------------------------------------------------------------

def evidence(found, blocks):
    """(n, len(names())) float64 features on every frame, from pipeline.rules' output."""
    tracking, ball = found["tracking"], found["ball"]
    n = tracking.n_frames
    near, on_ball, owner = found["near"], found["on_ball"], found["owner"]
    ball_velocity = velocity(ball, FPS)
    seen = ~np.isnan(ball[:, 0])

    stretches = play_stretches(n, found["stoppages"], blocks)
    since_restart, until_stop = np.zeros(n), np.zeros(n)
    spell_age, spell_left = np.zeros(n), np.zeros(n)
    restart_team = np.full(n, control.NOBODY, dtype=np.int8)
    for start, stop, team in stretches:
        index = np.arange(start, stop)
        since_restart[start:stop] = index - start
        until_stop[start:stop] = stop - 1 - index
        restart_team[start:stop] = team
        for run_start, run_stop, _ in value_runs(owner[start:stop]):
            spell = np.arange(run_stop - run_start)
            spell_age[start + run_start:start + run_stop] = spell
            spell_left[start + run_start:start + run_stop] = spell[::-1]

    home_low = home_defends_low_x(n, blocks, tracking.home_attacks_right_first_half)
    middle = tracking.pitch[0] / 2.0
    shape = team_shape(tracking.players, tracking.speed, tracking.team_of_slot)
    last = np.maximum.accumulate(np.where(seen, np.arange(n), -1))
    last_ball = np.where((last >= 0)[:, None], ball[np.maximum(last, 0)],
                         [middle, tracking.pitch[1] / 2.0])
    found_contacts = contacts(near, ball_velocity)

    columns = []
    for team in (HOME, AWAY):
        attack = np.where(home_low == (team == HOME), 1.0, -1.0)  # +x is the team's attack
        distance = np.nan_to_num(np.minimum(found["distance"][:, team], FAR), nan=FAR)
        relative = np.nan_to_num(np.minimum(found["relative"][:, team], FAST), nan=FAST)
        is_near, in_control = near == team, on_ball == team
        touching = span_mask(n, [(a, b) for a, b, t in found["touches"] if t == team])
        contacting = span_mask(n, [(a, b) for a, b, t in found_contacts if t == team])
        centroid = shape["centroid"][:, team].astype(np.float64)
        along = (centroid[:, 0] - middle) * attack
        ahead = np.concatenate((along[ADVANCE_HALF:], np.full(ADVANCE_HALF, np.nan)))
        behind = np.concatenate((np.full(ADVANCE_HALF, np.nan), along[:-ADVANCE_HALF]))
        advance = np.nan_to_num((ahead - behind) * FPS / (2.0 * ADVANCE_HALF))
        columns += [distance, relative]
        columns += [window_mean(is_near, low, high) for _, low, high in WINDOWS]
        columns += [window_mean(in_control, low, high) for _, low, high in WINDOWS]
        columns += [seconds(run_length(is_near)),
                    seconds(since(touching)), seconds(until(touching)),
                    seconds(since(contacting)), seconds(until(contacting)),
                    seconds(since(is_near)), seconds(until(is_near)),
                    (owner == team).astype(np.float64), (restart_team == team).astype(np.float64),
                    np.nan_to_num(along),
                    np.nan_to_num(shape["spread"][:, team, 0].astype(np.float64)),
                    np.nan_to_num(shape["spread"][:, team, 1].astype(np.float64)),
                    np.nan_to_num(shape["mean_speed"][:, team].astype(np.float64)),
                    shape["n_tracked"][:, team].astype(np.float64), advance,
                    (last_ball[:, 0] - middle) * attack,
                    np.nan_to_num(np.hypot(*(last_ball - centroid).T), nan=FAR)]
    columns += [seen.astype(np.float64), seconds(since(seen)), seconds(until(seen)),
                np.minimum(np.nan_to_num(np.hypot(*ball_velocity.T)), 40.0), turn(ball_velocity),
                seconds(since_restart), seconds(until_stop), seconds(spell_age),
                seconds(spell_left)]
    return np.column_stack(columns)


# -- the model ---------------------------------------------------------------------------------------

def load(path=MODEL_PATH):
    """The shipped model: a dict as devset/train_model.py writes it."""
    with open(path) as handle:
        return json.load(handle)


def frame_counts(values):
    """The whole frame counts behind seconds() values. Rounding makes it exact under any few float
    steps of error in log1p or expm1: the counts are integers, the error a millionth of a frame."""
    return np.rint(np.expm1(values) * FPS)


def frame_cuts(thresholds):
    """The largest whole frame count that goes left at each seconds() threshold. A threshold standing
    on a count's log(1 + k / FPS) (within ON_FRAME) keeps that count on the left, where value <=
    threshold put it; one between two counts keeps the lower."""
    counts = np.expm1(np.asarray(thresholds, dtype=np.float64)) * FPS
    nearest = np.rint(counts)
    return np.where(np.abs(counts - nearest) < ON_FRAME, nearest, np.floor(counts))


def tree_scores(features, trees):
    """Sum over the trees of each row's leaf value. A tree is a dict of per-node lists: feature (-1
    at a leaf), threshold, left, right, missing_left and value; a row goes left where its feature is
    at most the threshold, and where it is NaN if missing_left.

    On a time feature (time_columns) both sides are first turned into whole frame counts
    (frame_counts, frame_cuts), so the comparison is between integers and no rounding of log1p can
    move a frame across a split."""
    timed = time_columns()
    if features.shape[1] != len(timed):
        raise ValueError("expected {} feature columns, got {}".format(
            len(timed), features.shape[1]))
    features = features.copy()
    features[:, timed] = frame_counts(features[:, timed])
    out = np.zeros(len(features))
    for tree in trees:
        feature = np.asarray(tree["feature"])
        threshold = np.asarray(tree["threshold"], dtype=np.float64)
        on_time = (feature >= 0) & timed[np.maximum(feature, 0)]
        threshold = np.where(on_time, frame_cuts(threshold), threshold)
        left, right = np.asarray(tree["left"]), np.asarray(tree["right"])
        missing_left = np.asarray(tree["missing_left"], dtype=bool)
        node = np.zeros(len(features), dtype=np.int64)
        active = np.flatnonzero(feature[node] >= 0)
        while len(active):
            at = node[active]
            value = features[active, feature[at]]
            go_left = np.where(np.isnan(value), missing_left[at], value <= threshold[at])
            node[active] = np.where(go_left, left[at], right[at])
            active = active[feature[node[active]] >= 0]
        out += np.asarray(tree["value"], dtype=np.float64)[node]
    return out


def antisymmetric(features):
    """The logistic model's design: home minus away for every team feature, then every shared
    feature times the difference in M2c's owner. Swapping the teams negates every column."""
    k = len(TEAM_FEATURES)
    owner = TEAM_FEATURES.index("owner")
    difference = features[:, :k] - features[:, k:2 * k]
    return np.concatenate([difference, features[:, 2 * k:] * difference[:, owner:owner + 1]],
                          axis=1)


def log_odds(features, model):
    """Per-frame log-odds that the home team has the ball, clipped to +/-CLIP."""
    if model["kind"] == "trees":
        raw = (tree_scores(features, model["trees"])
               - tree_scores(swap(features), model["trees"])) / 2.0
    elif model["kind"] == "logistic":
        raw = antisymmetric(features) @ (np.asarray(model["weights"])
                                         / np.asarray(model["scale"]))
    else:
        raise ValueError("unknown model kind {!r}".format(model["kind"]))
    return np.clip(raw, -CLIP, CLIP)
