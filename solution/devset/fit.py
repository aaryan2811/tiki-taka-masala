"""Fit the degradation parameters from the competition tracking; write solution/devset/params.json.

    MTAH_DATA_DIR=/path/to/data python solution/devset/fit.py --dfl-dir /path/to/dfl

Only aggregate numbers leave the competition data: quantiles of stretch lengths, shares, smoothing
widths, tag counts per set-piece type. Never a frame-level mask, a position, or anything indexed by
frame -- RULES.md forbids publishing derived data that would reconstruct the tracking, and a list of
21 quantiles of ~100 run lengths cannot. Lengths are stored as quantiles at degrade.PROBS.

Most parameters are measured directly. Three are matched instead: a smoothing width, or the size of
the ball's misplacement, is the value at which the DFL matches show the same statistic as the
competition. Those need the DFL matches too (--dfl-dir), and only the calibration two
(dfl.CALIBRATION): the held-out matches must not shape the simulation they are scored on.

  ball     how often the ball is lost and found again, per frame, in each phase of play and dead
           ball (degrade.PHASES). "Dead" is the pipeline's own stoppages on the real match; settled
           play is graded frames outside them and outside every slowdown, so that stoppages the
           pipeline misses do not count as play.
           Smoothing width from how long a kick takes to ramp up; misplacement from how near the
           nearest player is.
  players  smoothing width from how much a track wobbles frame to frame; tracklet lifetimes;
           re-acquisition gaps; where the speed channel is missing.
  keepers  how often a keeper slot shows the other team's keeper, and for how long.
  tags     set-piece tags per type, the share naming no team, the window per type, and where a
           window starts relative to the restart.
"""

import argparse
import contextlib
import functools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from devset import degrade, dfl  # noqa: E402  (needs the path above)
from possession import atd, deadball, keepers, labels, pipeline  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession.repo import load_smart_tagging  # noqa: E402
from possession.runs import true_runs  # noqa: E402

# A kick, for the ramp measurement: a speed peak above KICK_PEAK m/s, with the ball below
# KICK_FROM m/s somewhere in the KICK_LOOKBACK frames before it, every one of them seen. The ramp is
# the frames from the last one under 20% of the peak to the first one over 80%. On the competition
# ball the median is 14 frames (244 kicks); on the raw DFL ball it is 1-2.
KICK_PEAK = 12.0  # m/s
KICK_FROM = 2.0  # m/s
KICK_LOOKBACK = 40  # frames
KICK_SPACING = 50  # frames: one kick per two seconds, so a long ramp is not counted twice

# Re-acquisition: a tracklet that ends is taken as the same person picked up again by the first
# tracklet of the same team to start within REACQUIRE_WINDOW frames, no further away than a player
# can run in the meantime (REACQUIRE_REACH m plus REACQUIRE_SPEED m/s). 66% of tracklet ends find
# one; the rest are untracked for longer, by an amount set so that the share of untracked players
# matches the competition's.
# The stoppages params.json was fitted on, pinned: crossings and tags only (no foul test), starting
# where M2 starts them (no M2c starts), with the restart kick after a 2 s rest. Later changes to the
# dead-ball stage (M2c, and the 1 s rest of the re-check on DFL) would otherwise move the
# degradation, and with it M4's training data, whenever the fit is rerun.
FIT_KICK_REST_FRAMES = 50  # frames, 2 s

REACQUIRE_WINDOW = 250  # frames
REACQUIRE_REACH = 2.0  # m
REACQUIRE_SPEED = 7.0  # m/s

# A keeper slot standing in the other team's half for at least this long shows the other keeper.
# Shorter stays are the slot sweeping between the goals (possession/keepers.py drops those).
KEEPER_SWAP_MIN = 25  # frames

# The ball's misplacement varies slowly, like a tracker locked on to the wrong blob for a while.
# The time constant is an assumption, not a fit: one second.
OFFSET_TAU = 25.0  # frames

# Searched grids for the matched parameters.
BALL_SIGMAS = np.arange(0.0, 12.01, 0.5)  # frames
PLAYER_SIGMAS = np.arange(0.0, 4.01, 0.25)  # frames
OFFSET_SDS = np.arange(0.0, 1.01, 0.1)  # m

# A tag is matched to the pipeline's restart within this many frames, to measure where its window
# starts relative to the restart.
TAG_MATCH = 250  # frames, 10 s

KICK_OFF = "Kick Off"


def quantiles(values):
    """Quantiles at degrade.PROBS, rounded to what a frame count or a metre can mean."""
    return [round(float(v), 4) for v in np.quantile(np.asarray(values, dtype=np.float64),
                                                    degrade.PROBS)]


def graded_mask(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        mask[start:stop] = True
    return mask


def interior_runs(mask, blocks):
    """[(start, stop)] True runs that start and end strictly inside one block: runs cut by a block
    edge have an unknown length."""
    out = []
    for block_start, block_stop in blocks:
        starts, stops = true_runs(mask[block_start:block_stop])
        out += [(block_start + int(a), block_start + int(b)) for a, b in zip(starts, stops)
                if a > 0 and b < block_stop - block_start]
    return out


# -- ball ------------------------------------------------------------------------------------------

@contextlib.contextmanager
def fitted_kick_rest(frames=FIT_KICK_REST_FRAMES):
    """deadball.kicks with the rest the fit was made with, restored however the block exits."""
    saved = deadball.kicks
    deadball.kicks = functools.partial(saved, rest_frames=frames)
    try:
        yield
    finally:
        deadball.kicks = saved


def competition_states(tracking, blocks, tags):
    """(dead, play) masks on the real match: the pipeline's stoppages, and graded frames outside
    them and outside every slowdown.

    The stoppages are M2's -- confirmed by a crossing or a tag -- not M2b's, whose slowdown test was
    calibrated on the dev set itself: fitting the degradation on stoppages found by that test would
    feed the dev set back into its own simulation. M2c's starts and the restart rest are pinned to
    what they were when params.json was fitted (FIT_KICK_REST_FRAMES), so a rerun writes the same
    file whatever the dead-ball stage has become."""
    with fitted_kick_rest():
        _, info = pipeline.label_frames(tracking, blocks, tags, foul_stoppages=False,
                                        stoppage_starts=False, learned_model=False)
    dead = deadball.dead_mask(tracking.n_frames, info["stoppages"])
    graded = graded_mask(tracking.n_frames, blocks)
    slow = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in deadball.slow_runs(deadball.collective_speed(tracking.speed), graded):
        slow[start:stop] = True
    return dead, graded & ~dead & ~slow, info["stoppages"]


def visibility_rates(seen, graded, state):
    """(lose, find): per-frame chance that a seen ball is lost, and that a lost one is found, on
    frames in `state` -- transitions counted on the frame they land on."""
    before = np.concatenate(([False], seen[:-1]))
    frames = state & graded & np.concatenate(([False], graded[:-1]))
    lose = np.sum(frames & before & ~seen) / float(np.sum(frames & before))
    find = np.sum(frames & ~before & seen) / float(np.sum(frames & ~before))
    return float(lose), float(find)


def kick_ramps(ball, valid):
    """Frames each kick takes to ramp from 20% to 80% of its peak speed (see KICK_PEAK)."""
    speed = ball_stage.jumps(ball)
    with np.errstate(invalid="ignore"):
        peaks = np.flatnonzero((speed[1:-1] > KICK_PEAK) & (speed[1:-1] >= speed[:-2])
                               & (speed[1:-1] >= speed[2:])) + 1
    out, last = [], -KICK_SPACING
    for peak in peaks:
        low = peak - KICK_LOOKBACK
        if peak - last < KICK_SPACING or low < 1 or not valid[low:peak + 1].all():
            continue
        window = speed[low:peak + 1]
        if np.isnan(window).any() or window.min() > KICK_FROM:
            continue
        window = window[int(np.argmin(window)):]
        rise_start = np.flatnonzero(window <= 0.2 * speed[peak])[-1]
        rise_stop = np.flatnonzero(window >= 0.8 * speed[peak])[0]
        out.append(int(rise_stop - rise_start))
        last = peak
    return np.array(out)


def nearest_distance(ball, players):
    """Distance from the ball to the nearest tracked player, per frame; NaN without either."""
    gap = np.hypot(*(players - ball[:, None, :]).transpose(2, 0, 1))
    gap = np.where(np.isnan(gap), np.inf, gap).min(axis=1)
    return np.where(np.isfinite(gap), gap, np.nan)


def fit_ball(tracking, blocks, dead, play, matches, rng):
    """The ball section of params.json, and the numbers behind it for printing."""
    seen = ~np.isnan(tracking.ball[:, 0])
    graded = graded_mask(tracking.n_frames, blocks)
    phase = np.full(tracking.n_frames, -1, dtype=np.int8)
    for start, stop in blocks:
        phase[start:stop] = degrade.phases(dead[start:stop])
    phase[(phase == degrade.PHASES.index("alive_late")) & ~play] = -1  # unsettled: not counted
    params = {
        "rates": {name: [round(v, 6) for v in visibility_rates(seen, graded, phase == index)]
                  for index, name in enumerate(degrade.PHASES)},
        "phase_frames": {name: int(np.sum(graded & (phase == index)))
                         for index, name in enumerate(degrade.PHASES)},
        "phase_seen_share": {name: round(float(seen[graded & (phase == index)].mean()), 4)
                             for index, name in enumerate(degrade.PHASES)},
        "seen_share": {"dead": float(seen[graded & dead].mean()),
                       "alive": float(seen[play].mean())},
        "seen_share_graded": float(seen[graded].mean()),
        # For checking the simulation only: the lengths of every seen and unseen stretch, and how
        # fast the tracked ball moves.
        "seen_frames": quantiles([b - a for a, b in interior_runs(seen, blocks)]),
        "unseen_frames": quantiles([b - a for a, b in interior_runs(~seen, blocks)]),
        "speed_p99": round(float(np.nanpercentile(ball_stage.jumps(tracking.ball)[graded], 99)), 2),
    }

    # Smoothing: the width at which DFL kicks ramp up as slowly as the competition's.
    target = float(np.median(kick_ramps(tracking.ball, seen & graded)))
    halves = [half for match in matches for half in match.halves]
    clean = [degrade.dead_ball_track(half.ball[:, :2], half.status) for half in halves]
    ramps = {}
    for sigma in BALL_SIGMAS:
        found = np.concatenate([kick_ramps(degrade.smooth(ball, sigma), half.status == 1)
                                for ball, half in zip(clean, halves)])
        ramps[float(sigma)] = float(np.median(found)) if len(found) else np.inf
    sigma = min(ramps, key=lambda s: (abs(ramps[s] - target), s))
    params.update({"smoothing_sigma": sigma, "kick_ramp_median": target,
                   "kick_ramp_median_dfl": ramps[sigma]})

    # Misplacement: the size at which the nearest player is as near as in the competition. The
    # lower quartile, because the upper tail of the competition's "play" still holds stretches of
    # the stoppages the pipeline misses, where the ball is far from everyone.
    distance = nearest_distance(tracking.ball, keepers.split(tracking, blocks).players)
    target_q25 = float(np.nanquantile(distance[play & seen], 0.25))
    smoothed = [degrade.smooth(ball, sigma) for ball in clean]
    q25 = {}
    for sd in OFFSET_SDS:
        values = []
        for ball, half in zip(smoothed, halves):
            moved = ball + degrade.slow_offset(len(ball), sd, OFFSET_TAU, rng)
            values.append(nearest_distance(moved, half.players)[half.status == 1])
        q25[float(sd)] = float(np.nanquantile(np.concatenate(values), 0.25))
    offset = min(q25, key=lambda s: (abs(q25[s] - target_q25), s))
    params.update({"offset_sd": round(offset, 2), "offset_tau": OFFSET_TAU,
                   "nearest_q25": target_q25, "nearest_q25_dfl": q25[offset]})
    return params


# -- players ---------------------------------------------------------------------------------------

def wobble(tracks):
    """Robust (MAD-based) standard deviation of each track's second difference, pooled over x and y
    and over a list of (n, s, 2) arrays. Robust because an identity switch in a competition slot is
    one huge second difference."""
    pooled = []
    for players in tracks:
        second = players[2:] - 2 * players[1:-1] + players[:-2]
        pooled.append(second[~np.isnan(second[..., 0])])
    second = np.concatenate(pooled)
    return float(1.4826 * np.median(np.abs(second - np.median(second, axis=0))))


def fit_players(tracking, blocks, matches):
    graded = graded_mask(tracking.n_frames, blocks)
    outfield = [slot for slot in range(len(tracking.team_of_slot))
                if slot not in tracking.keeper_slots]
    target = wobble([np.where(graded[:, None, None], tracking.players[:, outfield], np.nan)])
    halves = [half for match in matches for half in match.halves]
    found = {}
    for sigma in PLAYER_SIGMAS:
        found[float(sigma)] = wobble([np.stack(
            [degrade.smooth_stretches(half.players[:, person], sigma)
             for person in range(half.players.shape[1])], axis=1) for half in halves])
    sigma = min(found, key=lambda s: (abs(found[s] - target), s))
    return {"smoothing_sigma": sigma, "wobble": target, "wobble_dfl": found[sigma]}


def fit_slots(tracking, blocks):
    """Tracklet lifetimes, re-acquisition gaps and missing speed, over the outfield slots."""
    graded = graded_mask(tracking.n_frames, blocks)
    present = ~np.isnan(tracking.players[:, :, 0])
    outfield = [slot for slot in range(len(tracking.team_of_slot))
                if slot not in tracking.keeper_slots]

    lifetimes, births, deaths = [], [], []
    start_nan, interior_nan, tracked = [], [], 0
    for slot in outfield:
        team = int(tracking.team_of_slot[slot])
        for a, b in interior_runs(present[:, slot], blocks):
            lifetimes.append(b - a)
            births.append((a, team, tracking.players[a, slot]))
            deaths.append((b, team, tracking.players[b - 1, slot]))
            missing = np.isnan(tracking.speed[a:b, slot])
            tracked += b - a
            starts, stops = true_runs(missing)
            for s, e in zip(starts, stops):
                (start_nan if s == 0 else interior_nan).append(int(e - s))

    births.sort(key=lambda birth: birth[0])
    birth_frames = np.array([birth[0] for birth in births])
    used, gaps = set(), []
    for frame, team, position in deaths:
        low = np.searchsorted(birth_frames, frame)
        high = np.searchsorted(birth_frames, frame + REACQUIRE_WINDOW)
        for index in range(low, high):
            born, born_team, born_at = births[index]
            reach = REACQUIRE_REACH + REACQUIRE_SPEED * (born - frame) / 25.0
            if index not in used and born_team == team and np.hypot(*(born_at - position)) <= reach:
                used.add(index)
                gaps.append(born - frame)
                break
    reacquired = len(gaps) / float(len(deaths))

    # The untracked share of outfield players (10 per team on the pitch), and the mean of the late
    # re-acquisitions that reproduces it in an alternating tracked / untracked process.
    per_team = [np.sum(present[graded][:, [s for s in outfield
                                           if tracking.team_of_slot[s] == team]], axis=1).mean()
                for team in (0, 1)]
    untracked = 1.0 - float(np.mean(per_team)) / 10.0
    mean_gap = untracked / (1 - untracked) * float(np.mean(lifetimes))
    late = max(0.0, (mean_gap - reacquired * float(np.mean(gaps))) / (1 - reacquired)
               - REACQUIRE_WINDOW - 1)
    return {
        "lifetime_frames": quantiles(lifetimes),
        "n_tracklets": len(lifetimes),
        "reacquired_share": round(reacquired, 4),
        "reacquire_frames": quantiles(gaps),
        "reacquire_window": REACQUIRE_WINDOW,
        "late_reacquire_mean": round(late, 1),
        "untracked_share": round(untracked, 4),
        "outfield_tracked_per_team": [round(float(v), 3) for v in per_team],
        "tracked_per_team": round(float(np.sum(present[graded], axis=1).mean()) / 2.0, 3),
        "speed_missing": {
            "start_share": round(len(start_nan) / float(len(lifetimes)), 4),
            "start_frames": quantiles(start_nan or [1]),
            "interior_rate": len(interior_nan) / float(tracked),
            "interior_frames": quantiles(interior_nan or [1]),
            "share": round((sum(start_nan) + sum(interior_nan)) / float(tracked), 4),
        },
    }


def fit_keepers(tracking, blocks):
    graded = graded_mask(tracking.n_frames, blocks)
    home_low = keepers.home_defends_low_x(tracking.n_frames, blocks,
                                          tracking.home_attacks_right_first_half)
    middle = tracking.pitch[0] / 2.0
    present_total, lengths = 0, []
    for slot in tracking.keeper_slots:
        x = tracking.players[:, slot, 0]
        own_low = home_low if tracking.team_of_slot[slot] == 0 else ~home_low
        with np.errstate(invalid="ignore"):
            wrong = graded & ~np.isnan(x) & ((x < middle) != own_low)
        present_total += int(np.sum(graded & ~np.isnan(x)))
        starts, stops = true_runs(wrong)
        lengths += [int(b - a) for a, b in zip(starts, stops) if b - a >= KEEPER_SWAP_MIN]
    # The keeper slots' speed channel is missing on most of their frames (64% of graded ones,
    # against 2.4% for outfield slots), in long stretches: stored as alternating stretch lengths.
    missing_runs, present_runs, missing, tracked = [], [], 0, 0
    for slot in tracking.keeper_slots:
        seen = graded & ~np.isnan(tracking.players[:, slot, 0])
        nan_speed = seen & np.isnan(tracking.speed[:, slot])
        missing += int(nan_speed.sum())
        tracked += int(seen.sum())
        missing_runs += [b - a for a, b in interior_runs(nan_speed, blocks)]
        present_runs += [b - a for a, b in interior_runs(seen & ~nan_speed, blocks)]
    return {"swap_share": round(sum(lengths) / float(present_total), 4),
            "swap_frames": quantiles(lengths), "n_swaps": len(lengths),
            "speed_missing_share": round(missing / float(tracked), 4),
            "speed_missing_frames": quantiles(missing_runs),
            "speed_present_frames": quantiles(present_runs)}


# -- tags ------------------------------------------------------------------------------------------

def tag_type(tag):
    kind = (tag["tags"].get("Type") or "").strip()
    return KICK_OFF if kind.startswith(KICK_OFF) else kind


def fit_tags(tags, stoppages):
    pieces = [tag for tag in tags if (tag["code"] or "").strip().lower() == deadball.SET_PIECES]
    restarts = np.array([stoppage.stop for stoppage in stoppages])
    by_type, offsets = {}, []
    for tag in pieces:
        entry = by_type.setdefault(tag_type(tag), {"count": 0, "no_team": 0, "windows": []})
        entry["count"] += 1
        entry["no_team"] += (tag["tags"].get("Team") or "").strip().upper() == "N/A"
        entry["windows"].append(tag["end_frame"] - tag["start_frame"])
        gap = tag["start_frame"] - restarts
        near = gap[np.abs(gap) <= TAG_MATCH]
        if len(near):
            offsets.append(int(near[np.argmin(np.abs(near))]))
    return {
        "per_type": {kind: {"count": entry["count"],
                            "no_team_share": round(entry["no_team"] / float(entry["count"]), 4),
                            "window_frames": int(np.median(entry["windows"]))}
                     for kind, entry in sorted(by_type.items())},
        "n_tags": len(pieces),
        "start_offset_frames": quantiles(offsets),
        "n_matched": len(offsets),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder with the DFL match folders")
    parser.add_argument("--data-dir", default=None, help="competition data (default: games.py)")
    parser.add_argument("--matches", nargs="+", default=list(dfl.CALIBRATION),
                        help="DFL matches for the matched widths (default: the calibration two; "
                             "never the held-out ones)")
    parser.add_argument("--out", default=degrade.PARAMS_PATH)
    args = parser.parse_args()

    tracking = atd.load(data_dir=args.data_dir)
    blocks = labels.graded_blocks()
    tags = load_smart_tagging.load(data_dir=args.data_dir)
    match_ids = list(args.matches)
    matches = [dfl.load(args.dfl_dir, match_id) for match_id in match_ids]
    rng = np.random.default_rng(0)

    dead, play, stoppages = competition_states(tracking, blocks, tags)
    params = {
        "about": "Degradation parameters for the DFL dev set, fitted by solution/devset/fit.py on "
                 "the competition's tactical ATD (graded frames) and Smart Tagging. Lengths in "
                 "frames at 25 fps, distances in metres; *_frames lists are quantiles at "
                 "degrade.PROBS.",
        "credit": "The *_dfl values and the matched widths are measured on the DFL / Sportec open "
                  "data (Bassek et al. 2025, https://doi.org/10.6084/m9.figshare.28196177.v1), "
                  "CC BY 4.0, provided with authorization from the DFL.",
        "dfl_matches_for_matched_widths": match_ids,
        "ball": fit_ball(tracking, blocks, dead, play, matches, rng),
        "players": fit_players(tracking, blocks, matches),
        "slots": fit_slots(tracking, blocks),
        "keepers": fit_keepers(tracking, blocks),
        "tags": fit_tags(tags, stoppages),
    }
    with open(args.out, "w") as handle:
        json.dump(params, handle, indent=1, sort_keys=True)
        handle.write("\n")

    ball, slots = params["ball"], params["slots"]
    print("ball seen: {:.1%} of graded frames; {:.1%} in the pipeline's stoppages, {:.1%} in "
          "play".format(ball["seen_share_graded"], ball["seen_share"]["dead"],
                        ball["seen_share"]["alive"]))
    print("ball smoothing sigma {} frames (kick ramp {} frames, DFL {}); offset sd {} m "
          "(nearest-player q25 {:.2f} m, DFL {:.2f})".format(
              ball["smoothing_sigma"], ball["kick_ramp_median"], ball["kick_ramp_median_dfl"],
              ball["offset_sd"], ball["nearest_q25"], ball["nearest_q25_dfl"]))
    print("player smoothing sigma {} frames (wobble {:.5f} m, DFL {:.5f})".format(
        params["players"]["smoothing_sigma"], params["players"]["wobble"],
        params["players"]["wobble_dfl"]))
    print("tracklets: {} lifetimes, median {:.0f} frames; re-acquired {:.0%}; untracked {:.1%}; "
          "speed missing {:.1%}".format(slots["n_tracklets"], slots["lifetime_frames"][10],
                                        slots["reacquired_share"], slots["untracked_share"],
                                        slots["speed_missing"]["share"]))
    print("keeper slot shows the other keeper on {:.1%} of its frames ({} stretches)".format(
        params["keepers"]["swap_share"], params["keepers"]["n_swaps"]))
    print("tags: {} set pieces, {} matched to a restart".format(params["tags"]["n_tags"],
                                                                params["tags"]["n_matched"]))
    print("wrote {}".format(os.path.relpath(args.out)))


if __name__ == '__main__':
    main()
