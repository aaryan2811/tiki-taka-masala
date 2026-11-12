"""Where M2's restart rest, restart search, crossing window and minimum stoppage come from, measured
on the DFL calibration matches. Reproduces DESIGN.md section 4, "Re-checked on DFL", items 1-4, and
the numbers next to KICK_REST_FRAMES, RESTART_BACK, OOB_BEFORE and MIN_DEAD_FRAMES in config.py.

    python3 -I solution/experiments/dead_ball_windows.py --dfl-dir /path/to/dfl

The calibration matches only (devset/dfl.CALIBRATION), degraded as run_devset.py degrades them, 2
seeds, with synthetic tags; the rules run on them as shipped (M2c, no learned team). The held-out
five and the competition's public window are never read. Four parts:

  lengths   DFL's real dead balls (BallStatus runs, not degraded): the share shorter than each
            candidate rest and minimum length. A restart after a shorter dead ball cannot have
            rested that long.
  rest      kick-like events (deadball.kicks) at each candidate rest on the repaired degraded ball,
            by where the truth puts them: at a true restart (within RESTART_REACH), in open play,
            or in dead time, where the ball is the dev set's invention. Then, deciding nothing,
            where our stoppages end against the true restart with each rest: the invented ball
            rests exactly degrade.REST frames before every restart, so that side is not DFL's.
  back      for each truth dead ball one of our stoppages overlaps, the end of the latest slowdown
            candidate overlapping it minus the true restart: the share each RESTART_BACK
            candidate reaches.
  crossing  truth dead balls that start with a line crossing: how long after it the first slowdown
            candidate overlapping them starts, and the share each OOB_BEFORE candidate reaches;
            and candidates wholly in open play that a crossing within each window would confirm.
"""

import argparse
import contextlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from devset import adapter, degrade, dfl, tags, truth  # noqa: E402  (needs the paths above)
from devset.run_devset import match_seed, table  # noqa: E402
from possession import config, deadball, pipeline  # noqa: E402
from possession.repo import pm  # noqa: E402
from possession.runs import true_runs  # noqa: E402
from stoppage_ceiling import M2C, pair, truth_stoppages  # noqa: E402

FPS = config.FPS
RESTS = (12, 25, 50, 75)  # frames: 0.5, 1, 2 and 3 s of rest before a kick
LENGTHS = (12, 25, 50, 75, 100)  # frames: dead balls shorter than these are counted
BACKS = (4, 6, 8, 10, 12, 15)  # s, RESTART_BACK candidates
WINDOWS = (10, 15, 20, 25, 30)  # s, OOB_BEFORE candidates
RESTART_REACH = FPS  # a kick this close to a true restart is that restart
CROSSING_BEFORE, CROSSING_AFTER = FPS, 2 * FPS  # a crossing this close to a dead ball's start
END_SLACK = FPS  # our stoppage's end this close to the true restart is on time


class Seen(object):
    """One degraded calibration match: the shipped rules' stoppages, what they read, the truth."""

    def __init__(self, match, params, seed):
        self.match_id, self.seed = match.match_id, seed
        self.tracking, self.blocks = degrade.to_tracking(match, params,
                                                         match_seed(match.match_id, seed))
        self.tag_list = tags.synthetic(match, self.blocks, params, match_seed(match.match_id, seed))
        with adapter.no_known_dead_spans():
            found = pipeline.rules(self.tracking, self.blocks, self.tag_list)
        self.ball, self.stoppages = found["ball"], found["stoppages"]
        self.pitch = found["tracking"].pitch
        graded = np.zeros(self.tracking.n_frames, dtype=bool)
        for start, stop in self.blocks:
            graded[start:stop] = True
        self.graded = graded
        self.candidates = deadball.slow_runs(deadball.collective_speed(found["tracking"].speed),
                                             graded)
        self.crossings = deadball.line_crossings(self.ball, self.pitch)
        self.raw = truth.variants(match)["raw"]
        ends = [stop for _, stop in self.blocks]
        # truth dead balls that end in a restart (not with the half)
        self.oracle = [t for t in truth_stoppages(self.raw, self.blocks) if t.stop not in ends]
        self.restarts = np.array([t.stop for t in self.oracle], dtype=np.int64)


@contextlib.contextmanager
def rest_of(frames):
    """deadball.detect's kicks with rest_frames=frames inside the block."""
    saved = deadball.kicks
    deadball.kicks = lambda ball, **kwargs: saved(ball, rest_frames=frames, **kwargs)
    try:
        yield
    finally:
        deadball.kicks = saved


def seconds(frames):
    return "{:.1f} s".format(frames / float(FPS))


def share(part, whole):
    return "{} / {} ({:.1%})".format(part, whole, part / float(whole)) if whole else "0 / 0"


def quantiles(values, probs=(0.1, 0.25, 0.5, 0.75, 0.9)):
    if not len(values):
        return "-"
    return " / ".join("{:.1f}".format(q) for q in np.quantile(np.asarray(values) / float(FPS),
                                                               probs))


# -- lengths ---------------------------------------------------------------------------------------

def dead_lengths(matches):
    """Frame lengths of every dead ball in the matches' halves that ends in a restart."""
    out = []
    for match in matches:
        for half in match.halves:
            starts, stops = true_runs(half.status == 0)
            out += [int(b - a) for a, b in zip(starts, stops) if a > 0 and b < half.n_frames]
    return np.array(out, dtype=np.int64)


def lengths_part(matches):
    lengths = dead_lengths(matches)
    rows = [[seconds(limit), share(int(np.sum(lengths < limit)), len(lengths))]
            for limit in LENGTHS]
    return ("## DFL's dead balls ({}, real BallStatus, {} ending in a restart): shorter than\n\n"
            "{}\n\nLength quantiles q10 / q25 / q50 / q75 / q90: {} s".format(
                ", ".join(m.match_id for m in matches), len(lengths),
                table(["length", "dead balls shorter"], rows), quantiles(lengths)))


# -- rest ------------------------------------------------------------------------------------------

def where_kicks(seen, rest):
    """(at a restart, in open play, in dead time) counts of kick-like events on graded frames."""
    frames = deadball.kicks(seen.ball, rest_frames=rest)
    frames = frames[seen.graded[frames]]
    if len(seen.restarts):
        nearest = np.min(np.abs(frames[:, None] - seen.restarts[None, :]), axis=1)
    else:
        nearest = np.full(len(frames), np.iinfo(np.int64).max)
    at_restart = nearest <= RESTART_REACH
    dead = ~at_restart & (seen.raw[frames] == pm.D)
    play = ~at_restart & np.isin(seen.raw[frames], (pm.H, pm.A))
    return int(at_restart.sum()), int(play.sum()), int(dead.sum())


def restarts_found(seen, rest):
    """True restarts with a kick-like event within RESTART_REACH."""
    frames = deadball.kicks(seen.ball, rest_frames=rest)
    if not len(seen.restarts) or not len(frames):
        return 0
    return int(np.sum(np.min(np.abs(seen.restarts[:, None] - frames[None, :]), axis=1)
                      <= RESTART_REACH))


def end_errors(seen, rest):
    """Our stoppage's stop minus the true restart, per pair, with the rules run at this rest."""
    with rest_of(rest):
        stoppages = adapter.label(seen.tracking, seen.blocks, seen.tag_list, **M2C)[1]["stoppages"]
    return [ours.stop - theirs.stop for ours, theirs in pair(stoppages, seen.oracle)]


def rest_part(seen):
    n = float(len(seen))
    events, ends = [], []
    for rest in RESTS:
        counts = np.sum([where_kicks(one, rest) for one in seen], axis=0)
        found = sum(restarts_found(one, rest) for one in seen)
        total = sum(len(one.restarts) for one in seen)
        events.append([seconds(rest), "{:.1f}".format(counts[1] / n), "{:.1f}".format(counts[0] / n),
                       "{:.1f}".format(counts[2] / n), share(found, total)])
        errors = np.concatenate([end_errors(one, rest) for one in seen])
        ends.append([seconds(rest), len(errors),
                     share(int(np.sum(np.abs(errors) <= END_SLACK)), len(errors)),
                     share(int(np.sum(errors < -END_SLACK)), len(errors)),
                     share(int(np.sum(errors > END_SLACK)), len(errors))])
    return "\n".join([
        "## Kick-like events per match and seed, by rest before them (repaired degraded ball)\n",
        table(["rest", "in open play", "at a true restart (within 1 s)",
               "in dead time (invented ball)", "true restarts with one within 1 s"], events),
        "\n## Where our stoppages end, with each rest (decides nothing: the invented ball rests "
        "{} before every restart)\n".format(seconds(degrade.REST)),
        table(["rest", "pairs", "within 1 s of the true restart", "more than 1 s early",
               "more than 1 s late"], ends)])


# -- back ------------------------------------------------------------------------------------------

def back_gaps(seen):
    """(slowdown end minus true restart per truth dead ball our stoppages overlap, how many of
    those no slowdown candidate overlaps)."""
    gaps, none = [], 0
    for theirs in seen.oracle:
        if not any(min(s.stop, theirs.stop) > max(s.start, theirs.start) for s in seen.stoppages):
            continue
        over = [c for c in seen.candidates if c[0] < theirs.stop and c[1] > theirs.start]
        if not over:
            none += 1
            continue
        gaps.append(max(c[1] for c in over) - theirs.stop)
    return gaps, none


def back_part(seen):
    gaps, none = [], 0
    for one in seen:
        g, n = back_gaps(one)
        gaps += g
        none += n
    gaps = np.array(gaps)
    before = gaps[gaps >= 0]
    rows = [["{} s".format(back), share(int(np.sum(before <= back * FPS)), len(before))]
            for back in BACKS]
    return "\n".join([
        "## The true restart against the end of the latest slowdown over its dead ball\n",
        "Truth dead balls our stoppages overlap: {}; no slowdown over them: {}. Restart before the "
        "slowdown ends: {}; after it: {} (q10 / q25 / q50 / q75 / q90 of slowdown end minus "
        "restart: {} s).\n".format(len(gaps) + none, none, len(before), int(np.sum(gaps < 0)),
                                   quantiles(gaps)),
        "Of the restarts before the slowdown ends, how many each backward search reaches "
        "(q10 / q25 / q50 / q75 / q90: {} s):\n".format(quantiles(before)),
        table(["RESTART_BACK", "reached"], rows)])


# -- crossing --------------------------------------------------------------------------------------

def crossing_delays(seen):
    """(first overlapping slowdown's start minus the crossing, per truth dead ball that starts with
    a crossing; how many of those no slowdown overlaps)."""
    delays, none = [], 0
    for theirs in seen.oracle:
        near = seen.crossings[(seen.crossings >= theirs.start - CROSSING_BEFORE)
                              & (seen.crossings <= theirs.start + CROSSING_AFTER)]
        if not len(near):
            continue
        crossing = int(near[0])
        over = [c for c in seen.candidates if c[0] < theirs.stop and c[1] > crossing]
        if not over:
            none += 1
            continue
        delays.append(min(c[0] for c in over) - crossing)
    return delays, none


def open_play_candidates(seen):
    """Candidates whose every frame the truth marks as open play."""
    return [c for c in seen.candidates if np.all(np.isin(seen.raw[c[0]:c[1]], (pm.H, pm.A)))]


def confirmed(seen, candidate, window):
    start, stop = candidate
    return bool(np.any((seen.crossings >= start - window) & (seen.crossings < stop)))


def crossing_part(seen):
    delays, none = [], 0
    for one in seen:
        d, n = crossing_delays(one)
        delays += d
        none += n
    delays = np.array(delays)
    open_play = [(one, c) for one in seen for c in open_play_candidates(one)]
    rows = []
    for window in WINDOWS:
        false = sum(confirmed(one, c, window * FPS) for one, c in open_play)
        rows.append(["{} s".format(window), share(int(np.sum(delays <= window * FPS)),
                                                  len(delays)),
                     "{} ({:.1f} per match)".format(false, false / float(len(seen)))])
    return "\n".join([
        "## Dead balls that start with a crossing, and open-play slowdowns a crossing confirms\n",
        "Truth dead balls with a crossing from {} before to {} after their start: {}; no slowdown "
        "over them: {}. Slowdown start minus crossing, q10 / q25 / q50 / q75 / q90: {} s. "
        "Candidates wholly in open play: {} ({:.1f} per match).\n".format(
            seconds(CROSSING_BEFORE), seconds(CROSSING_AFTER), len(delays) + none, none,
            quantiles(delays), len(open_play), len(open_play) / float(len(seen))),
        table(["OOB_BEFORE", "crossing dead balls reached", "open-play candidates confirmed"],
              rows)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--seeds", type=int, default=2, help="degradation repeats per match")
    args = parser.parse_args()

    params = degrade.load_params()
    matches, seen = [], []
    for match_id in dfl.CALIBRATION:
        match = dfl.load(args.dfl_dir, match_id, verbose=False)
        matches.append(match)
        seen += [Seen(match, params, seed) for seed in range(args.seeds)]
        print("done {}".format(match_id), file=sys.stderr)
    print("# {} ({} seeds, tags)\n".format(", ".join(dfl.CALIBRATION), args.seeds))
    for part in (lambda: lengths_part(matches), lambda: rest_part(seen),
                 lambda: back_part(seen), lambda: crossing_part(seen)):
        print(part())
        print()


if __name__ == "__main__":
    main()
