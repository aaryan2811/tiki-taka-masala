"""Re-check M2b's slowdown test on DFL matches, calibration and held-out apart. Reports only.

    python solution/devset/foul_check.py --dfl-dir /path/to/dfl
    python solution/devset/foul_check.py --dfl-dir /path/to/dfl --seeds 5

M2b confirms a slowdown candidate that has neither a line crossing nor a set-piece tag when the
players' collective speed falls below DEEP_SLOW_SPEED (0.75 m/s) or the slowdown lasts
LONG_SLOW_FRAMES (15 s) or more. Both numbers were read off the two calibration matches
(dfl.CALIBRATION; solution/experiments/foul_stoppages.py, its dfl part): there no candidate wholly
in open play passed, and 58 of the 103 mostly dead ones did. This counts the same thing on every
match, two ways:

  clean     the calibration's own measurement: DFL's player speed channel, the raw DFL ball for line
            crossings, whole halves. On the calibration matches it reproduces M2b's counts, which is
            the check that the held-out numbers mean the same thing.
  degraded  what the pipeline meets on the dev set: the degraded slot speeds with keepers split, the
            repaired degraded ball for crossings, and the synthetic tags, over every seed.

Each candidate left unconfirmed (no crossing; degraded: and no tag either) is classed by the share
of its frames BallStatus marks dead: wholly in play (none), mostly dead (more than half) or mixed. A
wholly-in-play candidate that passes the test is a false stoppage; a mostly dead one that passes is
a caught one. The clean part also repeats the JOIN_GAP check: of two consecutive slowdowns where the
first ends with play still dead, how often the gap between them is dead throughout.

Nothing here changes a threshold: tuning on the held-out matches would spend them.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from devset import degrade, dfl, tags, truth  # noqa: E402  (needs the path above)
from devset.run_devset import match_seed  # noqa: E402
from possession import config, deadball, keepers  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession.repo import pm  # noqa: E402

FPS = config.FPS


class Candidate(object):
    """One slowdown candidate: where it is, what confirms it, and what the truth says."""

    def __init__(self, match_id, start, stop, crossed, tagged, dead_share, deepest, passes,
                 seed=None):
        self.match_id, self.seed = match_id, seed
        self.start, self.stop = int(start), int(stop)
        self.crossed, self.tagged = bool(crossed), bool(tagged)
        self.dead_share, self.deepest, self.passes = float(dead_share), float(deepest), bool(passes)

    @property
    def seconds(self):
        return (self.stop - self.start) / float(FPS)

    @property
    def kind(self):
        if self.dead_share == 0:
            return "play"
        return "dead" if self.dead_share > 0.5 else "mixed"


def judge(match_id, collective, dead, crossings, slow, pieces=(), seed=None):
    """Candidate per slowdown, with M2's crossing and tag windows and M2b's test."""
    out = []
    for start, stop in slow:
        crossed = np.any((crossings >= start - config.OOB_BEFORE) & (crossings < stop))
        tagged = any(start - config.TAG_BEFORE <= tag["start_frame"] <= stop + config.TAG_AFTER
                     for tag in pieces)
        with np.errstate(invalid="ignore"):
            deepest = np.nanmin(collective[start:stop])
        out.append(Candidate(match_id, start, stop, crossed, tagged, np.mean(dead[start:stop]),
                             deepest, deadball.beyond_open_play(collective, start, stop), seed))
    return out


def clean_candidates(match):
    """(candidates, gaps) from the calibration's measurement on one match, half by half. gaps are
    (frames, dead throughout) between consecutive slowdowns whose first ends with play dead."""
    shift = np.array(match.pitch, dtype=np.float32) / 2.0
    out, gaps = [], []
    for half in match.halves:
        collective = deadball.collective_speed(half.speed)
        crossings = deadball.line_crossings(half.ball[:, :2] + shift, match.pitch)
        slow = deadball.slow_runs(collective, np.ones(half.n_frames, dtype=bool))
        out += judge(match.match_id, collective, half.status == 0, crossings, slow)
        dead = half.status == 0
        gaps += [(start - stop, bool(dead[stop:start].all()))
                 for (_, stop), (start, _) in zip(slow[:-1], slow[1:]) if dead[stop - 1]]
    return out, gaps


def join_lines(gaps):
    """The JOIN_GAP check, as in experiments/foul_stoppages.py."""
    lines = ["  consecutive slowdowns, the first ending with play dead: gap -> dead throughout"]
    for low, high in ((0, config.JOIN_GAP), (config.JOIN_GAP, 2 * config.JOIN_GAP),
                      (2 * config.JOIN_GAP, 5 * config.JOIN_GAP)):
        group = [whole for gap, whole in gaps if low <= gap < high]
        lines.append("    {:>4.1f}-{:<4.1f} s: {} of {}".format(low / float(FPS), high / float(FPS),
                                                              sum(group), len(group)))
    return lines


def degraded_candidates(match, params, seed):
    """What the pipeline's dead-ball stage sees on one degraded match."""
    tracking, blocks = degrade.to_tracking(match, params, match_seed(match.match_id, seed))
    pieces = tags.synthetic(match, blocks, params, match_seed(match.match_id, seed))
    split = keepers.split(tracking, blocks)
    ball, _ = ball_stage.repair(split.ball)
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True
    collective = deadball.collective_speed(split.speed)
    dead = truth.match_codes(match)[0] == pm.D
    return judge(match.match_id, collective, dead, deadball.line_crossings(ball, split.pitch),
                 deadball.slow_runs(collective, graded), pieces, seed)


def report(title, found, n_repeats=1, use_tags=False):
    """Printable lines: what the test does to the candidates nothing else confirms."""
    left = [c for c in found if not c.crossed and not (use_tags and c.tagged)]
    by_kind = {kind: [c for c in left if c.kind == kind] for kind in ("play", "mixed", "dead")}
    passing = [c for c in left if c.passes]
    frames = sum(c.stop - c.start for c in passing)
    dead_frames = sum((c.stop - c.start) * c.dead_share for c in passing)
    per = " per seed" if n_repeats > 1 else ""
    lines = ["  {}: {} candidates{}, {} confirmed by a crossing{}; {} left: {} wholly in play, {} "
             "mixed, {} mostly dead".format(
                 title, len(found) / float(n_repeats), per,
                 sum(c.crossed or (use_tags and c.tagged) for c in found) / float(n_repeats),
                 " or a tag" if use_tags else "", len(left) / float(n_repeats),
                 len(by_kind["play"]) / float(n_repeats), len(by_kind["mixed"]) / float(n_repeats),
                 len(by_kind["dead"]) / float(n_repeats))]
    play = by_kind["play"]
    lines.append("    the test passes: wholly in play {} of {} (false stoppages), mixed {} of {}, "
                 "mostly dead {} of {} (caught); dead frames in what it passes {:.0%}".format(
                     sum(c.passes for c in play), len(play),
                     sum(c.passes for c in by_kind["mixed"]), len(by_kind["mixed"]),
                     sum(c.passes for c in by_kind["dead"]), len(by_kind["dead"]),
                     dead_frames / float(frames) if frames else 0.0))
    if len(play) > 1:
        lengths = sorted(c.seconds for c in play)
        lowest = sorted(c.deepest for c in play)
        lines.append("    wholly in play: longest {:.1f} s (next {:.1f} s), lowest collective "
                     "speed {:.2f} m/s (next {:.2f})".format(lengths[-1], lengths[-2], lowest[0],
                                                      lowest[1]))
    for c in sorted((c for c in play if c.passes), key=lambda c: (c.match_id, c.start)):
        lines.append("      false: {}{} frames {}-{} ({:.1f} s, lowest {:.2f} m/s)".format(
            c.match_id, "" if c.seed is None else " seed {}".format(c.seed), c.start, c.stop,
            c.seconds, c.deepest))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True)
    parser.add_argument("--matches", nargs="+", help="match ids (default: every one found)")
    parser.add_argument("--seeds", type=int, default=5, help="degradation repeats per match")
    args = parser.parse_args()

    params = degrade.load_params()
    match_ids = args.matches or dfl.available_matches(args.dfl_dir)
    matches = [dfl.load(args.dfl_dir, match_id) for match_id in match_ids]
    print("M2b's test: lowest collective speed below {} m/s, or {} s or more".format(
        config.DEEP_SLOW_SPEED, config.LONG_SLOW_FRAMES // FPS))
    for title, group in (("calibration", [m for m in matches if m.match_id in dfl.CALIBRATION]),
                         ("held-out", [m for m in matches if m.match_id not in dfl.CALIBRATION])):
        if not group:
            continue
        print("\n{} ({})".format(title, ", ".join(m.match_id for m in group)))
        measured = [clean_candidates(m) for m in group]
        clean = [c for candidates, _ in measured for c in candidates]
        print("\n".join(report("clean", clean)))
        print("\n".join(join_lines([gap for _, gaps in measured for gap in gaps])))
        degraded = [c for m in group for seed in range(args.seeds)
                    for c in degraded_candidates(m, params, seed)]
        print("\n".join(report("degraded, no tags", degraded, args.seeds)))
        print("\n".join(report("degraded, with tags", degraded, args.seeds, use_tags=True)))
        print("  per match, clean: " + "; ".join(
            "{} play {}/{} dead {}/{}".format(
                m.match_id,
                sum(c.passes for c in clean if c.match_id == m.match_id and not c.crossed
                    and c.kind == "play"),
                sum(1 for c in clean if c.match_id == m.match_id and not c.crossed
                    and c.kind == "play"),
                sum(c.passes for c in clean if c.match_id == m.match_id and not c.crossed
                    and c.kind == "dead"),
                sum(1 for c in clean if c.match_id == m.match_id and not c.crossed
                    and c.kind == "dead")) for m in group))


if __name__ == '__main__':
    main()
