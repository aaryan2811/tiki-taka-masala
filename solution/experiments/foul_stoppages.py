"""Stoppages with neither a line crossing nor a set-piece tag, what the tracking can see of them, and
what M2b does. Reproduces the numbers in DESIGN.md section 4, "M2b: foul stoppages".

    python solution/experiments/foul_stoppages.py
    python solution/experiments/foul_stoppages.py --dfl J03WMX.npz J03WPY.npz

Needs the match data (like run.py). Five parts, the first three over the whole match and without
ground truth:

  candidates  every slowdown candidate: what confirms it under M2 and under M2b, and how deep and
              long the ones M2 confirms and the ones it leaves are.
  added       the stoppages M2b adds: length, how they end, the ball, the players either side.
  restarts    the ball-based restart signal that was measured and not shipped: kicks ending
              confirmed stoppages against kicks in fast open play.
  versions    M3b, then the slowdown test, then the join: public-window metrics and full-match
              statistics, as in the table in DESIGN.md.
  dfl         (with --dfl) the same candidates on two open DFL matches, against their per-frame
              BallStatus: where DEEP_SLOW_SPEED, LONG_SLOW_FRAMES and JOIN_GAP come from. The files
              were made by the DFL feasibility audit's parser (a research note kept out of this
              line) and live outside the repository. solution/devset/foul_check.py repeats the same
              measurement from the dev set's own parser (devset/dfl.py).
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate  # noqa: E402
from possession import atd, config, deadball, keepers, labels, pipeline, repo  # noqa: E402
from possession import ball as ball_stage  # noqa: E402
from possession.repo import pm  # noqa: E402
from possession.runs import true_runs  # noqa: E402

DFL_HALVES = ("firstHalf", "secondHalf")
DFL_PITCH = (105.0, 68.0)  # both audited matches; their positions have the origin at the centre
KMH = 3.6  # DFL player speed is in km/h

# The ball-based restart signal (restarts part): a kick after at least this much rest, with the ball
# unseen for at least this share of it, taken by someone other than a keeper.
REST_FRAMES = 150  # 6 s
UNSEEN_SHARE = 0.8
OPEN_PLAY_CLEAR = 200  # frames: no slowdown this long before an open-play kick, as RESTART_BACK


def graded_mask(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        mask[start:stop] = True
    return mask


def deepest(collective, start, stop):
    return float(np.nanmin(collective[start:stop]))


def prepare(tracking, blocks):
    """(split tracking, repaired ball, keeper columns): keepers.split appends two columns, one per
    team, for each keeper slot the XML flags."""
    split = keepers.split(tracking, blocks)
    ball, _ = ball_stage.repair(split.ball)
    n_slots = len(split.team_of_slot)
    return split, ball, set(range(n_slots - 2 * len(tracking.keeper_slots), n_slots))


def confirmations(split, ball, blocks, tags):
    """Per candidate: (start, stop, crossed, tagged, slow), with M2's windows."""
    graded = graded_mask(split.n_frames, blocks)
    collective = deadball.collective_speed(split.speed)
    crossings = deadball.line_crossings(ball, split.pitch)
    pieces = [tag for tag in tags if (tag["code"] or "").strip().lower() == deadball.SET_PIECES]
    out = []
    for start, stop in deadball.slow_runs(collective, graded):
        crossed = bool(np.any((crossings >= start - config.OOB_BEFORE) & (crossings < stop)))
        tagged = any(start - config.TAG_BEFORE <= tag["start_frame"] <= stop + config.TAG_AFTER
                     for tag in pieces)
        out.append((start, stop, crossed, tagged,
                    deadball.beyond_open_play(collective, start, stop)))
    return out, collective


def candidates_part(split, ball, blocks, tags):
    found, collective = confirmations(split, ball, blocks, tags)
    m2 = [c for c in found if c[2] or c[3]]
    left = [c for c in found if not (c[2] or c[3])]
    print("candidates  ({} slowdowns below {} m/s for {} s or more)".format(
        len(found), config.SLOW_SPEED, config.SLOW_MIN_FRAMES // config.FPS))
    print("  confirmed by M2 (crossing or tag): {}; of these the slowdown test also passes {}".format(
        len(m2), sum(c[4] for c in m2)))
    print("  left by M2: {}; the slowdown test passes {}".format(len(left), sum(c[4] for c in left)))
    for name, group in (("confirmed by M2", m2), ("left by M2", left)):
        seconds = np.array([(stop - start) / float(config.FPS) for start, stop, _, _, _ in group])
        low = np.array([deepest(collective, start, stop) for start, stop, _, _, _ in group])
        print("  {:<16} length s  p25 {:5.1f} median {:5.1f} p75 {:5.1f};  lowest collective "
              "speed p25 {:.2f} median {:.2f} p75 {:.2f} m/s".format(
                  name, *(list(np.percentile(seconds, [25, 50, 75]))
                          + list(np.percentile(low, [25, 50, 75])))))
    window = np.flatnonzero(pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT)) != pm.X)
    public = [c for c in left if c[0] <= window[-1] and c[1] > window[0]]
    print("  left by M2 inside the public window: " + ", ".join(
        "{}-{} ({:.1f} s, lowest {:.2f} m/s, test {})".format(
            start, stop, (stop - start) / float(config.FPS), deepest(collective, start, stop),
            "passes" if slow else "fails") for start, stop, _, _, slow in public))


def added_part(split, ball, blocks, tags):
    before = deadball.detect(split, ball, blocks, tags, foul_stoppages=False)
    after = deadball.detect(split, ball, blocks, tags)
    collective = deadball.collective_speed(split.speed)
    kicked = set(deadball.kicks(ball).tolist())
    seen = ~np.isnan(ball[:, 0])
    old = deadball.dead_mask(split.n_frames, before)
    new = [s for s in after if not old[s.start:s.stop].any()]
    seconds = sorted((s.stop - s.start) / float(config.FPS) for s in new)
    print("\nadded  ({} stoppages before, {} after; {} wholly new, median {:.1f} s, "
          "{} of them shorter than 10 s)".format(len(before), len(after), len(new),
                                                 seconds[len(seconds) // 2],
                                                 sum(s < 10 for s in seconds)))
    print("  {:>15} {:>6} {:>5} {:>6} {:>11} {:>11} {:>5} {:>5}".format(
        "frames", "s", "team", "lowest", "10 s before", "10 s after", "ball", "kick"))
    for s in new:
        print("  {:>7}-{:<7} {:>6.1f} {:>5} {:>6.2f} {:>11.2f} {:>11.2f} {:>5.2f} {:>5}".format(
            s.start, s.stop, (s.stop - s.start) / float(config.FPS),
            "-" if s.team == deadball.NOBODY else split.team_names[s.team][:5],
            deepest(collective, s.start, s.stop),
            float(np.nanmean(collective[max(0, s.start - 250):s.start])),
            float(np.nanmean(collective[s.stop:s.stop + 250])),
            float(seen[s.start:s.stop].mean()), "yes" if s.stop in kicked else "no"))
    grown = [s for s in after if old[s.start:s.stop].any() and not np.array_equal(
        old[s.start:s.stop], np.ones(s.stop - s.start, dtype=bool))]
    print("  existing stoppages lengthened: {} ({} frames)".format(
        len(grown), int(sum((~old[s.start:s.stop]).sum() for s in grown))))
    dead_before = deadball.dead_mask(split.n_frames, before)
    dead_after = deadball.dead_mask(split.n_frames, after)
    graded = graded_mask(split.n_frames, blocks)
    print("  dead share of graded frames: {:.1%} -> {:.1%}".format(
        dead_before[graded].mean(), dead_after[graded].mean()))


def restarts_part(split, ball, keeper_columns, blocks, tags):
    """The ball signal that would have to find the short fouls: a long rest with the ball unseen."""
    found, _ = confirmations(split, ball, blocks, tags)
    stoppages = deadball.detect(split, ball, blocks, tags, foul_stoppages=False)
    dead = deadball.dead_mask(split.n_frames, stoppages)
    graded = graded_mask(split.n_frames, blocks)
    in_candidate = np.zeros(split.n_frames, dtype=bool)
    for start, stop, _, _, _ in found:
        in_candidate[start:stop] = True
    kick_frames = deadball.kicks(ball)
    with np.errstate(invalid="ignore"):
        moving = ball_stage.jumps(ball) > config.KICK_SPEED
    seen = ~np.isnan(ball[:, 0])

    def signal(kick):
        last = kick - 1
        while last > 0 and not moving[last]:
            last -= 1
        rest = kick - 1 - last
        gap = np.hypot(*(split.players[kick - 1] - ball[kick - 1]).T)
        taker = int(np.nanargmin(gap))
        return (rest >= REST_FRAMES and 1 - seen[last + 1:kick].mean() >= UNSEEN_SHARE
                and taker not in keeper_columns)

    ending = [s.stop for s in stoppages if s.evidence != "known" and s.stop in set(kick_frames)]
    open_play = [k for k in kick_frames if graded[k] and not in_candidate[k - OPEN_PLAY_CLEAR:k].any()
                 and not dead[max(0, k - 2 * OPEN_PLAY_CLEAR):k + 1].any()]
    print("\nrestarts  (kick after {:.0f} s or more of rest, ball unseen for {:.0%} of it, "
          "not a keeper)".format(REST_FRAMES / float(config.FPS), UNSEEN_SHARE))
    for name, kicks in (("kicks ending M2's stoppages", ending),
                        ("kicks in fast open play", open_play)):
        print("  {:<28} {:>3} of {}".format(name, sum(signal(k) for k in kicks), len(kicks)))
    collective = deadball.collective_speed(split.speed)
    would = []
    for start, stop, crossed, tagged, slow in found:
        if crossed or tagged or slow:
            continue
        window = kick_frames[(kick_frames >= max(start + config.MIN_DEAD_FRAMES,
                                                 stop - config.RESTART_BACK))
                             & (kick_frames < stop + config.RESTART_AHEAD)]
        if len(window) and signal(int(window[-1])):
            would.append("{}-{} ({:.1f} s, lowest {:.2f} m/s)".format(
                start, int(window[-1]), (stop - start) / float(config.FPS),
                deepest(collective, start, stop)))
    print("  candidates left by M2b it would add: " + (", ".join(would) or "none"))


def versions_part(tracking, blocks, tags):
    gt = pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT))
    public = gt != pm.X
    print("\nversions  (public window, bootstrap 1000; then the full match)")
    original = deadball.join
    try:
        for name, options, joined in (("M3b", dict(foul_stoppages=False), True),
                                      ("+ slowdown test", {}, False),
                                      ("M2b: + join", {}, True)):
            deadball.join = original if joined else (lambda stoppages, kick_frames: stoppages)
            codes, info = pipeline.label_frames(tracking, blocks, tags, learned_model=False,
                                                **options)
            result = pm.evaluate(gt, codes, bootstrap_n=1000)
            predicted, true = public & (codes == pm.D), public & (gt == pm.D)
            print("  {}".format(name))
            print("    spellF1 {:.3f} [{:.2f}, {:.2f}]  macroF1 {:.3f}  possessions {} (true {})"
                  .format(result["spell_f1_mean"], result["spell_bootstrap_low"],
                          result["spell_bootstrap_high"], result["macro_f1"],
                          result["n_pred_spells"], result["n_gt_spells"]))
            print("    spell F1 at IoU {}: {}".format(
                " / ".join(str(t) for t in pm.SPELL_IOUS),
                " / ".join("{:.3f}".format(result["spell_f1"][t]["f1"]) for t in pm.SPELL_IOUS)))
            print("    transition F1 " + "  ".join(
                "+-{}: {:.2f} ({} of {})".format(tau, result["transitions"][tau]["f1"],
                                                 result["transitions"][tau]["n_matched"],
                                                 result["transitions"][tau]["n_gt"])
                for tau in pm.TAUS))
            print("    D frames: precision {:.3f}  recall {:.3f}".format(
                (predicted & true).sum() / float(predicted.sum()),
                (predicted & true).sum() / float(true.sum())))
            print("    full match: {} stoppages; ".format(len(info["stoppages"]))
                  + evaluate.full_match(codes)[4].strip())
    finally:
        deadball.join = original


def dfl_halves(paths):
    """(name, speed m/s, alive, ball in our corner-origin metres) per half of each DFL file."""
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            for half in DFL_HALVES:
                ball = np.stack([stored[half + "_ball_X"] + DFL_PITCH[0] / 2,
                                 stored[half + "_ball_Y"] + DFL_PITCH[1] / 2], axis=1)
                yield ("{} {}".format(os.path.basename(path)[:6], half),
                       stored[half + "_pspeed"] / KMH, stored[half + "_ball_BallStatus"] == 1, ball)


def dfl_part(paths):
    """M2's candidates on DFL, against BallStatus. Only the players are used for the slowdown test:
    DFL freezes the ball during every dead ball, so nothing ball-based can be checked there."""
    print("\ndfl  ({})".format(", ".join(os.path.basename(path) for path in paths)))
    rows, speeds, pairs, starts = [], [], [], []
    for name, speed, alive, ball in dfl_halves(paths):
        collective = deadball.collective_speed(speed)
        speeds.append(collective)
        crossings = deadball.line_crossings(ball, DFL_PITCH)
        dead_starts, dead_stops = true_runs(~alive)
        found = deadball.slow_runs(collective, np.ones(len(alive), dtype=bool))
        for start, stop in found:
            crossed = bool(np.any((crossings >= start - config.OOB_BEFORE) & (crossings < stop)))
            share = float(np.mean(~alive[start:stop]))
            rows.append((crossed, share, stop - start, deepest(collective, start, stop),
                         deadball.beyond_open_play(collective, start, stop)))
            if not crossed and share > 0.5 and rows[-1][4]:
                run = np.flatnonzero((dead_starts < stop) & (dead_stops > start))
                starts.append(start - dead_starts[run[0]])
        for (_, stop), (start, _) in zip(found[:-1], found[1:]):
            if not alive[stop - 1]:
                pairs.append((start - stop, float(np.mean(~alive[stop:start]))))

    ours = np.concatenate(speeds)
    print("  collective speed 10th / 50th / 90th percentile: {:.2f} / {:.2f} / {:.2f} m/s".format(
        *np.nanpercentile(ours, [10, 50, 90])))
    crossed = [r for r in rows if r[0]]
    left = [r for r in rows if not r[0]]
    dead = [r for r in left if r[1] > 0.5]
    alive_rows = [r for r in left if r[1] == 0]
    print("  candidates {}: with a crossing {} (wholly alive {}); without {} (mostly dead {}, "
          "wholly alive {})".format(len(rows), len(crossed), sum(r[1] == 0 for r in crossed),
                                    len(left), len(dead), len(alive_rows)))
    print("  wholly alive, without a crossing: longest {:.1f} s (next {:.1f} s), lowest collective "
          "speed {:.2f} m/s".format(*(sorted(r[2] / float(config.FPS) for r in alive_rows)[-1:-3:-1]
                                      + [min(r[3] for r in alive_rows)])))
    for low, long_s in ((0.70, 20), (0.70, 15), (config.DEEP_SLOW_SPEED,
                                                  config.LONG_SLOW_FRAMES // config.FPS)):
        passes = [r for r in left if r[3] < low or r[2] >= long_s * config.FPS]
        frames = sum(r[2] for r in passes)
        print("  below {:.2f} m/s or {} s or more: mostly dead {} of {}, wholly alive {} of {}, "
              "dead frames {:.0%}".format(low, long_s, sum(r[1] > 0.5 for r in passes), len(dead),
                                          sum(r[1] == 0 for r in passes), len(alive_rows),
                                          sum(r[2] * r[1] for r in passes) / float(frames)))
    starts = np.array(starts)
    print("  slowdown start minus dead start, for those it confirms: median {:.0f} frames; "
          "{} of {} start before the stoppage, earliest {} frames".format(
              np.median(starts), int(np.sum(starts < 0)), len(starts), int(starts.min())))
    print("  consecutive slowdowns, the first ending with play dead: gap length -> share of the gap "
          "dead")
    for low, high in ((0, config.JOIN_GAP), (config.JOIN_GAP, 2 * config.JOIN_GAP),
                      (2 * config.JOIN_GAP, 5 * config.JOIN_GAP)):
        group = [share for gap, share in pairs if low <= gap < high]
        print("    {:>4.1f}-{:<4.1f} s: {} pairs, gap dead throughout in {}".format(
            low / float(config.FPS), high / float(config.FPS), len(group),
            sum(share == 1.0 for share in group)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl", nargs="*", default=(),
                        help="the dev-set audit's DFL .npz files, for the dfl part")
    args = parser.parse_args()

    tracking = atd.load(verbose=False)
    tags = repo.load_smart_tagging.load()
    blocks = labels.graded_blocks()
    split, ball, keeper_columns = prepare(tracking, blocks)
    candidates_part(split, ball, blocks, tags)
    added_part(split, ball, blocks, tags)
    restarts_part(split, ball, keeper_columns, blocks, tags)
    versions_part(tracking, blocks, tags)
    if args.dfl:
        dfl_part(args.dfl)


if __name__ == '__main__':
    main()
