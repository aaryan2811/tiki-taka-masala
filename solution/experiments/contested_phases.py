"""Which touches the truth refuses to credit, what the tracking can see of them, and what M3b does.
Reproduces the numbers in DESIGN.md section 4, "M3b: contested phases".

    python solution/experiments/contested_phases.py

Needs the match data (like run.py). Four parts, all over the whole match and without ground truth
except the last:

  keepers     how often each goalkeeper slot stands in the other team's half, and how fast it moves
              when it changes keeper, against outfield slots (KEEPER_MAX_SPEED).
  duels       single controlled touches between two touches of the other team: how long until the
              other team is back on the ball, and how near its player was during the touch
              (CONTEST_GAP, CONTEST_RADIUS).
  effects     what each step changes over the whole match, and how the result moves with the duel
              settings either side of the chosen ones.
  versions    M3, then keepers split, then duels resolved: the public-window metrics and the
              full-match statistics, as in the table in DESIGN.md.
"""

import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate  # noqa: E402
from possession import atd, config, contest, control, deadball, keepers, labels, pipeline, repo  # noqa: E402,E501
from possession import ball as ball_stage  # noqa: E402
from possession.features import nearest_by_team, relative_speed, velocity  # noqa: E402
from possession.repo import pm  # noqa: E402

FAST = 12.0  # m/s, reported against KEEPER_MAX_SPEED
WAIT_BINS = (0, 3, 6, 12, 25, 50, 75, 100, 150, 250, 10 ** 9)  # frames
DISTANCE_BINS = (0, 1, 2, 3, 4, 5, 6, 8, 10, np.inf)  # m


def graded_mask(n_frames, blocks):
    mask = np.zeros(n_frames, dtype=bool)
    for start, stop in blocks:
        mask[start:stop] = True
    return mask


def keeper_part(tracking, blocks):
    graded = graded_mask(tracking.n_frames, blocks)
    home_low = keepers.home_defends_low_x(tracking.n_frames, blocks,
                                          tracking.home_attacks_right_first_half)
    middle = tracking.pitch[0] / 2.0
    speed = np.hypot(*velocity(tracking.players).transpose(2, 0, 1))
    outfield = [slot for slot in range(len(tracking.team_of_slot))
                if slot not in tracking.keeper_slots]
    steps = speed[graded][:, outfield]
    steps = steps[~np.isnan(steps)]
    print("keepers  (graded frames)")
    print("  outfield slots move at most {:.1f} m/s at the 99.9th percentile".format(
        np.percentile(steps, 99.9)))
    for slot in tracking.keeper_slots:
        team = tracking.team_of_slot[slot]
        x = tracking.players[:, slot, 0]
        with np.errstate(invalid="ignore"):
            own_low = home_low if team == 0 else ~home_low
            wrong = graded & ~np.isnan(x) & ((x < middle) != own_low)
            fast = graded & (speed[:, slot] > FAST)
        print("  slot {} (XML: {} keeper): in the other team's half on {} frames; {} frames "
              "faster than {:.0f} m/s".format(slot, tracking.team_names[team], int(wrong.sum()),
                                              int(fast.sum()), FAST))


def touches_and_distance(tracking, blocks, tags):
    split = keepers.split(tracking, blocks)
    ball, _ = ball_stage.repair(split.ball)
    distance, slot = nearest_by_team(ball, split.players, split.team_of_slot)
    on_ball = control.on_ball_team(distance, relative_speed(ball, split.players, slot))
    stoppages = deadball.detect(split, ball, blocks, tags)
    alive = graded_mask(split.n_frames, blocks) & ~deadball.dead_mask(split.n_frames, stoppages)
    block_of = np.full(split.n_frames, -1)
    for index, (start, stop) in enumerate(blocks):
        block_of[start:stop] = index
    found = [touch for touch in control.touches(on_ball) if alive[touch[0]]]
    return found, distance, alive, block_of


def histogram(values, bins, unit):
    counts, _ = np.histogram(values, bins)
    return "  ".join("{}-{}{}: {}".format(low, "" if high >= 10 ** 9 else high, unit, count)
                     for low, high, count in zip(bins[:-1], bins[1:], counts))


def duel_part(tracking, blocks, tags):
    found, distance, alive, block_of = touches_and_distance(tracking, blocks, tags)

    waits = []
    for before, touch, after in zip(found[:-2], found[1:-1], found[2:]):
        if not (before[2] == after[2] != touch[2]
                and block_of[before[0]] == block_of[touch[0]] == block_of[after[0]]
                and alive[before[1]:after[0]].all()):
            continue
        waits.append(after[0] - touch[1])
    waits = np.array(waits)
    print("\nduels  (keeper slots split)")
    print("  single touches between two touches of the other team, no dead ball between: {}".format(
        len(waits)))
    print("  frames until the other team's touch begins: " + histogram(waits, WAIT_BINS, ""))
    print("  shortest wait above {} frames: {}".format(config.CONTEST_GAP,
                                                       waits[waits > config.CONTEST_GAP].min()))

    direct, later = [], []
    for touch, following in zip(found[:-1], found[1:]):
        if following[2] == touch[2]:
            continue
        other = distance[touch[0]:touch[1], 1 - touch[2]]
        farthest = np.max(np.where(np.isnan(other), np.inf, other))
        (direct if following[0] - touch[1] <= config.CONTEST_GAP else later).append(farthest)
    print("  touches the other team took over within {} frames: {}; later: {}".format(
        config.CONTEST_GAP, len(direct), len(later)))
    print("  farthest the other team's nearest player got from the ball during the touch:")
    print("    direct  " + histogram(direct, DISTANCE_BINS, " m"))
    print("    later   " + histogram(later, DISTANCE_BINS, " m"))
    print("    later, median {:.1f} m".format(np.median(later)))


def label_changes(codes, blocks):
    out = set()
    for start, stop in blocks:
        for frame in np.flatnonzero(codes[start + 1:stop] != codes[start:stop - 1]) + start + 1:
            out.add((int(frame), pm.LABELS[codes[frame - 1]], pm.LABELS[codes[frame]]))
    return out


def effects_part(tracking, blocks, tags):
    m3, m3_info = pipeline.label_frames(tracking, blocks, tags, split_keepers=False, duels=False,
                                         learned_model=False)
    kept, kept_info = pipeline.label_frames(tracking, blocks, tags, duels=False,
                                            learned_model=False)
    m3b, m3b_info = pipeline.label_frames(tracking, blocks, tags, learned_model=False)
    print("\neffects  (whole match)")
    for name, before, after in (("keepers split", m3, kept), ("duels resolved", kept, m3b)):
        old, new = label_changes(before, blocks), label_changes(after, blocks)
        print("  {}: {} label changes removed, {} added".format(name, len(old - new),
                                                                len(new - old)))
    restarts = [(a.start, a.team, b.team)
                for a, b in zip(m3_info["stoppages"], kept_info["stoppages"]) if a.team != b.team]
    changed = ", ".join("stoppage at {} ({} -> {})".format(start, old, new)
                        for start, old, new in restarts)
    print("  restart team changed by the keeper split: {}".format(changed or "none"))
    print("  void touches: {}".format(len(m3b_info["void"])))

    print("  duel settings either side of the chosen ones (changes of possession, possessions, "
          "under 2 s):")
    original = contest.resolve
    try:
        for gap in (0, config.CONTEST_GAP, 4, 8):
            line = []
            for radius in (2.0, config.CONTEST_RADIUS, 4.0):
                contest.resolve = functools.partial(original, gap=gap, radius=radius)
                codes, info = pipeline.label_frames(tracking, blocks, tags, learned_model=False)
                spells = pm.possession_spells(pm.fill_abstentions(codes, blocks), blocks)
                short = sum(stop - start < 2 * pm.FPS for start, stop, _ in spells)
                line.append("{:.1f} m: {} / {} / {}".format(radius, info["changes"],
                                                            len(spells), short))
            print("    gap {}  ".format(gap) + "   ".join(line))
    finally:
        contest.resolve = original


def versions_part(tracking, blocks, tags):
    gt = pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT))
    print("\nversions  (public window, bootstrap 1000; then the full match)")
    for name, options in (("M3", dict(split_keepers=False, duels=False)),
                          ("+ keepers split", dict(duels=False)),
                          ("duels only", dict(split_keepers=False)),
                          ("M3b: keepers + duels", {})):
        codes, info = pipeline.label_frames(tracking, blocks, tags, learned_model=False,
                                            **options)
        result = pm.evaluate(gt, codes, bootstrap_n=1000)
        print("  {}".format(name))
        print("    spellF1 {:.3f} [{:.2f}, {:.2f}]  macroF1 {:.3f}  possessions {} (true {})"
              .format(result["spell_f1_mean"], result["spell_bootstrap_low"],
                      result["spell_bootstrap_high"], result["macro_f1"], result["n_pred_spells"],
                      result["n_gt_spells"]))
        print("    spell F1 at IoU {}: {}".format(
            " / ".join(str(t) for t in pm.SPELL_IOUS),
            " / ".join("{:.3f}".format(result["spell_f1"][t]["f1"]) for t in pm.SPELL_IOUS)))
        print("    transition F1 " + "  ".join(
            "+-{}: {:.2f} ({} of {})".format(tau, result["transitions"][tau]["f1"],
                                             result["transitions"][tau]["n_matched"],
                                             result["transitions"][tau]["n_gt"])
            for tau in pm.TAUS))
        print("    full match: " + evaluate.full_match(codes)[4].strip())
        print("    changes {}, void touches {}".format(info["changes"], len(info["void"])))


def main():
    tracking = atd.load(verbose=False)
    tags = repo.load_smart_tagging.load()
    blocks = labels.graded_blocks()
    keeper_part(tracking, blocks)
    duel_part(tracking, blocks, tags)
    effects_part(tracking, blocks, tags)
    versions_part(tracking, blocks, tags)


if __name__ == '__main__':
    main()
