"""M4's target: is a short spell just before the ball goes dead a possession? Two training targets
compared by leaving one calibration match out. Reproduces DESIGN.md section 4, "Re-checked on DFL",
item 5.

    python3 -I solution/experiments/target_dead_ball.py --dfl-dir /path/to/dfl

The merged truths (devset/truth.py) absorb a spell shorter than 2 s or 3 s lying between two spells
of the other team. A spell with a dead ball on one side is never absorbed. After a dead ball the
challenge's definition settles that: play restarts with a clear possessor. Before one it does not,
so this script tries the other answer: the same truth with every spell shorter than the threshold
that lies between the other team's spell and a dead ball absorbed into the other team's spell
("+out").

  current   trees trained on merged2s, as train_model.py ships
  +out      trees trained on merged2s+out

Everything else is learned_gate.py's: two folds (train on one calibration match, test on the other),
TRAIN_SEEDS with tags for training, switch cost and minimum length tuned on the training match with
the objective train_model.OBJECTIVE (raw and merged3s as truth.py builds them), the test match with
run_devset.py's seeds 0-4, with and without tags. Both folds pooled. The held-out five are never
read. Scored under raw, merged2s, merged3s, merged2s+out and merged3s+out; the decision reads +out
minus current under raw, merged3s and merged3s+out, with and without tags. M2c and current minus
M2c are printed too, so the gate's row can be checked against DESIGN.md.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devset import degrade, dfl, score, train_model, truth  # noqa: E402  (needs the path above)
from devset.run_devset import TAG_MODES, table  # noqa: E402
from possession import config, learned  # noqa: E402
from possession.repo import pm  # noqa: E402

FPS = config.FPS
TEST_SEEDS = range(5)  # run_devset.py's
M2C, CURRENT, OUT = "M2c", "current", "+out"
VARIANTS = {CURRENT: "merged2s", OUT: "merged2s+out"}  # name: training target
TRUTHS = ("raw", "merged2s", "merged3s", "merged2s+out", "merged3s+out")
DECIDING = ("raw", "merged3s", "merged3s+out")


def merge(codes, seconds, before_dead=False, fps=FPS):
    """truth.merge_lost_back, and with before_dead also every spell shorter than `seconds` whose
    left neighbour is the other team's spell and right neighbour a dead ball, absorbed into the
    left one. Same scan: left to right, the joined spell checked again."""
    if seconds <= 0:
        return codes.copy(), 0
    limit = seconds * fps
    breaks = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(codes)]))
    runs = [[int(bounds[i]), int(bounds[i + 1]), int(codes[bounds[i]])]
            for i in range(len(bounds) - 1)]
    absorbed = 0
    i = 1
    while i < len(runs) - 1:
        before, middle, after = runs[i - 1], runs[i], runs[i + 1]
        short = (middle[2] in pm.POSSESSION_LABELS and middle[1] - middle[0] < limit
                 and before[2] in pm.POSSESSION_LABELS and before[2] != middle[2])
        if short and before[2] == after[2]:
            runs[i - 1:i + 2] = [[before[0], after[1], before[2]]]
            i = max(1, i - 1)
        elif short and before_dead and after[2] == pm.D:
            runs[i - 1:i + 1] = [[before[0], middle[1], before[2]]]
            absorbed += 1
            i = max(1, i - 1)
        else:
            i += 1
    out = np.empty_like(codes)
    for start, stop, code in runs:
        out[start:stop] = code
    return out, absorbed


def match_truths(match):
    """truth.variants(match) plus merged2s+out and merged3s+out, and how many spells +out absorbs
    at each threshold."""
    out = truth.variants(match)
    n_frames, blocks = degrade.layout(match)
    absorbed = {}
    for name, seconds in (("merged2s", 2.0), ("merged3s", 3.0)):
        codes = np.full(n_frames, pm.X, dtype=np.int8)
        same = np.full(n_frames, pm.X, dtype=np.int8)
        count = 0
        for half, (start, stop) in zip(match.halves, blocks):
            codes[start:stop], n = merge(truth.half_codes(half), seconds, before_dead=True)
            same[start:stop], _ = merge(truth.half_codes(half), seconds)
            count += n
        assert np.array_equal(same, out[name]), "merge() must reproduce truth.py without +out"
        out[name + "+out"] = codes
        absorbed[name] = (count, int(np.sum(codes != out[name])))
    return out, absorbed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    args = parser.parse_args()

    params = degrade.load_params()
    matches = {match_id: dfl.load(args.dfl_dir, match_id, verbose=False)
               for match_id in dfl.CALIBRATION}
    truths, absorbed = {}, []
    for match_id, match in matches.items():
        truths[match_id], counts = match_truths(match)
        for name, (spells, frames) in counts.items():
            absorbed.append([match_id, name + "+out", spells, frames])

    scored, tuned = {}, []
    for test_id in dfl.CALIBRATION:
        train_id = [m for m in dfl.CALIBRATION if m != test_id][0]
        train = [train_model.Prepared(matches[train_id], params, seed, truths=truths[train_id])
                 for seed in train_model.TRAIN_SEEDS]
        models = {}
        for name, target in VARIANTS.items():
            models[name] = train_model.train(train, "trees", target, train_model.GRID)
            tuned.append([name, "train {}, test {}".format(train_id, test_id),
                          models[name]["switch_cost"], models[name]["min_frames"]])
            print("trained {} on {}".format(name, train_id), file=sys.stderr)
        del train

        for seed in TEST_SEEDS:
            for mode, use_tags in TAG_MODES:
                test = train_model.Prepared(matches[test_id], params, seed, use_tags,
                                            truths[test_id])
                predictions = {M2C: test.codes}
                for name, model in models.items():
                    predictions[name] = test.segmented(learned.log_odds(test.features, model),
                                                       model["switch_cost"], model["min_frames"])
                for name, codes in predictions.items():
                    for variant in TRUTHS:
                        by_seed = scored.setdefault((name, mode, variant),
                                                    [[] for _ in TEST_SEEDS])
                        by_seed[seed].append(score.Scored(test.truths[variant], codes))
        print("tested on {}".format(test_id), file=sys.stderr)

    print(report(scored, tuned, absorbed))


def report(scored, tuned, absorbed):
    n_draws = 1000
    headline, paired, cells = [], [], []
    for mode, _ in TAG_MODES:
        for name in (M2C, CURRENT, OUT):
            headline.append([name, mode] + ["{:.3f}".format(
                score.mean_summary(scored[(name, mode, v)])["spell_f1"]) for v in TRUTHS])
        for a, b in ((CURRENT, M2C), (OUT, M2C), (OUT, CURRENT)):
            row = ["{} - {}".format(a, b), mode]
            for variant in TRUTHS:
                point, low, high, above = score.paired(scored[(a, mode, variant)],
                                                       scored[(b, mode, variant)], n_draws)
                row.append("{:+.3f} [{:+.3f}, {:+.3f}] {:.0%}".format(point, low, high, above))
                if (a, b) == (OUT, CURRENT) and variant in DECIDING:
                    cells.append(point)
            paired.append(row)
    verdict = "+out replaces the target" if min(cells) > 0 else "the current target stays"
    return "\n".join([
        "## Spells +out absorbs in the calibration truths\n",
        table(["match", "truth", "spells absorbed", "frames changed"], absorbed),
        "\n## Leave one calibration match out: both test matches pooled, {} seeds\n".format(
            len(TEST_SEEDS)),
        table(["pipeline", "tags"] + ["spellF1 " + v for v in TRUTHS], headline),
        "\n## Paired: [90% CI], share of draws above zero\n",
        table(["difference", "tags"] + list(TRUTHS), paired),
        "\n## Decision: +out - current above zero under {}, with and without tags\n".format(
            ", ".join(DECIDING)),
        " / ".join("{:+.3f}".format(c) for c in cells) + ": " + verdict,
        "\n## Segmentation tuned in each fold\n",
        table(["variant", "fold", "switch cost", "minimum frames"], tuned),
    ])


if __name__ == "__main__":
    main()
