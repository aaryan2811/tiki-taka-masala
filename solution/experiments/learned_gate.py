"""M4's go/no-go gate: the learned model against M2c, one calibration match left out at a time.
Reproduces the gate tables in DESIGN.md section 4, "M4".

    python3 -I solution/experiments/learned_gate.py --dfl-dir /path/to/dfl
    python3 -I solution/experiments/learned_gate.py --dfl-dir /path/to/dfl \
        --variants "trees, merged2s"

Two folds: train on J03WMX and test on J03WPY, then the other way round. The held-out five are never
read. In each fold, for each variant:

  train   devset/train_model.train on the training match alone: TRAIN_SEEDS, with tags; switch
          cost and minimum length tuned on that match with out-of-half probabilities
  test    the test match degraded with run_devset.py's seeds (TEST_SEEDS), with and without tags;
          the rules run on it (M2c), then the model and segment.decode replace the team on its play

Both folds' test matches are pooled and scored against M2c on the same tracking with score.paired,
under the three truths. The gate (written into DESIGN.md before the first run): a variant passes if
its paired difference is at least GATE spellF1 under raw and merged3s, with and without tags.

Also printed: the switch cost and minimum length each fold tuned, possessions per minute, and the
team on play frames (M2c's, the model's frame by frame, and after segmentation) against the merged
3 s truth, with the ball seen and unseen.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devset import degrade, dfl, score, truth  # noqa: E402  (needs the path above)
from devset import train_model  # noqa: E402
from devset.run_devset import TAG_MODES, table  # noqa: E402
from possession import learned  # noqa: E402
from possession.repo import pm  # noqa: E402

TEST_SEEDS = range(5)  # run_devset.py's
GATE = 0.01
M2C = "M2c"
NO_MINIMUM = {"switch_cost": train_model.GRID["switch_cost"], "min_frames": (1,)}

# name: (model kind, target truth, segmentation grid)
VARIANTS = {
    "trees, merged2s": ("trees", "merged2s", train_model.GRID),
    "trees, merged2s, no minimum length": ("trees", "merged2s", NO_MINIMUM),
    "logistic, merged2s": ("logistic", "merged2s", train_model.GRID),
    "trees, merged3s": ("trees", "merged3s", train_model.GRID),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    params = degrade.load_params()
    matches = {match_id: dfl.load(args.dfl_dir, match_id, verbose=False)
               for match_id in dfl.CALIBRATION}
    truths = {match_id: truth.variants(match) for match_id, match in matches.items()}

    # scored[(pipeline, tag mode, truth)][seed] = [Scored per test match]
    scored, tuned, frames = {}, [], {}
    for test_id in dfl.CALIBRATION:
        train_id = [m for m in dfl.CALIBRATION if m != test_id][0]
        train = [train_model.Prepared(matches[train_id], params, seed, truths=truths[train_id])
                 for seed in train_model.TRAIN_SEEDS]
        models = {}
        for name in args.variants:
            kind, target, grid = VARIANTS[name]
            models[name] = train_model.train(train, kind, target, grid)
            tuned.append([name, "train {}, test {}".format(train_id, test_id),
                          models[name]["switch_cost"], models[name]["min_frames"]])
            print("trained {} on {}".format(name, train_id), file=sys.stderr)
        del train

        for seed in TEST_SEEDS:
            for mode, use_tags in TAG_MODES:
                test = train_model.Prepared(matches[test_id], params, seed, use_tags,
                                            truths[test_id])
                predictions = {M2C: (test.codes, None)}
                for name, model in models.items():
                    log_odds = learned.log_odds(test.features, model)
                    predictions[name] = (test.segmented(log_odds, model["switch_cost"],
                                                        model["min_frames"]), log_odds)
                for name, (codes, log_odds) in predictions.items():
                    for variant, gt in test.truths.items():
                        by_seed = scored.setdefault((name, mode, variant),
                                                    [[] for _ in TEST_SEEDS])
                        by_seed[seed].append(score.Scored(gt, codes))
                    if mode == TAG_MODES[0][0]:
                        count_frames(frames, name, test, codes, log_odds)
        print("tested on {}".format(test_id), file=sys.stderr)

    print(report(scored, tuned, frames, args.variants))


def count_frames(frames, name, test, codes, log_odds):
    """Add right / total counts of the team on play frames, against the merged 3 s truth, by
    whether the ball is seen: after segmentation, and for the model frame by frame."""
    gt = test.truths["merged3s"]
    play = np.isin(test.codes, (pm.H, pm.A)) & np.isin(gt, (pm.H, pm.A))
    seen = test.features[:, learned.names().index("seen")] > 0
    guesses = [(name, codes)]
    if log_odds is not None:
        guesses.append((name + ", frame by frame", np.where(log_odds > 0, pm.H, pm.A)))
    for label, guess in guesses:
        for ball, mask in (("seen", play & seen), ("unseen", play & ~seen)):
            counts = frames.setdefault((label, ball), [0, 0])
            counts[0] += int(np.sum(guess[mask] == gt[mask]))
            counts[1] += int(mask.sum())


def report(scored, tuned, frames, names):
    variants = [name for name, _ in truth.VARIANTS]
    n_draws = 1000
    headline, paired, verdicts = [], [], []
    for mode, _ in TAG_MODES:
        for name in [M2C] + names:
            cells = [score.mean_summary(scored[(name, mode, v)]) for v in variants]
            rate = "{:.2f} ({:.2f} / {:.2f})".format(cells[0]["pred_per_min"],
                                                     cells[0]["gt_per_min"],
                                                     cells[-1]["gt_per_min"])
            headline.append([name, mode] + ["{:.3f}".format(c["spell_f1"]) for c in cells]
                            + [rate])
    for name in names:
        passed = []
        for mode, _ in TAG_MODES:
            cells = [name + " - " + M2C, mode]
            for variant in variants:
                point, low, high, above = score.paired(scored[(name, mode, variant)],
                                                       scored[(M2C, mode, variant)], n_draws)
                cells.append("{:+.3f} [{:+.3f}, {:+.3f}] {:.0%}".format(point, low, high, above))
                if variant in ("raw", "merged3s"):
                    passed.append(point)
            paired.append(cells)
        verdicts.append([name, " / ".join("{:+.3f}".format(p) for p in passed),
                         "{:+.3f}".format(np.mean(passed)),
                         "GO" if min(passed) >= GATE else "no-go"])
    order = sorted(frames, key=lambda key: (key[1], key[0]))  # by ball seen, then pipeline
    accuracy = [[label, ball, "{:.1%}".format(frames[(label, ball)][0]
                                              / float(frames[(label, ball)][1])),
                 frames[(label, ball)][1]] for label, ball in order]
    return "\n".join([
        "## Leave one calibration match out: both test matches pooled, {} seeds\n".format(
            len(TEST_SEEDS)),
        table(["pipeline", "tags"] + ["spellF1 " + v for v in variants]
              + ["poss/min (truth raw / merged3s)"], headline),
        "\n## Paired against M2c: [90% CI], share of draws above zero\n",
        table(["difference", "tags"] + variants, paired),
        "\n## Gate: at least {:+.2f} under raw and merged3s, with and without tags\n".format(GATE),
        table(["variant", "raw tags / merged3s tags / raw no tags / merged3s no tags",
               "mean", "verdict"], verdicts),
        "\n## Segmentation tuned in each fold\n",
        table(["variant", "fold", "switch cost", "minimum frames"], tuned),
        "\n## Team on play frames, against the merged 3 s truth (tags)\n",
        table(["pipeline", "ball", "right", "frames"], accuracy),
    ])


if __name__ == "__main__":
    main()
