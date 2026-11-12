"""Score a submission on the public window, and check that its full match looks like football.

    python solution/evaluate.py evaluation/results/submission.csv
    python solution/evaluate.py evaluation/results/submission.csv --bootstrap 1000

Two halves, for two different questions:

  public window   17 possessions of real ground truth. Enough to show the pipeline is *correct*
                  -- frames aligned, labels meaning what we think -- and not enough to choose
                  between two versions of it (EVALUATION.md). Read the bootstrap interval, not the
                  point estimate.
  full match      no ground truth at all, only what the file claims: possessions per minute,
                  durations, how many are implausibly short, how much is dead ball. A file can
                  score well on five minutes and still claim a match nobody played.

The official board is still evaluation/run_evaluation.py; this adds the full-match view it does
not have.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from possession import labels, repo  # noqa: E402  (needs the path above)
from possession.repo import pm  # noqa: E402

SHORT_SPELLS_S = (1, 2)  # report possessions shorter than these many seconds


def public_window(codes, bootstrap_n):
    """The headline metrics against the public ground truth, as printable lines."""
    gt = pm.label_codes(pm.read_label_csv(repo.PUBLIC_GT))
    result = pm.evaluate(gt, codes, bootstrap_n=bootstrap_n)
    lines = ["public window  ({} graded frames, {} true possessions)".format(
        result["n_graded"], result["n_gt_spells"])]
    spell = "  spellF1  {:.4f}".format(result["spell_f1_mean"])
    macro = "  macroF1  {:.4f}".format(result["macro_f1"])
    if bootstrap_n:
        spell += "   90% CI [{:.4f}, {:.4f}]".format(result["spell_bootstrap_low"],
                                                    result["spell_bootstrap_high"])
        macro += "   90% CI [{:.4f}, {:.4f}]".format(result["bootstrap_low"],
                                                    result["bootstrap_high"])
    lines += [spell, macro,
              "  possessions {} vs {} true, mean {:.1f} s vs {:.1f} s".format(
                  result["n_pred_spells"], result["n_gt_spells"],
                  result["mean_spell_duration_s"], result["gt_mean_spell_duration_s"]),
              "  dead ball F1 {:.4f}   alive/dead F1 {:.4f}".format(
                  result["per_class"]["D"]["f1"], result["alive_dead_f1"])]
    return lines


def full_match(codes):
    """What the file claims about the whole match, per half and in total, as printable lines.

    Abstentions are filled the way the scorer fills them, so the counts are the ones spellF1 sees.
    """
    blocks = labels.graded_blocks()
    filled = pm.fill_abstentions(codes, blocks)
    rows = [("half {}".format(i + 1), [block]) for i, block in enumerate(blocks)]
    rows.append(("match", blocks))

    header = "  {:<7} {:>6} {:>7} {:>8} {:>8}  {}  {:>6} {:>6}".format(
        "", "poss", "per min", "mean s", "median s",
        "  ".join("<{}s".format(s) for s in SHORT_SPELLS_S), "D", "home")
    lines = ["full match  (no ground truth -- what the file claims)", header]
    for name, runs in rows:
        spells = pm.possession_spells(filled, runs)
        seconds = np.array([stop - start for start, stop, _ in spells]) / float(pm.FPS)
        n_graded = sum(stop - start for start, stop in runs)
        graded = np.concatenate([filled[start:stop] for start, stop in runs])
        alive = np.sum((graded == pm.H) | (graded == pm.A))
        short = "  ".join("{:>3}".format(int(np.sum(seconds < s))) for s in SHORT_SPELLS_S)
        lines.append("  {:<7} {:>6} {:>7.2f} {:>8.1f} {:>8.1f}  {}  {:>5.1%} {:>5.1%}".format(
            name, len(spells), len(spells) / (n_graded / float(pm.FPS) / 60.0),
            seconds.mean() if len(seconds) else 0.0,
            np.median(seconds) if len(seconds) else 0.0, short,
            np.mean(graded == pm.D), np.sum(graded == pm.H) / float(alive) if alive else 0.0))

    meta_path = repo.PUBLIC_GT.replace(".csv", "_meta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    lines.append("  for scale, the public window's truth: {:.2f} possessions per minute, mean "
                 "{:.1f} s".format(meta["n_possessions"] / meta["public_window"]["minutes_of_play"],
                                   meta["mean_possession_duration_s"]))
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="the frame,label CSV")
    parser.add_argument("--bootstrap", type=int, default=0,
                        help="draws for confidence intervals on the public window (try 1000)")
    args = parser.parse_args()

    reason = labels.check(args.submission)
    if reason:
        print("FAIL  {}: {}".format(args.submission, reason))
        sys.exit(1)
    codes = pm.label_codes(pm.read_label_csv(args.submission))

    print("{}\n".format(os.path.basename(args.submission)))
    print("\n".join(public_window(codes, args.bootstrap)))
    print("")
    print("\n".join(full_match(codes)))


if __name__ == '__main__':
    main()
