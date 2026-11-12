"""Does the real match look like what M4 was trained on? Feature by feature, no labels needed.
Reproduces the drift table in DESIGN.md section 4, "M4".

    MTAH_DATA_DIR=/path/to/data python3 -I solution/experiments/learned_drift.py \
        --dfl-dir /path/to/dfl

For every feature of possession/learned.py: its quantiles on the real match's play frames (graded,
outside the rules' stoppages, with tags) and on the DFL calibration matches' (degraded with
run_devset's seeds 0 and 1, with tags), and the population stability index of the real match
against DFL's deciles. Only aggregates are printed. A PSI of 0.25 or more is flagged as drift, 0.10
to 0.25 as some.

Also: how many splits of the shipped model use each feature, which says what it leans on.

What no marginal can show: the real ball runs 15-30 frames behind the real one, identities switch
inside a slot, and the ball sticks on the centre spot (devset/degrade.py). A feature keyed to fine
ball-player geometry can carry those errors with the same distribution as on DFL.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from devset import degrade, dfl, train_model  # noqa: E402  (needs the path above)
from devset.run_devset import table  # noqa: E402
from possession import atd, deadball, labels, learned, pipeline  # noqa: E402
from possession.repo import load_smart_tagging, pm  # noqa: E402

QUANTILES = (5, 25, 50, 75, 95)
DFL_SEEDS = (0, 1)
DRIFT, SOME = 0.25, 0.10


def psi(reference, other, bins=10):
    """Population stability index of other against reference's quantile bins (merged where they
    coincide, so a feature with few values gets one bin per value)."""
    edges = np.unique(np.percentile(reference, np.linspace(0, 100, bins + 1)[1:-1]))
    expected = np.bincount(np.searchsorted(edges, reference, side="right"),
                           minlength=len(edges) + 1) / float(len(reference))
    actual = np.bincount(np.searchsorted(edges, other, side="right"),
                         minlength=len(edges) + 1) / float(len(other))
    expected, actual = np.maximum(expected, 1e-4), np.maximum(actual, 1e-4)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def real_features(data_dir):
    tracking = atd.load(data_dir=data_dir, verbose=False)
    tags = load_smart_tagging.load(data_dir=data_dir)
    blocks = labels.graded_blocks()
    found = pipeline.rules(tracking, blocks, tags)
    dead = deadball.dead_mask(tracking.n_frames, found["stoppages"])
    codes = labels.to_codes(found["owner"], dead, tracking.n_frames, blocks)
    play = np.isin(codes, (pm.H, pm.A))
    return learned.evidence(found, blocks)[play]


def dfl_features(dfl_dir):
    params = degrade.load_params()
    out = []
    for match_id in dfl.CALIBRATION:
        match = dfl.load(dfl_dir, match_id, verbose=False)
        for seed in DFL_SEEDS:
            one = train_model.Prepared(match, params, seed)
            out.append(one.features[np.isin(one.codes, (pm.H, pm.A))])
    return np.concatenate(out)


def split_counts(model):
    counts = np.zeros(len(learned.names()), dtype=int)
    if model["kind"] == "trees":
        for tree in model["trees"]:
            for feature in tree["feature"]:
                if feature >= 0:
                    counts[feature] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--data-dir", default=None, help="data bundle root (default: games.py)")
    args = parser.parse_args()

    real, reference = real_features(args.data_dir), dfl_features(args.dfl_dir)
    counts = split_counts(learned.load())
    rows, flagged = [], []
    for column, name in enumerate(learned.names()):
        value = psi(reference[:, column], real[:, column])
        verdict = "drift" if value >= DRIFT else "some" if value >= SOME else ""
        if verdict:
            flagged.append(name)
        rows.append([name, counts[column],
                     " / ".join("{:.2f}".format(v)
                                for v in np.percentile(real[:, column], QUANTILES)),
                     " / ".join("{:.2f}".format(v)
                                for v in np.percentile(reference[:, column], QUANTILES)),
                     "{:.3f}".format(value), verdict])
    print("## Features on play frames: real match ({} frames) against DFL calibration ({} frames, "
          "seeds {})\n".format(len(real), len(reference), ", ".join(map(str, DFL_SEEDS))))
    print(table(["feature", "model splits", "real q5 / q25 / q50 / q75 / q95",
                 "DFL q5 / q25 / q50 / q75 / q95", "PSI", "flag"], rows))
    print("\nflagged: {}".format(", ".join(flagged) or "none"))


if __name__ == "__main__":
    main()
