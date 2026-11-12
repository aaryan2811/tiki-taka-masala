"""Train M4's per-frame model on the DFL calibration matches; write solution/possession/model.json.

    python3 -I solution/devset/train_model.py --dfl-dir /path/to/dfl

The model (possession/learned.py) reads the evidence the rules leave behind on every frame of play
and gives P(home has the ball). It is trained on the two calibration matches only (dfl.CALIBRATION),
degraded as run_devset.py degrades them but with their own seeds (TRAIN_SEEDS), several per match as
augmentation, each run with synthetic tags. Its target is the truth merged at TARGET: the DFL flag
credits touches lost straight back to the other team, and ours does not (truth.py).

Rows are every STRIDE-th frame the rules call play (graded, outside their stoppages) on which the
truth names a team. Two kinds of model, both symmetric between the teams (learned.py):

  trees     sklearn's HistGradientBoostingClassifier on every row and its swapped copy, with fixed
            settings (TREES), no early stopping and one thread, exported node by node to JSON
  logistic  L2-penalised logistic regression on learned.antisymmetric's columns, each divided by
            its standard deviation, fitted by Newton's method; no intercept

Then segment.py's switch cost and minimum length are tuned on the training matches themselves, with
probabilities each half gets from a model trained on the other half (tune), over GRID. The objective
is the mean of spellF1 under the raw and the merged 3 s truth.

Everything is deterministic, so a rerun writes the same file. Only the DFL data goes in; credit the
DFL under CC BY 4.0 (devset/README.md, "Dataset and licence").
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from devset import adapter, degrade, dfl, score, tags, truth  # noqa: E402  (needs the path above)
from devset.run_devset import match_seed  # noqa: E402
from possession import deadball, labels, learned, pipeline, segment  # noqa: E402
from possession.repo import pm  # noqa: E402

TRAIN_SEEDS = (10, 11, 12, 13)  # never the seeds run_devset.py scores (0-4)
TARGET = "merged2s"
STRIDE = 5  # frames between training rows: neighbouring frames repeat each other
TREES = {"max_iter": 100, "learning_rate": 0.1, "max_leaf_nodes": 15, "min_samples_leaf": 200,
         "l2_regularization": 1.0}
LOGISTIC_L2 = 1.0  # penalty on the standardised weights, in units of the summed log-likelihood
GRID = {"switch_cost": (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
        "min_frames": (1, 25, 50)}
OBJECTIVE = ("raw", "merged3s")  # truths whose spellF1 the tuning averages
CREDIT = ("Trained on the DFL / Sportec open data (Bassek et al. 2025, "
          "https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0, provided with "
          "authorization from the DFL; degraded as solution/devset/degrade.py describes.")


class Prepared(object):
    """One degraded DFL match with the rules run on it: M2c's owner and stoppages, the features on
    every frame, and the truth. features is dropped by keep_rows() once the rows are taken."""

    def __init__(self, match, params, seed, use_tags=True, truths=None):
        self.match_id, self.seed = match.match_id, seed
        tracking, self.blocks = degrade.to_tracking(match, params, match_seed(match.match_id, seed))
        tag_list = tags.synthetic(match, self.blocks, params, match_seed(match.match_id, seed)) \
            if use_tags else ()
        with adapter.no_known_dead_spans():
            found = pipeline.rules(tracking, self.blocks, tag_list)
        self.owner, self.stoppages = found["owner"], found["stoppages"]
        self.n_frames = tracking.n_frames
        self.dead = deadball.dead_mask(self.n_frames, self.stoppages)
        self.codes = labels.to_codes(self.owner, self.dead, self.n_frames, self.blocks)
        self.features = learned.evidence(found, self.blocks)
        self.truths = truths if truths is not None else truth.variants(match)

    def rows(self, target=TARGET, block=None, stride=STRIDE):
        """(features, home) on every stride-th frame of play with a team in the truth; block limits
        them to one graded block."""
        gt = self.truths[target]
        keep = np.isin(self.codes, (pm.H, pm.A)) & np.isin(gt, (pm.H, pm.A))
        if block is not None:
            inside = np.zeros(self.n_frames, dtype=bool)
            inside[slice(*self.blocks[block])] = True
            keep &= inside
        index = np.flatnonzero(keep)[::stride]
        return self.features[index], (gt[index] == pm.H).astype(np.int8)

    def segmented(self, log_odds, switch_cost, min_frames):
        """Codes with the play decoded from log_odds; stoppages as the rules found them."""
        owner = segment.decode(log_odds, self.owner, self.stoppages, self.blocks, switch_cost,
                               min_frames)
        return labels.to_codes(owner, self.dead, self.n_frames, self.blocks)


# -- the two kinds of model --------------------------------------------------------------------------

def fit_trees(features, home, settings=TREES):
    """A trees model dict (learned.log_odds) from rows and their swapped copies."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits

    both = np.concatenate([features, learned.swap(features)])
    target = np.concatenate([home, 1 - home])
    model = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **settings)
    with threadpool_limits(limits=1):  # summation order fixed, so the trees are reproducible
        model.fit(both, target)
    trees = []
    for (predictor,) in model._predictors:
        nodes = predictor.nodes
        leaf = nodes["is_leaf"].astype(bool)
        trees.append({"feature": np.where(leaf, -1, nodes["feature_idx"]).tolist(),
                      "threshold": np.where(leaf, 0.0, nodes["num_threshold"]).tolist(),
                      "left": nodes["left"].astype(int).tolist(),
                      "right": nodes["right"].astype(int).tolist(),
                      "missing_left": nodes["missing_go_to_left"].astype(int).tolist(),
                      "value": np.where(leaf, nodes["value"], 0.0).tolist()})
    return {"kind": "trees", "settings": dict(settings),
            "base": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def fit_logistic(features, home, l2=LOGISTIC_L2, iterations=50):
    """A logistic model dict (learned.log_odds), by Newton's method on the standardised design."""
    design = learned.antisymmetric(features)
    scale = design.std(axis=0)
    scale[scale == 0] = 1.0
    design = design / scale
    weights = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(design @ weights)))
        gradient = design.T @ (home - p) - l2 * weights
        hessian = (design * (p * (1 - p))[:, None]).T @ design + l2 * np.eye(len(weights))
        step = np.linalg.solve(hessian, gradient)
        weights += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return {"kind": "logistic", "l2": l2, "scale": scale.tolist(), "weights": weights.tolist()}


FITS = {"trees": fit_trees, "logistic": fit_logistic}


def fit(prepared, kind="trees", target=TARGET, block=None):
    """A model of `kind` from the rows of every Prepared in the list."""
    rows = [one.rows(target, block) for one in prepared]
    return FITS[kind](np.concatenate([r[0] for r in rows]), np.concatenate([r[1] for r in rows]))


# -- segmentation ------------------------------------------------------------------------------------

def out_of_half(prepared, kind="trees", target=TARGET):
    """{(match, seed): log-odds} where each graded block's come from a model trained on the other
    block of every match in prepared."""
    out = {(one.match_id, one.seed): np.zeros(one.n_frames) for one in prepared}
    for block in (0, 1):
        model = fit(prepared, kind, target, block=1 - block)
        for one in prepared:
            start, stop = one.blocks[block]
            out[(one.match_id, one.seed)][start:stop] = learned.log_odds(
                one.features[start:stop], model)
    return out


def objective(scored_by_truth):
    return float(np.mean([score.summary(scored_by_truth[name])["spell_f1"] for name in OBJECTIVE]))


def tune(prepared, log_odds, grid=GRID):
    """(switch_cost, min_frames, {(switch cost, min frames): objective}): the grid point whose
    segmentation of log_odds scores best on prepared, pooled over its matches and seeds."""
    results = {}
    for min_frames in grid["min_frames"]:
        for switch_cost in grid["switch_cost"]:
            scored = {name: [] for name in OBJECTIVE}
            for one in prepared:
                codes = one.segmented(log_odds[(one.match_id, one.seed)], switch_cost, min_frames)
                for name in OBJECTIVE:
                    scored[name].append(score.Scored(one.truths[name], codes))
            results[(switch_cost, min_frames)] = objective(scored)
    best = max(sorted(results), key=lambda key: results[key])
    return best[0], best[1], results


def train(prepared, kind="trees", target=TARGET, grid=GRID):
    """The model dict to ship: fitted on every row of prepared, with the switch cost and minimum
    length tuned on out-of-half log-odds."""
    switch_cost, min_frames, results = tune(prepared, out_of_half(prepared, kind, target), grid)
    model = fit(prepared, kind, target)
    model.update({"target": target, "switch_cost": switch_cost, "min_frames": min_frames,
                  "features": learned.names(),
                  "tuning": [[c, m, round(v, 6)] for (c, m), v in sorted(results.items())]})
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--kind", default="trees", choices=sorted(FITS))
    parser.add_argument("--out", default=learned.MODEL_PATH, help="where the model goes")
    args = parser.parse_args()

    params = degrade.load_params()
    prepared = []
    for match_id in dfl.CALIBRATION:
        match = dfl.load(args.dfl_dir, match_id, verbose=False)
        truths = truth.variants(match)
        prepared += [Prepared(match, params, seed, truths=truths) for seed in TRAIN_SEEDS]
        print("prepared {}".format(match_id), file=sys.stderr)
    model = train(prepared, args.kind)
    model.update({"trained_on": list(dfl.CALIBRATION), "seeds": list(TRAIN_SEEDS),
                  "stride": STRIDE, "credit": CREDIT})
    with open(args.out, "w") as handle:
        json.dump(model, handle, separators=(",", ":"), sort_keys=True)
        handle.write("\n")
    print("wrote {}: {} model, switch cost {}, minimum {} frames".format(
        os.path.relpath(args.out), model["kind"], model["switch_cost"], model["min_frames"]),
        file=sys.stderr)


if __name__ == "__main__":
    main()
