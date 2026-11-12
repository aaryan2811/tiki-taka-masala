"""Tests for M4's training (devset/train_model.py) on random rows: no DFL download needed.

    python -m pytest solution/devset/tests

The tree export is checked against scikit-learn itself, so these skip where it is not installed;
run.py never needs it.
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from devset import train_model  # noqa: E402  (needs the path above)
from possession import learned  # noqa: E402

SMALL = {"max_iter": 10, "learning_rate": 0.3, "max_leaf_nodes": 7, "min_samples_leaf": 20,
         "l2_regularization": 1.0}


def rows(n=2000, seed=0):
    """Random features where home has the ball when M2c says so, unless the ball has been unseen for
    over 6 s. The time features are whole frame counts through learned.seconds, as on real data."""
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n, len(learned.names())))
    timed = learned.time_columns()
    features[:, timed] = learned.seconds(rng.integers(0, 200, (n, int(timed.sum()))))
    names = learned.names()
    owner = rng.integers(0, 2, n)
    features[:, names.index("owner_home")] = owner == 0
    features[:, names.index("owner_away")] = owner == 1
    noisy = features[:, names.index("since_seen")] > learned.seconds(150)
    home = np.where(noisy, rng.integers(0, 2, n), owner == 0).astype(np.int8)
    return features, home


def test_the_exported_trees_score_as_scikit_learn_does():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier

    features, home = rows()
    model = train_model.fit_trees(features, home, SMALL)
    reference = HistGradientBoostingClassifier(early_stopping=False, random_state=0, **SMALL)
    reference.fit(np.concatenate([features, learned.swap(features)]),
                  np.concatenate([home, 1 - home]))
    test, _ = rows(500, seed=1)
    timed = learned.time_columns()
    assert any(f >= 0 and timed[f] for tree in model["trees"] for f in tree["feature"])
    assert np.allclose(model["base"] + learned.tree_scores(test, model["trees"]),
                       reference.decision_function(test), rtol=0, atol=1e-12)


def test_training_is_deterministic_and_symmetric():
    pytest.importorskip("sklearn")
    features, home = rows()
    first, second = (train_model.fit_trees(features, home, SMALL) for _ in range(2))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    test, _ = rows(500, seed=1)
    for model in (first, train_model.fit_logistic(features, home)):
        log_odds = learned.log_odds(test, model)
        assert np.allclose(learned.log_odds(learned.swap(test), model), -log_odds)
        owner_home = test[:, learned.names().index("owner_home")] > 0
        assert np.mean((log_odds > 0) == owner_home) > 0.7     # it learned to follow M2c's owner
