"""Tests for M4 on synthetic data: the learned evidence (learned.py) and the segmentation
(segment.py). No DFL download and no competition data needed.

    python -m pytest solution/tests
"""

import itertools
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import atd, control, deadball, learned, pipeline, segment  # noqa: E402
from possession.repo import games, pm  # noqa: E402

PITCH = games.GAMES[games.GAME]["pitch"]


# -- building blocks ---------------------------------------------------------------------------------

def test_since_and_until_count_frames_to_the_nearest_true():
    mask = np.array([0, 1, 0, 0, 1, 0], dtype=bool)
    assert list(learned.since(mask, cap=9)) == [9, 0, 1, 2, 0, 1]
    assert list(learned.until(mask, cap=9)) == [1, 0, 2, 1, 0, 9]


def test_window_mean_uses_the_frames_that_exist():
    values = np.array([1.0, 0.0, 1.0, 1.0])
    assert np.allclose(learned.window_mean(values, -2, 0), [0.0, 1.0, 0.5, 0.5])
    assert np.allclose(learned.window_mean(values, 0, 2), [0.5, 0.5, 1.0, 1.0])


def test_play_stretches_lie_between_stoppages_and_carry_the_restart_team():
    stoppages = [deadball.DeadBall(30, 40, 1, "oob"),
                 deadball.DeadBall(60, 70, control.NOBODY, "slow")]
    assert learned.play_stretches(100, stoppages, [(10, 90)]) == [
        (10, 30, control.NOBODY), (40, 60, 1), (70, 90, control.NOBODY)]


# -- segmentation ------------------------------------------------------------------------------------

def brute_force(log_odds, switch_cost, min_frames, first):
    """The best score segment.viterbi can reach, by trying every team sequence."""
    best = -np.inf
    for teams in itertools.product((0, 1), repeat=len(log_odds)):
        if first is not None and teams[0] != first:
            continue
        lengths = [len(list(run)) for _, run in itertools.groupby(teams)]
        if any(length < min_frames for length in lengths[1:-1]):
            continue
        gain = sum(z / 2 if team == 0 else -z / 2 for z, team in zip(log_odds, teams))
        best = max(best, gain - switch_cost * (len(lengths) - 1))
    return best


def test_viterbi_finds_the_best_path():
    rng = np.random.default_rng(0)
    for trial in range(120):
        log_odds = rng.normal(0, 3, int(rng.integers(1, 9)))
        switch_cost, min_frames = float(rng.uniform(0, 4)), int(rng.integers(1, 4))
        first = (None, 0, 1)[trial % 3]
        teams = segment.viterbi(log_odds, switch_cost, min_frames, first)
        changes = int(np.sum(teams[1:] != teams[:-1]))
        gain = float(np.sum(np.where(teams == 0, log_odds, -log_odds)) / 2)
        assert gain - switch_cost * changes == pytest.approx(
            brute_force(log_odds, switch_cost, min_frames, first))
        if first is not None:
            assert teams[0] == first


def test_a_short_weak_spell_is_absorbed_and_a_strong_one_kept():
    log_odds = np.full(200, 2.0)
    log_odds[100:110] = -1.0                  # 10 frames of weak away evidence: worth 10
    assert (segment.viterbi(log_odds, 20.0) == 0).all()
    log_odds[100:110] = -6.0                  # strong: worth 30 against two changes at 20
    teams = segment.viterbi(log_odds, 20.0)
    assert (teams[100:110] == 1).all() and (teams[:100] == 0).all() and (teams[110:] == 0).all()
    assert (segment.viterbi(log_odds, 20.0, min_frames=25) == 0).all()


def test_the_restart_team_starts_the_stretch_and_can_lose_the_ball_at_once():
    teams = segment.viterbi(np.full(50, 3.0), 5.0, min_frames=25, first=1)
    assert teams[0] == 1 and (teams[1:] == 0).all()


def test_decode_changes_play_only():
    owner = np.zeros(100, dtype=np.int8)
    stoppages = [deadball.DeadBall(40, 60, 0, "oob")]
    log_odds = np.full(100, -4.0)             # away all along
    out = segment.decode(log_odds, owner, stoppages, [(10, 90)], 5.0)
    assert (out[:10] == 0).all() and (out[90:] == 0).all()    # outside the block
    assert (out[40:60] == 0).all()                            # dead: left to the dead mask
    assert (out[10:40] == 1).all()
    assert out[60] == 0 and (out[61:90] == 1).all()           # the home restart, lost at once


# -- the model ---------------------------------------------------------------------------------------

def owner_tree(strength=2.0):
    """A one-split tree on M2c's owner: strength towards the team M2c names."""
    column = learned.names().index("owner_home")
    return {"feature": [column, -1, -1], "threshold": [0.5, 0.0, 0.0], "left": [1, 0, 0],
            "right": [2, 0, 0], "missing_left": [1, 0, 0], "value": [0.0, -strength, strength]}


def test_tree_scores_follow_the_splits():
    features = np.zeros((3, len(learned.names())))
    features[1, learned.names().index("owner_home")] = 1.0
    features[2, learned.names().index("owner_home")] = np.nan
    assert list(learned.tree_scores(features, [owner_tree()])) == [-2.0, 2.0, -2.0]


def test_a_time_threshold_on_a_frame_count_survives_float_steps():
    """A threshold exactly at log(1 + k / FPS) keeps frame k on the left and k + 1 on the right
    however log1p rounds, and a threshold between two counts splits them the same way."""
    column = learned.names().index("since_seen")
    assert learned.time_columns()[column]
    k = 37
    for threshold in (float(learned.seconds(k)),
                      float((learned.seconds(k) + learned.seconds(k + 1)) / 2)):
        tree = {"feature": [column, -1, -1], "threshold": [threshold, 0.0, 0.0],
                "left": [1, 0, 0], "right": [2, 0, 0], "missing_left": [0, 0, 0],
                "value": [0.0, -1.0, 1.0]}
        features = np.zeros((2, len(learned.names())))
        for steps in (-4, -1, 0, 1, 4):
            for row, count in enumerate((k, k + 1)):
                value = float(learned.seconds(count))
                for _ in range(abs(steps)):
                    value = np.nextafter(value, np.inf if steps > 0 else -np.inf)
                features[row, column] = value
            assert list(learned.tree_scores(features, [tree])) == [-1.0, 1.0]
        for nudged in (np.nextafter(threshold, np.inf), np.nextafter(threshold, -np.inf)):
            assert learned.frame_cuts([nudged])[0] == k


def test_both_kinds_of_model_are_symmetric_between_the_teams():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(20, len(learned.names())))
    trees = {"kind": "trees", "trees": [owner_tree(), owner_tree(0.5)]}
    columns = learned.antisymmetric(features).shape[1]
    logistic = {"kind": "logistic", "weights": list(rng.normal(size=columns)),
                "scale": list(rng.uniform(0.5, 2.0, columns))}
    for model in (trees, logistic):
        assert np.allclose(learned.log_odds(learned.swap(features), model),
                           -learned.log_odds(features, model))


def two_team_match(n=1200):
    """Home has the ball, the away team wins it at 400, the ball goes out over the touchline at
    794 and home throws it in at 1001; everyone slows down while it is dead."""
    rng = np.random.default_rng(2)
    home = np.zeros((n, 3, 2), dtype=np.float32)
    away = np.zeros((n, 3, 2), dtype=np.float32)
    for slot in range(3):
        home[:, slot] = [30.0 + 10 * slot, 20.0 + 10 * slot]
        away[:, slot] = [60.0 + 10 * slot, 25.0 + 10 * slot]
    home += rng.normal(0, 0.05, home.shape).astype(np.float32)
    away += rng.normal(0, 0.05, away.shape).astype(np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    ball[50:350] = home[50:350, 0] + 0.5                      # home on the ball
    ball[400:760] = away[400:760, 1] + 0.5                    # away on the ball
    ball[760:800, 0] = away[760, 1, 0]
    ball[760:800, 1] = away[760, 1, 1] + 0.3 * np.arange(40)  # out over the touchline
    home[950:1060, 2] = [55.0, PITCH[1] + 0.2]                # the thrower on the line
    ball[1001:1060, 0] = 55.0 - 0.3 * np.arange(59)
    ball[1001:1060, 1] = PITCH[1] - 1.0
    players = np.concatenate([home, away], axis=1)
    speed = np.full((n, 6), 3.0, dtype=np.float32)
    speed[800:1000] = 0.4
    return atd.Tracking(ball=ball, ball_fields=ball / PITCH, players=players, speed=speed,
                        team_of_slot=np.array([0, 0, 0, 1, 1, 1]), team_names=("H", "A"),
                        pitch=PITCH, home_attacks_right_first_half=True)


def swapped(tracking):
    """The same match with the teams' labels exchanged: the same players, the same ends."""
    return atd.Tracking(ball=tracking.ball, ball_fields=tracking.ball_fields,
                        players=tracking.players, speed=tracking.speed,
                        team_of_slot=1 - tracking.team_of_slot,
                        team_names=tracking.team_names[::-1], pitch=tracking.pitch,
                        home_attacks_right_first_half=not tracking.home_attacks_right_first_half)


def test_the_features_of_a_swapped_match_are_the_swapped_features():
    tracking, blocks = two_team_match(), [(0, 1200)]
    found = pipeline.rules(tracking, blocks)
    assert len(found["stoppages"]) == 1 and len(found["touches"]) >= 2
    features = learned.evidence(found, blocks)
    assert features.shape == (1200, len(learned.names())) and np.isfinite(features).all()
    mirrored = learned.evidence(pipeline.rules(swapped(tracking), blocks), blocks)
    assert np.allclose(mirrored, learned.swap(features))


def test_a_model_that_copies_m2c_gives_its_codes_back():
    tracking, blocks = two_team_match(), [(0, 1200)]
    rules_codes, _ = pipeline.label_frames(tracking, blocks, learned_model=False)
    model = {"kind": "trees", "trees": [owner_tree()], "switch_cost": 1.0, "min_frames": 1}
    codes, info = pipeline.label_frames(tracking, blocks, learned_model=True, model=model)
    assert info["learned"] and (codes == pm.D).sum() > 0
    assert np.array_equal(codes, rules_codes)


def test_the_shipped_model_matches_the_features():
    if not os.path.exists(learned.MODEL_PATH):
        pytest.skip("no model.json shipped")
    model = learned.load()
    assert model["features"] == learned.names()
    assert model["kind"] in ("trees", "logistic") and model["switch_cost"] > 0
    assert "CC BY 4.0" in model["credit"]
    with open(learned.MODEL_PATH) as handle:
        assert json.load(handle) == model


def test_the_shipped_model_runs_by_default():
    if not os.path.exists(learned.MODEL_PATH):
        pytest.skip("no model.json shipped")
    tracking, blocks = two_team_match(), [(0, 1200)]
    codes, info = pipeline.label_frames(tracking, blocks)
    rules_codes, _ = pipeline.label_frames(tracking, blocks, learned_model=False)
    assert info["learned"]
    assert np.array_equal(codes == pm.D, rules_codes == pm.D)    # the stoppages are the rules'
    assert set(np.unique(codes[(codes != pm.D) & (codes != pm.X)])) <= {pm.H, pm.A}
