"""Tests for the DFL dev set, on synthetic data, so they run on a fresh clone with no DFL download.

    python -m pytest solution/devset/tests

The synthetic DFL files follow the real format (dfl.py's docstring): FrameSets per person per half,
a ball FrameSet carrying BallStatus and BallPossession, positions centred on the spot. Two tests use
the real data when it is there and skip otherwise: smooth50 against the published baseline file, and
the shape of a real degraded match.
"""

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from devset import adapter, baselines, degrade, dfl, score, tags, truth  # noqa: E402
from possession import config  # noqa: E402
from possession.repo import pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X
DFL_DIR = os.environ.get("MTAH_DFL_DIR",
                         os.path.join(os.path.expanduser("~"), "Developer", "mtah2027-external",
                                      "dfl"))


# -- synthetic DFL files ---------------------------------------------------------------------------

INFO = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest><MatchInformation>
  <General MatchId="DFL-MAT-TEST01" HomeTeamName="Home FC" HomeTeamId="T-HOME"
           GuestTeamName="Away FC" GuestTeamId="T-AWAY"/>
  <Environment PitchX="105.00" PitchY="68.00"/>
  <Teams>
    <Team TeamId="T-HOME"><Players>
      <Player PersonId="P-H1" PlayingPosition="TW"/><Player PersonId="P-H2" PlayingPosition="IVR"/>
    </Players></Team>
    <Team TeamId="T-AWAY"><Players>
      <Player PersonId="P-A1" PlayingPosition="TW"/><Player PersonId="P-A2" PlayingPosition="STZ"/>
    </Players></Team>
  </Teams>
</MatchInformation></PutDataRequest>
"""

EVENTS = """<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
<Event EventTime="2023-05-27T15:29:59.900+02:00"><KickOff GameSection="firstHalf">
  <Play Team="T-HOME"/></KickOff></Event>
<Event EventTime="2023-05-27T15:30:01.200+02:00"><ThrowIn Team="T-AWAY"/></Event>
<Event EventTime="2023-05-27T15:30:02.000+02:00"><Play Team="T-HOME"/></Event>
<Event EventTime="2023-05-27T16:30:00.400+02:00"><CornerKick Team="T-HOME"/></Event>
</PutDataRequest>
"""

N_FRAMES = 60  # per half


def _frames(n0, t0_seconds, rows, ball=False):
    out = []
    for i, row in enumerate(rows):
        stamp = "2023-05-27T{}:{:06.3f}+00:00".format(t0_seconds[0], t0_seconds[1] + i * 0.04)
        if ball:
            x, y, status, possession = row
            out.append('<Frame N="{}" T="{}" X="{:.2f}" Y="{:.2f}" Z="0.00" S="0.0" '
                       'BallStatus="{}" BallPossession="{}"/>'.format(n0 + i, stamp, x, y, status,
                                                                       possession))
        elif row is not None:
            out.append('<Frame N="{}" T="{}" X="{:.2f}" Y="{:.2f}" S="1.0"/>'.format(
                n0 + i, stamp, row[0], row[1]))
    return "\n".join(out)


def positions_xml():
    """Two halves of 60 frames; dead ball on frames 20-39 of the first half; the away striker comes
    on at frame 10 of the second half; a referee who must be skipped."""
    parts = ['<?xml version="1.0" encoding="UTF-8"?>\n<PutDataRequest><Positions>']
    for section, n0, clock in (("firstHalf", 10000, ("13:30", 0.0)),
                               ("secondHalf", 100000, ("14:30", 0.0))):
        ball = [(0.1 * i, 0.0, 0 if section == "firstHalf" and 20 <= i < 40 else 1,
                 1 if i < 30 else 2) for i in range(N_FRAMES)]
        people = {("T-HOME", "P-H1"): [(-50.0, 0.0)] * N_FRAMES,
                  ("T-HOME", "P-H2"): [(-10.0, 1.0)] * N_FRAMES,
                  ("T-AWAY", "P-A1"): [(50.0, 0.0)] * N_FRAMES,
                  ("T-AWAY", "P-A2"): [None] * 10 + [(10.0, -1.0)] * (N_FRAMES - 10)
                  if section == "secondHalf" else [(10.0, -1.0)] * N_FRAMES,
                  ("referee", "R1"): [(0.0, 5.0)] * N_FRAMES}
        for (team, person), rows in people.items():
            parts.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">'.format(
                section, team, person))
            parts.append(_frames(n0, clock, rows))
            parts.append("</FrameSet>")
        parts.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="B">'.format(section))
        parts.append(_frames(n0, clock, ball, ball=True))
        parts.append("</FrameSet>")
    parts.append("</Positions></PutDataRequest>\n")
    return "\n".join(parts)


@pytest.fixture
def dfl_folder(tmp_path, monkeypatch):
    """A DFL folder with one synthetic match, and the cache redirected into tmp_path."""
    folder = tmp_path / "dfl" / "TEST01"
    folder.mkdir(parents=True)
    stem = "DFL-COM-000001_DFL-MAT-TEST01.xml"
    (folder / ("DFL_02_01_matchinformation_" + stem)).write_text(INFO)
    (folder / ("DFL_03_02_events_raw_" + stem)).write_text(EVENTS)
    (folder / ("DFL_04_03_positions_raw_observed_" + stem)).write_text(positions_xml())
    monkeypatch.setattr(dfl, "CACHE_DIR", str(tmp_path / "cache"))
    return str(tmp_path / "dfl")


# -- dfl -------------------------------------------------------------------------------------------

def test_load_parses_halves_persons_and_flags(dfl_folder):
    match = dfl.load(dfl_folder, "TEST01", verbose=False)

    assert dfl.available_matches(dfl_folder) == ["TEST01"]
    assert match.team_names == ("Home FC", "Away FC") and match.pitch == (105.0, 68.0)
    first, second = match.halves
    assert first.n0 == 10000 and second.n0 == 100000 and first.n_frames == N_FRAMES
    assert first.players.shape == (N_FRAMES, 4, 2)  # the referee is skipped
    assert first.team.tolist() == [0, 0, 1, 1] and first.keeper.tolist() == [True, False] * 2
    assert np.all(first.status[20:40] == 0) and np.all(first.status[:20] == 1)
    assert first.possession[29] == 1 and first.possession[30] == 2
    assert np.isnan(second.players[9, 3, 0]) and second.players[10, 3, 0] == 10.0
    assert second.speed[10, 3] == pytest.approx(1.0 / 3.6) and np.isnan(second.speed[9, 3])


def test_set_pieces_land_on_their_half_and_frame(dfl_folder):
    match = dfl.load(dfl_folder, "TEST01", verbose=False)

    # the kick-off is stamped a little before the first frame and still counts, at row 0; a Play
    # is not a set piece; the corner falls in the second half at 0.4 s = row 10
    assert match.set_pieces == [(0, 0, "Kick Off", 0), (0, 30, "Throw-in", 1),
                                (1, 10, "Corner Kick", 0)]


def test_cache_round_trip(dfl_folder):
    parsed = dfl.load(dfl_folder, "TEST01", verbose=False)
    cached = dfl.load(dfl_folder, "TEST01", verbose=False)

    assert os.path.isfile(os.path.join(dfl.CACHE_DIR, "TEST01.npz"))
    assert cached.set_pieces == parsed.set_pieces and cached.team_ids == parsed.team_ids
    for one, other in zip(parsed.halves, cached.halves):
        np.testing.assert_array_equal(one.players, other.players)
        np.testing.assert_array_equal(one.possession, other.possession)
        np.testing.assert_array_equal(one.speed, other.speed)
        assert one.t0 == other.t0


# -- truth -----------------------------------------------------------------------------------------

def test_half_codes_follow_the_flags(dfl_folder):
    half = dfl.load(dfl_folder, "TEST01", verbose=False).halves[0]
    codes = truth.half_codes(half)

    assert np.all(codes[:20] == H) and np.all(codes[20:40] == D) and np.all(codes[40:] == A)


def test_merge_absorbs_short_sandwiched_spells_only():
    codes = np.array([H] * 100 + [A] * 30 + [H] * 100 + [A] * 30 + [D] * 50 + [A] * 100,
                     dtype=np.int8)
    merged = truth.merge_lost_back(codes, 2.0)

    assert np.all(merged[:230] == H)                 # a 1.2 s Away spell between two Home ones
    assert np.all(merged[230:260] == A)              # next to a dead ball: kept
    np.testing.assert_array_equal(truth.merge_lost_back(codes, 0.0), codes)
    np.testing.assert_array_equal(truth.merge_lost_back(codes, 1.0), codes)  # 30 frames > 25


def test_merge_cascades_like_the_audit_rule():
    # H(long) A(10) H(10) A(10) H(long): the first A joins its neighbours, after which the next A
    # is sandwiched by Home and goes too.
    codes = np.array([H] * 100 + [A] * 10 + [H] * 10 + [A] * 10 + [H] * 100, dtype=np.int8)

    assert np.all(truth.merge_lost_back(codes, 2.0) == H)


def test_match_codes_lay_out_halves_with_x_between(dfl_folder):
    match = dfl.load(dfl_folder, "TEST01", verbose=False)
    codes, blocks = truth.match_codes(match)

    assert blocks == [(0, N_FRAMES), (N_FRAMES + degrade.HALF_GAP, 2 * N_FRAMES + degrade.HALF_GAP)]
    assert np.all(codes[N_FRAMES:N_FRAMES + degrade.HALF_GAP] == X)
    assert pm.valid_runs(codes) == blocks


# -- degrade ---------------------------------------------------------------------------------------

def test_dead_ball_track_replaces_the_frozen_ball():
    n, a, b = 400, 50, 250                            # long enough for the full REST
    ball = np.zeros((n, 2), dtype=np.float32)
    ball[:, 0] = np.arange(n) * 0.4                   # 10 m/s along x
    status = np.ones(n, dtype=np.int8)
    status[a:b] = 0
    ball[a:b] = ball[a - 1]                           # DFL: frozen during the dead ball...
    ball[b:, 0] = 5.0 + np.arange(n - b) * 0.4        # ...then a jump to the restart spot
    ball[b:, 1] = 30.0
    out = degrade.dead_ball_track(ball, status)

    steps = np.hypot(*np.diff(out, axis=0).T)
    assert np.all(steps[a - 1:b - 1 - degrade.REST] > 0)  # nothing frozen until it is placed
    np.testing.assert_allclose(out[b - degrade.REST:b], [[4.6, 30.0]] * degrade.REST,
                               atol=1e-4)             # resting on the spot before the kick
    assert np.all(steps[:a - 1] > 0.39) and out[a, 0] > ball[a - 1, 0]  # rolls on
    assert steps[b - 1] < 1.0                         # no jump at the restart
    np.testing.assert_array_equal(out[b:], ball[b:])


def test_phases_mark_the_edges_of_each_state():
    dead = np.array([False] * 200 + [True] * 200 + [False] * 100)
    phase = degrade.phases(dead, edge=75)
    name = np.array(degrade.PHASES)[phase]

    assert set(name[:75]) == {"alive_early"} and set(name[75:125]) == {"alive_late"}
    assert set(name[125:200]) == {"alive_pre"} and set(name[200:275]) == {"dead_early"}
    assert set(name[325:400]) == {"dead_pre"} and set(name[400:475]) == {"alive_early"}
    assert set(name[475:]) == {"alive_late"}          # the last state of a half has no "pre"


def test_dropout_follows_each_phase_rates():
    rates = {name: [0.5, 0.5] for name in degrade.PHASES}
    rates["alive_late"] = [0.0, 1.0]                   # always seen once found
    rates["dead_late"] = [1.0, 0.0]                    # never seen once lost
    params = {"rates": rates, "seen_share": {"dead": 0.5, "alive": 0.5}}
    dead = np.array([False] * 1000 + [True] * 1000)
    seen = degrade.dropout_mask(dead, params, np.random.default_rng(0))

    assert seen[100:900].all() and not seen[1100:1900].any()


def test_deal_slots_never_doubles_a_slot():
    pieces = [(0, 50, 0), (10, 60, 1), (50, 80, 2), (60, 90, 0)]
    dealt = degrade.deal_slots(pieces, 2)

    for slot in (0, 1):
        spans = sorted((a, b) for s, a, b, _ in dealt if s == slot)
        assert all(prev[1] <= nxt[0] for prev, nxt in zip(spans, spans[1:]))
    with pytest.raises(ValueError):
        degrade.deal_slots([(0, 10, 0), (5, 15, 1), (6, 16, 2)], 2)


def test_smooth_keeps_nan_rows_and_slows_a_kick():
    track = np.zeros((100, 2), dtype=np.float32)
    track[50:, 0] = np.arange(50) * 0.8
    track[10:15] = np.nan
    out = degrade.smooth(track, 4.0)

    assert np.isnan(out[10:15]).all() and not np.isnan(out[15:]).any()
    assert np.max(np.abs(np.diff(np.diff(out[40:70, 0])))) < 0.8 / 4  # the ramp is spread out


def test_quantile_sampling_stays_inside_the_table():
    draws = degrade.sample(list(np.linspace(10, 30, len(degrade.PROBS))),
                           np.random.default_rng(0), 1000)

    assert draws.min() >= 10 and draws.max() <= 30 and abs(np.median(draws) - 20) < 1


def test_committed_params_have_every_section():
    params = degrade.load_params()

    for section in ("ball", "players", "slots", "keepers", "tags"):
        assert section in params
    assert set(params["ball"]["rates"]) == set(degrade.PHASES)
    for table in (params["slots"]["lifetime_frames"], params["slots"]["reacquire_frames"],
                  params["keepers"]["swap_frames"], params["tags"]["start_offset_frames"]):
        assert len(table) == len(degrade.PROBS) and list(table) == sorted(table)
    assert "mask" not in json.dumps(params).lower()  # aggregates only, never a frame-level mask


def test_to_tracking_shapes_and_pipeline_runs(dfl_folder):
    match = dfl.load(dfl_folder, "TEST01", verbose=False)
    params = degrade.load_params()
    tracking, blocks = degrade.to_tracking(match, params, seed=1)

    n_frames = 2 * N_FRAMES + degrade.HALF_GAP
    assert tracking.n_frames == n_frames and tracking.players.shape == (n_frames, 46, 2)
    assert tracking.keeper_slots == (22, 45)
    assert tracking.team_of_slot.tolist() == [0] * 23 + [1] * 23
    assert tracking.home_attacks_right_first_half      # home kicks off from x < 0
    assert np.isnan(tracking.players[N_FRAMES:N_FRAMES + degrade.HALF_GAP]).all()
    seen = tracking.ball[~np.isnan(tracking.ball[:, 0])]
    assert np.all((seen[:, 0] > 0) & (seen[:, 0] < 105))  # pitch-corner origin
    np.testing.assert_array_equal(degrade.to_tracking(match, params, seed=1)[0].players,
                                  tracking.players)    # reproducible

    codes, _ = adapter.label(tracking, blocks, tags.synthetic(match, blocks, params, seed=1))
    assert len(codes) == n_frames and np.all(codes[N_FRAMES:N_FRAMES + degrade.HALF_GAP] == X)


# -- adapter, tags, baselines, score ---------------------------------------------------------------

def test_known_dead_spans_are_emptied_and_restored():
    saved = config.KNOWN_DEAD_SPANS
    with adapter.no_known_dead_spans():
        assert config.KNOWN_DEAD_SPANS == ()
    assert config.KNOWN_DEAD_SPANS == saved
    with pytest.raises(RuntimeError):
        with adapter.no_known_dead_spans():
            raise RuntimeError
    assert config.KNOWN_DEAD_SPANS == saved


def test_synthetic_tags_follow_the_competition_counts(dfl_folder):
    match = dfl.load(dfl_folder, "TEST01", verbose=False)
    match.set_pieces = [(0, 30, "Throw-in", 1)] * 40 + [(0, 50, "Free Kick", 0)] * 20
    params = degrade.load_params()
    _, blocks = degrade.layout(match)
    made = tags.synthetic(match, blocks, params, seed=0)

    kinds = [tag["tags"]["Type"] for tag in made]
    per_type = params["tags"]["per_type"]
    assert kinds.count("Throw-in") == per_type["Throw-in"]["count"]
    assert kinds.count("Free Kick") == per_type["Free Kick"]["count"]
    assert all(tag["tags"]["Team"] == "N/A" for tag in made if tag["tags"]["Type"] == "Throw-in")
    assert all(tag["code"] == "SET PIECES" and tag["end_frame"] - tag["start_frame"] == 875
               for tag in made)


def test_restart_is_the_nearest_dead_ball_end(dfl_folder):
    half = dfl.load(dfl_folder, "TEST01", verbose=False).halves[0]

    assert tags.restart_of(half, 30) == 40             # the dead ball on 20-39 ends at 40
    assert tags.restart_of(half.__class__(
        "x", 0, 0, half.ball, np.ones_like(half.status), half.possession, half.players,
        half.team, half.keeper, half.person_ids), 30) == 30


def test_smooth50_is_nearest_player_then_the_starter_vote():
    n = 300
    tracking = degrade.atd.Tracking(
        ball=np.column_stack([np.linspace(0, 10, n), np.zeros(n)]).astype(np.float32),
        ball_fields=np.zeros((n, 3)), players=np.zeros((n, 2, 2), dtype=np.float32),
        speed=np.zeros((n, 2)), team_of_slot=np.array([0, 1], dtype=np.int8),
        team_names=("H", "A"), pitch=(105, 68))
    tracking.players[:, 1, 0] = 10.0                   # home at x 0, away at x 10
    tracking.ball[200:] = np.nan
    codes = baselines.smooth50(tracking, [(0, 250)])

    assert np.all(codes[:100] == H) and np.all(codes[160:250] == A) and np.all(codes[250:] == X)


def competition_tracking():
    """The real match's tracking, or a skip when the gated data is not here."""
    from possession import atd
    try:
        return atd.load(verbose=False)
    except IOError:
        pytest.skip("needs the competition data (data/ or MTAH_DATA_DIR)")


def test_smooth50_reproduces_the_published_file():
    from possession import labels, repo
    tracking = competition_tracking()
    blocks = labels.graded_blocks()
    codes = baselines.smooth50(tracking, blocks)
    published = pm.label_codes(pm.read_label_csv(
        os.path.join(repo.RESULTS_DIR, "baseline_smooth50.csv")))
    graded = codes != X

    np.testing.assert_array_equal(codes[graded], published[graded])


def test_scores_are_the_official_ones():
    gt = np.array([X] * 10 + [H] * 100 + [A] * 50 + [D] * 40 + [H] * 80 + [X] * 10, dtype=np.int8)
    pred = gt.copy()
    pred[120:170] = A                                  # a change 10 frames early
    one = score.Scored(gt, pred)
    official = pm.evaluate(gt, pred)

    assert score.summary([one])["spell_f1"] == pytest.approx(official["spell_f1_mean"])
    assert score.summary([one])["macro_f1"] == pytest.approx(official["macro_f1"])
    assert score.summary([score.Scored(gt, gt)])["spell_f1"] == 1.0
    both = pm.evaluate(np.concatenate([gt, gt]), np.concatenate([pred, gt]))
    assert score.summary([one, score.Scored(gt, gt)])["spell_f1"] == pytest.approx(
        both["spell_f1_mean"])                         # pooled = matches laid end to end
    point, low, high, above = score.paired([[one]], [[one]], n_draws=50)
    assert point == 0 and low == 0 and high == 0 and above == 0


# -- groups and the foul-stoppage check ------------------------------------------------------------

def test_groups_keep_calibration_apart_from_held_out():
    from devset import run_devset

    assert run_devset.groups(["J03WMX", "J03WN1", "J03WPY"]) == [
        ("Calibration matches", [0, 2]), ("Held-out matches", [1]), ("All matches", [0, 1, 2])]
    assert run_devset.groups(["J03WMX", "J03WPY"]) == [("Calibration matches", [0, 1])]
    scored = {("p", "tags", "raw"): [["a", "b", "c"], ["d", "e", "f"]]}
    assert run_devset.subset(scored, [0, 2]) == {("p", "tags", "raw"): [["a", "c"], ["d", "f"]]}


def test_foul_check_classes_candidates_by_the_truth():
    from devset import foul_check

    n = 2000
    collective = np.full(n, 1.2)
    collective[100:200] = 0.5                          # deep: passes
    dead = np.zeros(n, dtype=bool)
    dead[100:200] = True                               # ...and dead: caught
    dead[1000:1500] = True                             # long and dead, but crossed
    slow = [(100, 200), (400, 500), (1000, 1500), (1600, 1990)]
    found = foul_check.judge("M", collective, dead, np.array([990]), slow)

    assert [c.kind for c in found] == ["dead", "play", "dead", "play"]
    assert [c.passes for c in found] == [True, False, True, True]  # the last is 15.6 s long
    assert [c.crossed for c in found] == [False, False, True, False]
    lines = foul_check.report("x", found)
    assert "wholly in play 1 of 2 (false stoppages)" in lines[1]
    assert "mostly dead 1 of 1 (caught)" in lines[1]
