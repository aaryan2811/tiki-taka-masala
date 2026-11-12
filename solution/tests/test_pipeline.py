"""Tests on synthetic tracking, so the pipeline is checked before the real data exists.

    python -m pytest solution/tests

The synthetic files follow the raw format starter/baseline.py reads -- "<frame_id>:<slots>:<ball>",
"x,y,speed" groups, NaN for missing -- including a missing group written as a bare "NaN", which the
real file may or may not contain. Once the data arrives, stage 0 confirms which.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import (atd, ball, boundary, config, contest, control, deadball,  # noqa: E402
                        keepers, labels, pipeline)  # (needs the path above)
from possession.repo import games, pm  # noqa: E402

PITCH = games.GAMES[games.GAME]["pitch"]
N_VIDEO = games.GAMES[games.GAME]["n_video_frames"]

META = """<main><Metadata>
  <Teams>
    <Team id="T1"><Name>Home FC</Name></Team>
    <Team id="T2"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T1"/><Player id="P2" teamId="T1"/>
    <Player id="P3" teamId="T2"/><Player id="P4" teamId="T2"/>
  </Players>
</Metadata></main>
"""

KEEPER_META = """<main><Metadata>
  <Teams>
    <Team id="T1"><Name>Home FC</Name><ProviderTeamsParameters><ProviderParameter>
      <Name>attack_direction_first_half</Name><Value>right_to_left</Value>
    </ProviderParameter></ProviderTeamsParameters></Team>
    <Team id="T2"><Name>Away FC</Name></Team>
  </Teams>
  <Players>
    <Player id="P1" teamId="T1"><ProviderPlayerParameters><ProviderParameter>
      <Name>position_type</Name><Value>Unknown</Value>
    </ProviderParameter></ProviderPlayerParameters></Player>
    <Player id="P2" teamId="T1"><ProviderPlayerParameters><ProviderParameter>
      <Name>position_type</Name><Value>Goalkeeper</Value>
    </ProviderParameter></ProviderPlayerParameters></Player>
    <Player id="P3" teamId="T2"/>
  </Players>
</Metadata></main>
"""

RAW = [
    "1:0.10,0.20,1.5;0.30,0.40,2.0;0.50,0.60,0.5;0.70,0.80,0.0:0.11,0.21,3.0",
    "2:NaN;0.30,0.40,2.0;0.50,0.60,0.5;0.70,0.80,0.0:NaN",
    "3:0.10,0.20,1.5;0.30,0.40,2.0;NaN,NaN,NaN;0.70,0.80,0.0:NaN,NaN,NaN",
]


def write_bundle(root):
    """A tactical folder with the synthetic metadata and raw tracking, named as games.py expects."""
    paths = games.asset_paths(data_dir=str(root))
    os.makedirs(os.path.dirname(paths["atd_raw"]))
    with open(paths["atd_meta"], "w") as handle:
        handle.write(META)
    with open(paths["atd_raw"], "w") as handle:
        handle.write("\n".join(RAW) + "\n")
    return paths


# -- stage 0 -----------------------------------------------------------------------------------------

def test_parse_raw_handles_both_kinds_of_missing_group(tmp_path):
    paths = write_bundle(tmp_path)
    ball_fields, slot_fields = atd.parse_raw(paths["atd_raw"], 5, 4)

    assert ball_fields.shape == (5, 3) and slot_fields.shape == (5, 4, 3)
    assert np.allclose(ball_fields[0], [0.11, 0.21, 3.0])
    assert np.isnan(ball_fields[1]).all() and np.isnan(ball_fields[2]).all()
    assert np.isnan(slot_fields[1, 0]).all()        # bare "NaN" group
    assert np.isnan(slot_fields[2, 2]).all()        # "NaN,NaN,NaN" group
    assert np.allclose(slot_fields[2, 3], [0.70, 0.80, 0.0])
    assert np.isnan(slot_fields[3:]).all()          # frames absent from the file


def test_group_widths_when_no_line_is_complete(tmp_path):
    # 46 slots for 22 players: in the real file no line has every slot present.
    path = tmp_path / "sparse.txt"
    path.write_text("1:NaN;0.5,0.5,1.0,7:NaN\n2:NaN;NaN:0.1,0.2,0.3\n")
    assert atd.group_widths(str(path)) == (3, 4)


def test_load_scales_to_metres_and_caches(tmp_path, monkeypatch):
    write_bundle(tmp_path / "data")
    monkeypatch.setattr(atd, "CACHE_DIR", str(tmp_path / "cache"))

    first = atd.load(data_dir=str(tmp_path / "data"), verbose=False)
    assert first.n_frames == N_VIDEO
    assert list(first.team_of_slot) == [0, 0, 1, 1]
    assert first.team_names == ("Home FC", "Away FC")
    assert np.allclose(first.ball[0], [0.11 * PITCH[0], 0.21 * PITCH[1]])
    assert np.allclose(first.speed[0], [1.5, 2.0, 0.5, 0.0])

    assert os.path.isfile(atd._cache_path(games.GAME, "tactical"))
    second = atd.load(data_dir=str(tmp_path / "data"), verbose=False)
    assert np.array_equal(first.players, second.players, equal_nan=True)


def test_keeper_slots_and_attack_direction_come_from_the_metadata(tmp_path):
    path = tmp_path / "meta.xml"
    path.write_text(KEEPER_META)
    assert atd.keeper_slots(str(path)) == (1,)
    assert atd.home_attacks_right_first_half(str(path)) is False

    path.write_text(META)                   # no provider parameters at all
    assert atd.keeper_slots(str(path)) == ()


# -- stage 1 -----------------------------------------------------------------------------------------

def test_clean_removes_spikes_and_short_runs():
    track = np.tile(np.arange(30, dtype=np.float32)[:, None] * 0.5, (1, 2))  # 12.5 m/s, plausible
    track[5] = [80.0, 80.0]                 # there-and-back jump: a spike
    track[12:15] = np.nan
    track[16:18] = np.nan                   # leaves frame 15 visible alone: too short a run
    cleaned = ball.clean(track)

    assert np.isnan(cleaned[5]).all()
    assert np.isnan(cleaned[15]).all()
    assert not np.isnan(cleaned[[4, 6, 11, 18, 29]]).any()


def test_interpolate_fills_only_short_interior_gaps():
    track = np.tile(np.arange(100, dtype=np.float32)[:, None], (1, 2))
    track[:3] = np.nan                      # head of the array: nothing to interpolate from
    track[10:20] = np.nan                   # short interior gap
    track[40:80] = np.nan                   # longer than the default 25 frames
    filled, interpolated = ball.interpolate(track)

    assert np.isnan(filled[:3]).all()
    assert np.allclose(filled[10:20, 0], np.arange(10, 20))
    assert np.isnan(filled[40:80]).all()
    assert interpolated.sum() == 10


def keeper_match(n=200, keeper_x=95.0):
    """Home attacks right in the first half, so the away goal is at high x. Slot 0 is home, slot 1
    away, slot 2 the XML's home keeper -- standing at keeper_x."""
    players = np.full((n, 3, 2), 30.0, dtype=np.float32)
    players[:, 1] = [70.0, 30.0]
    players[:, 2] = [keeper_x, 34.0]
    return atd.Tracking(ball=np.full((n, 2), np.nan, dtype=np.float32),
                        ball_fields=np.full((n, 3), np.nan, dtype=np.float32), players=players,
                        speed=np.ones((n, 3), dtype=np.float32), team_of_slot=np.array([0, 1, 0]),
                        team_names=("H", "A"), pitch=PITCH, keeper_slots=(2,),
                        home_attacks_right_first_half=True)


def test_a_keeper_slot_belongs_to_the_team_whose_half_it_is_in():
    tracking = keeper_match()
    tracking.players[150:, 2] = [10.0, 34.0]     # in the low-x goal for the second half
    split = keepers.split(tracking, [(0, 100), (100, 200)])

    assert list(split.team_of_slot) == [0, 1, 0, 1] and split.keeper_slots == ()
    home_keeper, away_keeper = split.players[:, 2], split.players[:, 3]
    assert np.isnan(home_keeper[:100]).all() and np.allclose(away_keeper[:100], [95.0, 34.0])
    assert np.allclose(home_keeper[100:149], [95.0, 34.0])     # ends switched at half time
    assert np.allclose(away_keeper[151:], [10.0, 34.0])
    assert np.isnan(split.players[149:151, 2:]).all()          # both ends of the jump are dropped
    assert keepers.split(split, [(0, 100), (100, 200)]) is split


def test_a_keeper_in_the_wrong_slot_does_not_take_the_ball():
    tracking = keeper_match()
    tracking.players[:, 1] = [80.0, 34.0]
    tracking.ball[10:60] = [80.5, 34.0]                                  # at an away player's feet
    tracking.ball[60:120] = np.linspace([80.5, 34.0], [94.5, 34.0], 60)  # passed back, 6 m/s
    tracking.ball[120:180] = [94.5, 34.0]                                # held by the keeper

    codes, _ = pipeline.label_frames(tracking, [(0, 200)], split_keepers=False,
                                     learned_model=False)
    assert (codes[130:] == pm.H).all()       # read as the XML's home keeper
    codes, _ = pipeline.label_frames(tracking, [(0, 200)], learned_model=False)
    assert (codes == pm.A).all()             # it is away's own keeper


# -- stage 2 -----------------------------------------------------------------------------------------

def test_possession_changes_only_on_controlled_touches():
    n = 100
    on_ball = np.full(n, control.NOBODY, dtype=np.int8)
    on_ball[20:30] = 0                      # home touch
    on_ball[40:43] = 1                      # 3-frame away contact: a deflection
    on_ball[60:70] = 1                      # away touch
    owner = control.possession(control.touches(on_ball), n, [(10, 90)])

    assert (owner[:10] == control.NOBODY).all() and (owner[90:] == control.NOBODY).all()
    assert (owner[10:60] == 0).all()        # back-filled to the block head, held through 40-43
    assert (owner[60:90] == 1).all()        # changes on the first frame of the touch


def test_touch_running_into_a_block_counts_from_its_first_frame():
    n = 100
    on_ball = np.full(n, control.NOBODY, dtype=np.int8)
    on_ball[5:20] = 0                       # home on the ball across the block start
    on_ball[50:60] = 1
    owner = control.possession(control.touches(on_ball), n, [(10, 90)])

    assert (owner[10:50] == 0).all() and (owner[50:90] == 1).all()


def duel(n=150, opponent=1.0, back_after=0):
    """(touches, distance): away has the ball on [0, 50), home touches it on [50, 68) with away's
    nearest player `opponent` metres from the ball throughout, and away is back after back_after
    frames."""
    distance = np.full((n, 2), 6.0, dtype=np.float32)
    distance[50:68, 1] = opponent
    return [(0, 50, 1), (50, 68, 0), (68 + back_after, 120, 1)], distance


def test_a_duel_belongs_to_the_team_that_comes_out_with_the_ball():
    touch_list, distance = duel()
    kept, void = contest.resolve(touch_list, distance)
    assert void == [(50, 68, 0)]
    owner = control.possession(kept, 150, [(0, 150)])
    assert (owner == 1).all()


def test_a_touch_after_which_the_ball_travels_still_counts():
    touch_list, distance = duel(back_after=20)
    assert contest.resolve(touch_list, distance) == (touch_list, [])


def test_a_touch_clear_of_the_other_team_still_counts():
    touch_list, distance = duel(opponent=1.0)
    distance[60, 1] = 4.0                   # away's player not at the ball on one frame
    assert contest.resolve(touch_list, distance) == (touch_list, [])


def test_label_frames_end_to_end_on_synthetic_tracking():
    n = 400
    home = np.tile(np.linspace([10.0, 30.0], [60.0, 30.0], n), (1, 1)).astype(np.float32)
    away = np.tile(np.linspace([70.0, 40.0], [20.0, 40.0], n), (1, 1)).astype(np.float32)
    players = np.stack([home, away], axis=1)                   # (n, 2 slots, 2)
    track = np.full((n, 2), np.nan, dtype=np.float32)
    track[0:150] = home[0:150] + 0.5                           # home dribbles
    track[200:300] = away[200:300] + 0.5                       # away has it after a long gap
    track[250:253] = home[250:253] + 0.5                       # home deflection, 3 frames
    tracking = atd.Tracking(ball=track, ball_fields=track / PITCH, players=players,
                            speed=np.ones((n, 2), dtype=np.float32),
                            team_of_slot=np.array([0, 1]), team_names=("H", "A"), pitch=PITCH)

    codes, info = pipeline.label_frames(tracking, [(50, 350)], learned_model=False)

    assert info["touches"] == 3            # home, away, away again after the deflection
    assert (codes[:50] == pm.X).all() and (codes[350:] == pm.X).all()
    assert (codes[50:200] == pm.H).all()   # held through the 50-frame gap
    assert (codes[200:350] == pm.A).all()  # the deflection does not change possession


# -- stage 3 -----------------------------------------------------------------------------------------

def synthetic_stoppage_match(n=2000):
    """Home dribbles, the ball goes out over the touchline at 794, everyone slows on [800, 1200),
    away restarts with a kick at 1181. A second slowdown on [1500, 1700) has no evidence at all."""
    home = np.zeros((n, 2), dtype=np.float32)
    home[:, 0] = 20.0 + 0.08 * np.arange(n)                  # 2 m/s, at x 80.8 by frame 760
    home[:, 1] = 60.0
    away = np.tile(np.array([30.0, 20.0], dtype=np.float32), (n, 1))
    away[1100:1250] = [50.5, 67.5]

    track = np.full((n, 2), np.nan, dtype=np.float32)
    track[:760] = home[:760]
    track[760:800, 0] = home[760, 0]
    track[760:800, 1] = 60.0 + 0.24 * np.arange(40)          # 6 m/s towards the touchline
    track[1100:1181] = [50.0, 67.5]                          # placed for the throw-in
    track[1181:1250, 0] = 50.0 + 0.32 * np.arange(1, 70)     # thrown in at 8 m/s
    track[1181:1250, 1] = 67.5

    speed = np.full((n, 2), 3.0, dtype=np.float32)
    speed[800:1200] = 0.5
    speed[1500:1700] = 0.5
    return atd.Tracking(ball=track, ball_fields=track / PITCH,
                        players=np.stack([home, away], axis=1), speed=speed,
                        team_of_slot=np.array([0, 1]), team_names=("Home FC", "Away FC"),
                        pitch=PITCH)


def test_line_crossing_is_walked_back_to_the_line():
    tracking = synthetic_stoppage_match()
    assert list(deadball.line_crossings(tracking.ball, PITCH)) == [794]


def test_a_kick_needs_the_ball_rested_first():
    tracking = synthetic_stoppage_match()
    kicks = deadball.kicks(tracking.ball)
    assert 1181 in kicks
    assert not [k for k in kicks if 762 <= k < 1181]        # nothing while the ball is dead


def test_stoppages_need_evidence_and_end_at_the_restart_kick():
    # M2's rule: a crossing or a tag. Both slowdowns here are deeper than open play, which M2b also
    # takes as evidence, so it is switched off to check the other two.
    tracking = synthetic_stoppage_match()
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=(), foul_stoppages=False)
    assert [(s.start, s.stop, s.team, s.evidence) for s in found] == [(794, 1181, 1, "oob")]

    tag = {"code": "SET PIECES", "start_frame": 1550, "end_frame": 1600,
           "tags": {"Team": "Home FC", "Type": "Free Kick"}}
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=[tag], foul_stoppages=False)
    assert len(found) == 2
    second = found[1]
    assert second.evidence == "tag" and second.team == 0  # ball unseen: the tag names the team
    assert 1490 <= second.start <= 1510 and 1690 <= second.stop <= 1710


def test_a_slowdown_beyond_open_play_needs_no_crossing_or_tag():
    tracking = synthetic_stoppage_match()      # everyone at 0.5 m/s on [1500, 1700): no play does that
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=())
    assert [(s.start, s.stop, s.team, s.evidence) for s in found[:1]] == [(794, 1181, 1, "oob+slow")]
    assert len(found) == 2
    second = found[1]
    assert second.evidence == "slow" and second.team == deadball.NOBODY  # nothing names a team
    assert 1490 <= second.start <= 1510 and 1690 <= second.stop <= 1710

    tracking.speed[1500:1700] = 1.0            # a lull: slow, but no slower than open play gets
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=())
    assert [s.evidence for s in found] == ["oob+slow"]


def test_beyond_open_play_is_deeper_or_longer():
    collective = np.full(1000, 1.2, dtype=np.float32)
    assert not deadball.beyond_open_play(collective, 100, 225)       # 5 s of a lull
    collective[150] = 0.7
    assert deadball.beyond_open_play(collective, 100, 225)           # everyone walking at once
    assert deadball.beyond_open_play(collective, 200, 200 + config.LONG_SLOW_FRAMES)
    assert not deadball.beyond_open_play(collective, 200, 199 + config.LONG_SLOW_FRAMES)


def test_a_stoppage_with_no_restart_kick_joins_the_next():
    first = deadball.DeadBall(100, 200, 0, "slow")       # ends where the players sped up
    second = deadball.DeadBall(250, 400, 1, "tag")       # the kick-off, 2 s later
    joined = deadball.join([first, second], kick_frames=[400])
    assert [(s.start, s.stop, s.team, s.evidence) for s in joined] == [(100, 400, 1, "slow+tag")]

    assert len(deadball.join([first, second], kick_frames=[220, 400])) == 2  # played in between
    assert len(deadball.join([first, second], kick_frames=[200, 400])) == 2  # restarted at 200
    later = deadball.DeadBall(200 + config.JOIN_GAP, 400, 1, "tag")
    assert len(deadball.join([first, later], kick_frames=[400])) == 2       # too long a gap


def test_touches_during_a_stoppage_do_not_count_and_restart_team_takes_over():
    n = 100
    on_ball = np.full(n, control.NOBODY, dtype=np.int8)
    on_ball[10:20] = 1                      # away has it before the stoppage
    on_ball[45:55] = 1                      # away handles the ball while it is dead
    stoppage = deadball.DeadBall(40, 70, 0, "oob")
    owner = control.possession(control.touches(on_ball), n, [(0, n)], [stoppage])

    assert (owner[:70] == 1).all()          # the touch at 45 is ignored, not back-dated
    assert (owner[70:] == 0).all()          # home restarts


# -- stage 3, M2c ---------------------------------------------------------------------------------

def test_the_line_a_crossing_went_over():
    tracking = synthetic_stoppage_match()      # out over the y = 68 touchline at x 80.8
    line, end, x = deadball.crossed_line(tracking.ball, PITCH, 794)
    assert (line, end) == (deadball.TOUCHLINE, 1) and abs(x - 80.8) < 0.1

    ball = np.full((10, 2), [104.0, 30.0], dtype=np.float32)
    ball[5:] = [106.5, 31.0]                   # past the x = 105.3 goal line
    assert deadball.crossed_line(ball, PITCH, 5)[:2] == (deadball.GOAL_LINE, 1)


def test_the_thrower_stands_on_the_touchline_the_ball_crossed():
    players = np.array([[[50.0, 60.0],          # home, 8 m inside
                         [52.0, 68.2],          # away, just behind the y = 68 touchline
                         [40.0, 67.8]]],        # home, on the line but 12 m along it
                       dtype=np.float32)
    teams = np.array([0, 1, 0])
    assert deadball.thrower_team(players, teams, PITCH, 0, end=1, x=52.0) == 1
    assert deadball.thrower_team(players, teams, PITCH, 0) == 1           # either line
    assert deadball.thrower_team(players, teams, PITCH, 0, end=0, x=52.0) == deadball.NOBODY

    players[0, 1] = np.nan                     # the thrower is not tracked
    assert deadball.thrower_team(players, teams, PITCH, 0, end=1, x=52.0) == 0  # 12 m along
    assert deadball.thrower_team(players, teams, PITCH, 0, end=1, x=70.0) == deadball.NOBODY
    players[0, 2] = [40.0, 67.0]               # 1 m inside: not standing on the line
    assert deadball.thrower_team(players, teams, PITCH, 0, end=1, x=52.0) == deadball.NOBODY


def test_attackers_in_the_box_tell_a_corner_from_a_goal_kick():
    # Home defends x = 0, so at the x = 105.3 end away defends and home attacks.
    players = np.full((1, 6, 2), np.nan, dtype=np.float32)
    teams = np.array([0, 0, 0, 1, 1, 1])
    players[0, :3] = [[95.0, 30.0], [97.0, 36.0], [92.0, 40.0]]   # home crowd the box
    players[0, 3:] = [[100.0, 34.0], [96.0, 30.0], [94.0, 38.0]]
    assert deadball.goal_line_team(players, teams, PITCH, 0, end=1, home_defends_low=True) == 0

    players[0, 1:3] = [[80.0, 30.0], [70.0, 40.0]]                # out of it: a goal kick
    assert deadball.goal_line_team(players, teams, PITCH, 0, end=1, home_defends_low=True) == 1
    # At the x = 0 end home defends, and nobody attacks its box: a goal kick to home.
    assert deadball.goal_line_team(players, teams, PITCH, 0, end=0, home_defends_low=True) == 0


def test_restart_laws_give_an_unseen_throw_in_to_the_player_on_the_touchline():
    tracking = synthetic_stoppage_match()
    tracking.ball[1100:] = np.nan              # the throw-in is never seen, so nobody is nearest
    tracking.players[1100:1250, 1] = [81.0, 68.3]   # away waits behind the line where it went out
    blocks = [(0, 2000)]

    found = deadball.detect(tracking, tracking.ball, blocks, tags=(), foul_stoppages=False,
                            restart_laws=False)
    assert found[0].evidence == "oob" and found[0].team == deadball.NOBODY
    found = deadball.detect(tracking, tracking.ball, blocks, tags=(), foul_stoppages=False)
    assert found[0].evidence == "oob" and found[0].team == 1


def test_a_stoppage_starts_where_the_ball_went_out_if_it_never_came_back():
    # Out over the touchline at 794 and never seen again; the players jog for 20 s, then stop.
    tracking = synthetic_stoppage_match()
    tracking.ball[800:] = np.nan
    tracking.speed[800:1300] = 2.0
    tracking.speed[1300:1700] = 0.5
    tracking.speed[1500:1700] = 3.0            # no second slowdown

    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=(), stoppage_starts=False)
    assert [s.evidence for s in found] == ["slow"] and found[0].start > 1250
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=())
    assert [(s.start, s.evidence) for s in found] == [(794, "slow")]

    tracking.speed[1000:1050] = 4.0            # everyone sprints for 2 s: that was open play
    found = deadball.detect(tracking, tracking.ball, [(0, 2000)], tags=())
    assert found[0].start > 1250


def test_out_since_needs_the_ball_unseen_on_the_pitch():
    tracking = synthetic_stoppage_match()
    ball = tracking.ball.copy()
    ball[800:] = np.nan
    collective = np.full(2000, 1.0, dtype=np.float32)
    crossings = deadball.line_crossings(ball, PITCH)
    assert deadball.out_since(ball, PITCH, collective, crossings, 1300) == 794
    ball[1000] = [50.0, 30.0]                  # seen back in play
    assert deadball.out_since(ball, PITCH, collective, crossings, 1300) is None


def test_whistle_starts_where_the_slowdown_set_in():
    collective = np.full(600, 3.0, dtype=np.float32)
    collective[300:] = np.linspace(3.0, 0.5, 300)[:300]
    slow_start = int(np.flatnonzero(collective < config.SLOW_SPEED)[0])
    start = deadball.whistle(collective, slow_start)
    assert start < slow_start and collective[start - 1] >= config.STOP_SPEED > collective[start]

    collective[:] = 1.5                        # play already slow: nothing to walk back to
    assert deadball.whistle(collective, 400) == 400


# -- stage 5 -----------------------------------------------------------------------------------------

def tackle_then_loose_ball(n=200, pass_by=False):
    """(owner, near, ball): home dribbles along x at 2 m/s, away is near the ball on [80, 86), and
    away's controlled touch starts at 120 inside a run of away near the ball from 118. The tackle
    sends the ball off across the pitch at 8 m/s; with pass_by the ball rolls past unchanged."""
    step = np.zeros((n, 2), dtype=np.float32)
    step[:, 0] = 2.0
    if not pass_by:
        step[86:118] = [0.0, 8.0]
    track = (30.0 + np.cumsum(step, axis=0) / 25.0).astype(np.float32)

    owner = np.zeros(n, dtype=np.int8)
    owner[120:] = 1                         # M1: away from its controlled touch
    near = np.full(n, control.NOBODY, dtype=np.int8)
    near[20:60] = 0                         # home on the ball
    near[80:86] = 1                         # away reaches it
    near[118:150] = 1                       # away receives and controls it
    return owner, near, track


def test_change_moves_back_to_the_tackle_that_won_the_ball():
    owner, near, track = tackle_then_loose_ball()
    placed, moved = boundary.place_changes(owner, near, track)

    assert moved == [(120, 80)]
    assert (placed[:80] == 0).all() and (placed[80:] == 1).all()


def test_a_pass_rolling_past_is_not_a_contact():
    owner, near, track = tackle_then_loose_ball(pass_by=True)
    placed, moved = boundary.place_changes(owner, near, track)

    assert moved == [] and np.array_equal(placed, owner)


def test_old_team_back_on_the_ball_bounds_the_search():
    owner, near, track = tackle_then_loose_ball()
    near[95:100] = 0                        # home gets near it again after the tackle
    placed, moved = boundary.place_changes(owner, near, track)

    assert moved == [] and np.array_equal(placed, owner)


def test_no_walking_back_across_a_lost_ball():
    owner, near, track = tackle_then_loose_ball()
    track[90:115] = np.nan                  # gone for a second, as after ball.repair
    placed, moved = boundary.place_changes(owner, near, track)

    assert moved == [] and np.array_equal(placed, owner)


def test_change_at_a_restart_stays_on_the_kick():
    owner, near, track = tackle_then_loose_ball()
    stoppage = deadball.DeadBall(100, 120, 1, "oob")
    placed, moved = boundary.place_changes(owner, near, track, [stoppage])

    assert moved == [] and np.array_equal(placed, owner)


# -- labels ------------------------------------------------------------------------------------------

def test_full_length_file_validates_and_marks_known_dead_ball(tmp_path):
    blocks = labels.graded_blocks()
    assert blocks == [(11185, 90652), (110808, 190198)]

    owner = np.zeros(N_VIDEO, dtype=np.int8)
    dead = np.zeros(N_VIDEO, dtype=bool)
    dead[110808:113742] = True
    codes = labels.to_codes(owner, dead, N_VIDEO, blocks)
    path = labels.write(str(tmp_path / "synthetic.csv"), codes)

    assert labels.check(path) is None
    assert codes[11184] == pm.X and codes[11185] == pm.H
    assert (codes[110808:113742] == pm.D).all() and codes[113742] == pm.H
    assert codes[190198] == pm.X


def test_written_file_has_the_baseline_format_and_unix_line_endings(tmp_path):
    codes = np.array([pm.X, pm.H, pm.A, pm.D, pm.X], dtype=np.int8)
    path = labels.write(str(tmp_path / "mine.csv"), codes)

    with open(path, "rb") as handle:
        written = handle.read()
    assert written == b"frame,label\n0,X\n1,H\n2,A\n3,D\n4,X\n"
    assert b"\r" not in written


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, "-q"]))
