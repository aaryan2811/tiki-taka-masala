"""Every tunable number in the pipeline, in one place, with the measurement behind it.

Each value rests on physics or the laws of the game, on a distribution of this match's tracking read
without any labels, or on the two DFL calibration matches. None is set on the public window, which
the rules make a correctness check, "not training data and not a tuning set" (RULES.md). Some of the
first values were chosen after looking at public-window cases, before the DFL dev set existed. Each
comment below says where that happened and what justifies the value now (DESIGN.md section 4,
"Re-checked on DFL").

Values measured on DFL matches come from the DFL / Sportec open data (Bassek et al. 2025,
https://doi.org/10.6084/m9.figshare.28196177.v1), CC BY 4.0, provided with authorization from the
DFL.
"""

FPS = 25

# -- stage 1: ball repair ------------------------------------------------------------------------

# A hard-struck ball reaches ~35 m/s. A jump faster than this between consecutive frames, in and
# out of the same detection, is a bad detection rather than a shot.
BALL_MAX_SPEED = 40.0  # m/s

# Visible runs shorter than this are dropped as spurious: a real ball that is tracked at all is
# usually tracked for more than a tenth of a second.
BALL_MIN_RUN = 3  # frames

# Gaps up to this long are filled by straight-line interpolation. A ball in flight for one second is
# close enough to a straight line in the plane; longer gaps are left to the possession logic.
BALL_MAX_INTERP_GAP = 25  # frames

# A goalkeeper slot moving faster than this between frames is being handed to the other keeper and
# sweeping across the pitch (possession/keepers.py), not following a player. Outfield slots move at
# most 8.9 m/s at the 99.9th percentile of graded frames and the speed channel peaks at 11.3; the
# home keeper slot has 238 graded frames above 12 m/s, almost all of them above 15.
KEEPER_MAX_SPEED = 12.0  # m/s

# -- stage 2: controlled touches -------------------------------------------------------------------

# A team is "on the ball" on a frame when its nearest player is within this distance and nearer than
# any opponent. Tracking positions are body centres, so the ball at a player's feet reads as ~0.5-1 m
# away; 2 m leaves room for tracking noise.
CONTROL_RADIUS = 2.0  # m

# ...and the ball must be moving with that player. A dribbled ball is pushed a little ahead and
# stays within a few m/s of the player; a pass or shot going past is 8+ m/s faster. Measured on the
# tactical ATD: with a player within 2 m, half the frames are under 2 m/s relative, a tenth over 7.
# Without this, a pass rolling past an opponent for 10 frames counted as an opponent's touch. On the
# DFL calibration matches, of 18 settings of radius, relative speed and length, the shipped 2 m,
# 3 m/s and 5 frames scores best under the merged 3 s truth (DESIGN.md section 4, M4).
CONTROL_MAX_RELATIVE_SPEED = 3.0  # m/s

# A controlled touch is at least this many consecutive on-the-ball frames by one team. Shorter
# contacts are deflections and challenges, which CHALLENGE.md says do not change possession.
CONTROL_MIN_FRAMES = 5  # frames, 0.2 s

# A controlled touch is one side of a duel, and void, when the other team is on the ball again
# within CONTEST_GAP frames of it ending and its nearest player was within CONTEST_RADIUS of the ball
# on every frame of it (possession/contest.py). Measured over the whole match, keeper slots split:
#   - of 65 single touches between two touches of the other team, the other team's touch began on
#     the very frame this one ended in 15 cases; the next shortest wait is 3 frames, and above that
#     the waits spread out over several seconds with no second gap. So "straight back" is at most 2.
#   - for the 27 touches the other team took over that directly, its nearest player stayed within
#     3 m of the ball throughout in 22 cases, and in none did it get 3-4 m away at most. For touches
#     it took over later (189), the same distance is spread out, median 5.6 m.
# Over the whole match the result does not move for any gap from 0 to 4 frames and any radius from
# 2.5 to 4 m. (python solution/experiments/contested_phases.py reproduces these numbers.)
CONTEST_GAP = 2  # frames
CONTEST_RADIUS = 3.0  # m

# -- stage 3: dead ball ----------------------------------------------------------------------------

# Half-open video-frame spans known to be dead ball without looking at any data. The second half is
# graded from 110808 but kicks off at 113742, and FAQ.md states all 2934 frames between are D. That
# is a published boundary, not an answer.
KNOWN_DEAD_SPANS = ((110808, 113742),)

# The ball counts as off the pitch when it is this far beyond a line. The lines are drawn ~12 cm
# wide and tracking is noisier than that.
OUT_OF_BOUNDS_MARGIN = 0.5  # m

# ...for at least this many frames. Shorter excursions are tracking noise near a line.
OUT_OF_BOUNDS_MIN_FRAMES = 3

# Mean speed of all tracked players, averaged over this window, for the slowdown signal.
SLOWDOWN_WINDOW = 25  # frames

# A stoppage candidate is the players as a whole moving slower than this for at least this long.
# The full-match distribution is unimodal -- slow play is common -- so this only proposes; nothing
# becomes dead ball without confirmation. At 1.4 m/s for 3 s the candidates cover 50 of the 59
# Smart Tagging set pieces.
SLOW_SPEED = 1.4  # m/s
SLOW_MIN_FRAMES = 75  # 3 s

# Confirmation windows around a candidate [start, stop): an out-of-bounds crossing up to
# OOB_BEFORE frames before it starts, or a set-piece tag starting from TAG_BEFORE frames before it
# to TAG_AFTER frames after it ends. Players can keep jogging for a while after the ball goes out.
# M2 first looked back 10 s and moved to 15 s after looking at the public window's stoppages
# (DESIGN.md section 4, M2). On the DFL calibration matches, degraded, 2 seeds: of the 180 dead balls
# that start with a crossing and have a slowdown over them, the slowdown starts within 15 s of the
# crossing for 96% (10 s: 85%, 20 s: 97%), and a crossing within 15 s confirms 0.8 slowdowns lying
# wholly in open play per match (25 s: 1.8). The rule written before measuring kept 15 s (DESIGN.md
# section 4, "Re-checked on DFL"; python3 -I solution/experiments/dead_ball_windows.py).
OOB_BEFORE = 375  # 15 s
TAG_BEFORE = 250  # 10 s
TAG_AFTER = 125  # 5 s

# A candidate with neither confirmation is still a stoppage when the slowdown goes deeper or lasts
# longer than open play ever does: collective speed below DEEP_SLOW_SPEED at some point, or below
# SLOW_SPEED for LONG_SLOW_FRAMES. Most fouls are confirmed this way (DESIGN.md section 4, M2b).
# This match has no labels to measure it on, so it was measured on the two open DFL matches, where
# BallStatus marks every dead frame: of the 139 candidates there with no line crossing in OOB_BEFORE,
# 36 lie wholly in open play, and in none of them does collective speed fall below 0.79 m/s or the
# slowdown last longer than 14.5 s (the next longest is 11.6 s). Of the 103 in dead time, 58 pass one
# of the two tests, and 96% of their frames are dead. The collective speed is distributed the same on
# this match and on DFL (10th/50th/90th percentile 0.71/1.54/3.06 m/s here, 0.67/1.65/3.15 there), so
# the thresholds carry over. The shorter, shallower dead candidates look exactly like open play on
# every player feature measured there, so they are left alone. (python
# solution/experiments/foul_stoppages.py --dfl <the dev-set audit's two .npz files> reproduces
# these.)
DEEP_SLOW_SPEED = 0.75  # m/s
LONG_SLOW_FRAMES = 375  # 15 s

# Without a line crossing to start from, a dead ball starts this long before the slowdown does.
# After a crossing during fast play, players take a median 42 frames to slow down (16 crossings),
# but when play was already slow the slowdown starts *before* the stoppage, and a dead ball that
# starts early eats the end of a real possession -- the expensive error. So: no lag. M2 first used
# 1.7 s and dropped it after looking at the public window's stoppages (DESIGN.md section 4, M2). The
# DFL calibration matches give the reason now: 17 of the 58 stoppages the slowdown test confirms
# there start before the true one, by up to 8.6 s (M2b). Since M2c, `whistle` (STOP_SPEED, set on
# calibration) decides where a stoppage with no crossing starts, and this only anchors out_since.
SLOW_LAG = 0  # frames

# The restart is the kick: the ball starting to move faster than KICK_SPEED for KICK_MIN_FRAMES,
# after KICK_REST_FRAMES without moving that fast (placed, held, or unseen). Searched for from
# RESTART_BACK frames before the slowdown ends to RESTART_AHEAD after it, latest kick wins -- players
# speed up only after the restart, so the slowdown ends after it.
#
# The rest is what tells a restart from a pass received and played on. M2 first set the rest and the
# search after looking at the public window's stoppages (DESIGN.md section 4, M2); both are now set
# on the two DFL calibration matches, degraded, 2 seeds, by rules written down before measuring
# (DESIGN.md section 4, "Re-checked on DFL"; python3 -I solution/experiments/dead_ball_windows.py):
#   - kick-like events in open play per match: 88 after 0.5 s of rest, 50 after 1 s, 40 after 2 s,
#     35 after 3 s. 2 s removes only a fifth of what is left at 1 s, so 1 s, the side that misses
#     fewer quick restarts. None of the 196 calibration dead balls is shorter than 2 s. DFL freezes
#     its ball in dead time, so it cannot show how long a real ball rests before a restart, only that
#     it cannot rest longer than the dead ball lasts.
#   - of the 211 true restarts that come before the slowdown over their dead ball ends, the slowdown
#     ends a median 1.0 s later, and 8 s back reaches 98% of them (4 s: 94%).
KICK_SPEED = 4.0  # m/s
KICK_MIN_FRAMES = 3
KICK_REST_FRAMES = 25  # 1 s
RESTART_BACK = 200  # 8 s
RESTART_AHEAD = 12  # ~0.5 s

# A stoppage that ends where its slowdown ends, with no restart kick, is joined to the next one when
# that starts less than JOIN_GAP later and no kick comes between: the ball never came back into play,
# the players only jogged for a moment (back to their positions for a kick-off, say). On DFL, of the
# consecutive slowdowns where the first ends with play still dead, every gap up to 2.7 s was dead
# throughout (6 of 6); from 3.9 s on some held live play. Without the join the frames between become a
# possession of a second or two, as after two of this match's goals.
JOIN_GAP = 75  # 3 s

# Stoppages shorter than this are not emitted: a false short stoppage splits a possession (DESIGN.md
# section 1), and real ones this short are rare. Of the 196 dead balls on the DFL calibration matches,
# 3 (1.5%) are shorter than 3 s and 10 (5.1%) shorter than 4 s (python3 -I
# solution/experiments/dead_ball_windows.py). The first comment here cited the public window; the
# rule written before measuring on DFL kept 3 s (DESIGN.md section 4, "Re-checked on DFL").
MIN_DEAD_FRAMES = 75  # 3 s

# The restart team is the team of the player nearest the ball as it is kicked, if within this
# distance, searched over this many frames from the kick (the ball may be unseen at the kick itself).
RESTART_TEAM_RADIUS = 3.0  # m
RESTART_TEAM_SEARCH = 25  # frames

# -- stage 3, M2c: when a stoppage starts, and who restarts it (detect's stoppage_starts and
# restart_laws). Every number below was measured on the two DFL calibration matches, J03WMX and
# J03WPY, degraded as the dev set degrades them (2 seeds), never on the held-out five or the public
# window. (python solution/experiments/restart_rules.py reproduces them.)

# A stoppage confirmed by its slowdown or a tag can start at a crossing up to OUT_SINCE_BACK before
# it, if the ball has not been seen back inside the lines since and the players' collective speed
# has stayed under OPEN_PLAY_SPEED throughout (deadball.out_since). After a goal or a ball out for a
# goal kick the players can jog on for 15-40 s, longer than OOB_BEFORE, and the possession before
# the stoppage then runs on through all of it. Open play reaches 3 m/s: 64% of 10 s stretches of it
# do, and a 30 s one with the ball unseen would almost surely. Dead time does not: after their first
# 2 s, 98% of stoppages of 10 s or more stay under 3 m/s throughout. The ball must be unseen as well,
# because a crossing followed by a quick throw-in and a few seconds of slow play also stays under
# 3 m/s. On calibration the rule moves 10 starts, by 5-59 s (median 23 s), and 95% of the frames it
# adds are dead. Three of the ten now start 14-48 s before the truth's: the crossing belonged to the
# stoppage before, and the ball came back into play unseen for a few seconds. Looking back 120 s
# instead of 60 s moves nothing more: the speed gate stops the walk first.
OUT_SINCE_BACK = 1500  # frames, 60 s
OPEN_PLAY_SPEED = 3.0  # m/s

# A stoppage with no crossing starts where the slowdown set in rather than where the players are
# down to SLOW_SPEED: after the last frame in the STOP_BACK before the slowdown with collective
# speed at or above STOP_SPEED (deadball.whistle). Over the 83 calibration stoppages with no
# crossing that truly began before our slowdown, collective speed at the true start had a lower
# quartile of 1.56 m/s (median 1.86): play is often slow already when the referee stops it. So at
# 1.6 m/s the new start is still late about three times in four, the cheap side (SLOW_LAG). The
# slowdown begins a median 1.0 s (fouls) to 8.6 s (tag only) after the true start; 10 s bounds the
# walk so that it cannot cross a stretch of play. On calibration it moves 102 starts by a median
# 1.0 s, and 83% of the frames it adds are dead; starts more than 1 s early go from 8 to 19, more
# than 3 s early from 3 to 5.
STOP_SPEED = 1.6  # m/s
STOP_BACK = 250  # frames, 10 s

# Who restarts by the laws of the game, from where the players stand rather than who touched the
# ball last. Who touched it last before it went out cannot be read from this tracking: on
# calibration, the team nearest the smoothed ball before a touchline crossing gave the right
# throw-in team 48 times and the wrong one 54 times, and controlled touches and velocity-change
# contacts did no better.
#
# A throw-in's taker stands on or behind the touchline (Law 15): the player nearest the touchline
# the ball crossed, within THROW_REACH of where it crossed, in the THROW_WINDOW before our restart,
# if within THROW_LINE of the line. Tracked positions are body centres: at calibration throw-ins the
# thrower stands a median 0.43 m behind the line, 90% of them less than 0.5 m inside it, and 90%
# within 15 m of where the ball crossed (median 4.3 m). On calibration's 117 touchline crossings:
# 90 right, 4 wrong, 23 unknown, where the nearest player at the restart gives 48 right, 13 wrong,
# 56 unknown.
# The window covers our restart being late: it is a median 0.3 s and a 75th percentile 1.0 s after
# the true one. With no crossing seen, either touchline counts: 44 right and 2 wrong over 108
# stoppages (corners, free kicks by the line, throw-ins whose crossing the tracker missed).
THROW_LINE = 0.5  # m inside the line, at most
THROW_REACH = 15.0  # m along the line from the crossing
THROW_WINDOW = 50  # frames, 2 s

# After the ball goes out over a goal line, the attacking team's players inside that penalty area
# at the restart: 3-7 at the 29 calibration corners, 0-2 at the 37 goal kicks (Law 16 keeps them out
# of the area until the ball is in play). At least CORNER_MIN_ATTACKERS means a corner to the
# attackers, fewer a goal kick to the defenders: 77 right and 5 wrong over 82 goal-line crossings,
# against 53 right, 3 wrong and 26 unknown from the nearest player. Penalty area as in Law 1.
CORNER_MIN_ATTACKERS = 3
PENALTY_AREA_DEPTH = 16.5  # m from the goal line
PENALTY_AREA_HALF_WIDTH = 20.16  # m either side of the centre line

# -- stage 5: boundary placement -------------------------------------------------------------------

# A change of possession goes on the new team's first contact since the old team was last near the
# ball (possession/boundary.py). A contact is a run of frames with that team nearest the ball within
# CONTROL_RADIUS across which the ball's velocity changes by at least CONTACT_MIN_CHANGE, comparing
# the mean velocity over frames CONTACT_GAP..CONTACT_SPAN before the run with the same after it.
# Five frames (0.2 s) a side average out per-frame noise, and starting two frames out skips the
# frames where the foot is on the ball; both are shorter than the half second or more between two
# touches of one move.
#
# A pass rolling past a player loses well under 1 m/s to the grass in that time; the smoothed,
# sometimes misplaced tracked ball changes by more than that on its own. Measured over the whole
# match across windows of the same lengths with nobody within 5 m, so that no touch is possible
# (1126 windows): median 1.4 m/s, 75th percentile 3.8. Across runs of up to 12 frames with a player
# near the ball (656): 63% change by 4 m/s or more, against 24% of the far windows. Velocity change
# alone cannot find touches -- frame by frame it is no larger near a player than far from everyone --
# so this only filters candidates that proximity has already found.
# (python solution/experiments/boundary_timing.py reproduces these numbers.)
CONTACT_MIN_CHANGE = 4.0  # m/s
CONTACT_GAP = 2  # frames
CONTACT_SPAN = 6  # frames

# -- stages 2b and 4, M4: the learned team in open play -------------------------------------------

# The per-frame model (possession/learned.py) and the segmentation's two numbers (segment.py) are
# fitted together by devset/train_model.py on the two DFL calibration matches and stored in
# possession/model.json, which that script rewrites byte for byte; its own constants hold the
# training settings (trees, target truth, seeds, rows). As shipped:
#   switch_cost 20   each change of team costs 20 in summed half log-odds, so a spell survives only if
#                    its evidence outweighs the two changes it needs. Tuned on out-of-half
#                    probabilities over 2.5-160: the mean of raw and merged 3 s spellF1 peaks at 20
#                    (0.474) and falls on both sides (10: 0.471-0.472, 40: 0.470, 160: 0.416).
#   min_frames 50    a possession inside a stretch of play lasts at least 2 s. It matters only with a
#                    low switch cost: at 20 the objective is 0.4738-0.4741 for 1, 25 and 50 frames.
# These are the numbers after the restart rest went from 2 s to 1 s, which retrained the model; the
# settings did not move. In the leave-one-match-out gate on the shipped rules the two folds tuned
# 20 / 50 and 20 / 1 (DESIGN.md section 4, M4 and "Re-checked on DFL").
