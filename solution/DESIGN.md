# Possession pipeline — design

Goal: beat `baseline_smooth50` (full-match spellF1 **0.2433**) on the private full-match ground
truth, with a repository that regenerates the submitted file exactly.

Status of each stage is tracked in the [build order](#5-build-order) table at the bottom.

---

## 1. What the metric punishes

Measured by damaging the public 5-minute ground truth in controlled ways and scoring the result
(`python solution/experiments/metric_sensitivity.py` reproduces this):

| error put into a perfect answer | spellF1 |
|---|---|
| all possession changes 5 frames late | 0.976 |
| 12 frames late | 0.906 |
| 1 s (25 frames) late | 0.776 |
| possessions shorter than 2 s merged into the previous one | 0.827 |
| **no dead balls at all** (previous team carried through) | **0.812** |
| **one false 1 s dead ball inside every long possession** | **0.644** |

Four rules follow from it.

1. **A false dead ball is worse than a missed one.** Inside a possession it splits it in two, so
   the real possession and both halves all fail to match. The dead-ball detector is tuned for
   precision, not recall.
2. **Timing matters more than per-frame labels.** A majority-vote window moves every boundary,
   which is why window smoothing cannot get far past the baseline. Decide segments first, then place
   each boundary precisely.
3. **"Ball not visible" does not mean "ball dead".** smooth50 writes `D` wherever the ball has been
   missing for about 4 s, which splits real possessions during long occlusions. When the ball is
   lost, keep the previous team. Forward-filling and writing `X` give the same spellF1 (the scorer
   forward-fills `X`), and the filled label is right more often than `X` on macroF1.
4. **Missing short possessions costs less than inventing them.** Only output a short possession
   when the evidence is strong.

Two facts from the metric code (`evaluation/possession_metrics.py`) that shape everything:

- Only `H`/`A` runs are possessions. `D` matters only as the thing that ends them: `H H D D H H`
  is two possessions.
- Matching is one-to-one, same team only, by temporal IoU, F1 averaged over IoU 0.3–0.7.

---

## 2. Pipeline

```
raw ATD ─► 0 load + cache ─► 1 ball repair ─► 2 evidence ─┐
Smart Tagging ────────────────────────────────────────────┤
                                     3 dead-ball detector ─┼─► 4 segmentation ─► 5 boundary snap ─► CSV
```

### Stage 0 — load and audit the data

- Run the starter kit's `load_atd.py` and `load_smart_tagging.py`; both alignment checks must pass.
- Inspect the ATD XML channel definitions: does the ball group carry anything beyond x, y and
  speed? A team or ball-state field would change the plan a lot.
- Parse the raw ATD once into arrays (ball, players, team of each track slot) and cache them as
  `.npz` under `solution/cache/`, which is gitignored. The cache is frame-level tracking and must
  never be committed.
- Attack direction per half from the kickoff positions.

### Stage 1 — repair the ball track

- Drop isolated detections and impossible jumps (over ~40 m/s).
- Interpolate short gaps (up to ~1 s) linearly.
- Classify each longer gap by who is nearest the ball when it disappears and when it reappears:
  same team at both ends means no change of possession; different teams means a change somewhere
  inside the gap, which stage 4 places.
- Optional and expensive, but the biggest possible gain: a ball detector on the tactical video.
  Its homography is valid on 100% of frames, so detections convert straight to pitch positions.
- **Goalkeeper slots.** The tracker swaps the two keepers between the teams' keeper slots, so a
  keeper slot is read as the team whose half it stands in (section 4, M3b).

### Stage 2 — per-frame evidence: P(H), P(A), P(D)

- **Controlled touches, not nearest player.** A team touches the ball when one of its players stays
  within ~1.5–2 m and moves with it for several frames. Possession changes only on an opponent's
  controlled touch, which matches the rules: deflections and single touches do not count.
- **Duels.** A controlled touch made with the other team at the ball throughout, and taken straight
  back by it, does not count (section 4, M3b).
- **Team shape for frames without a ball.** The team in possession spreads out and pushes its line
  up; the opponent compacts and presses. Features: team width and depth, centroid relative to
  attack direction, centroid speed, pressing distances. Training labels come from confident
  ball-visible frames (no hand labelling). Labelled matches from elsewhere are also allowed — the
  open DFL/Sportec matches carry ball status and possession per frame and load with `kloppy`, and
  Metrica publishes sample games. Check licences and whether their possession definition matches,
  and add noise (drop the ball, drop identities) to resemble our tracking.
- **Smart Tagging as a soft prior.** Build Up, Progression, Creating Chances, Attacking Transition,
  Long Ball: the tagged team has the ball. Defending in any third: the other team has it.
  Defensive Transition: the tagged team has just lost it. Team `N/A` (every throw-in) carries no
  team evidence.

### Stage 3 — dead-ball detector (high precision, minimum length ~3 s)

- Signals: the ball's last position outside the pitch lines; all players slowing down together and
  staying slow; restart geometry (stationary ball at a corner, on a touchline, in the six-yard box,
  a wall, both teams in their own halves); Smart Tagging `SET PIECES` and `GOALS`, whose ~5 s
  windows pin where a dead ball ends.
- A slowdown deeper or longer than open play ever gets is a stoppage on its own, with no crossing
  or tag (section 4, M2b). Short, shallow ones still need a crossing or a tag.
- If the tactical video has audio, a whistle detector is cheap and strong.
- Restart team: by the laws of the game, from where the players stand at the restart — the thrower
  on the touchline, attackers crowding the penalty area at a corner and kept out of it at a goal
  kick (section 4, M2c). Then the player nearest the stationary ball when play restarts, then the
  tag. Who touched the ball last cannot be read from this tracking.
- Frames 110,808–113,741 are `D` (second half graded before its kickoff; published in FAQ.md).

### Stage 4 — segmentation

- A hidden semi-Markov model over states H, A, D fed by calibrated stage 2–3 evidence, with a
  duration model per state (minimum lengths) and a separate penalty per kind of switch (turnover,
  ball going dead, restart). Viterbi decoding.
- Tune the switch penalties so possessions per minute and mean duration are realistic — **never**
  on the 17 public possessions.
- M3b found that the rules' evidence does not separate the touches the truth credits from the ones
  it refuses, and a duration model fed the same evidence could only trade them against each other by
  length (section 4, M3b).
- **Built as M4, for the team in open play** (section 4, M4). A model trained on labelled DFL
  matches weighs all of stage 2's evidence together into P(home) per frame. A two-state Viterbi with
  a switch cost and a minimum length decodes it between the stoppages, which stay stage 3's. D is
  not a state: the stoppages and restart teams come from the rules unchanged.

### Stage 5 — boundary snapping

- Each possession change moves to the new team's decisive touch: its first contact since the old
  team was last near the ball, controlled or not (section 4, M3). When the ball is not visible, it
  moves to the frame where team shape changes most.
- Each dead-ball start moves to the ball leaving the pitch, even when that was long before the
  slowdown, provided the ball was never seen back on it; with no crossing, to where the players'
  slowdown set in (section 4, M2c). Each end goes to the restart kick.
- Section 1 suggested this was worth roughly 0.1–0.2 spellF1 on its own. On our predictions, perfect
  timing within ±25 frames is worth nothing, because our changes are seconds off (section 4, M3).

---

## 3. Validation

- **Public 5 minutes:** correctness only, with `--bootstrap 1000` before believing any difference.
  One possession there moves spellF1 by ~6 points.
- **Dev set:** external labelled matches with our noise added, scored with the repository's
  `possession_metrics.py`.
- **Every run:** sanity statistics — possessions per minute, `D` share, duration histogram, count
  of very short possessions (`python solution/evaluate.py <file>`).

Not allowed, ever: hand labels on this match; human-produced data about it from any source;
anything derived from the private ground truth; training, tuning or choosing between versions on the
public window; rules or constants tied to frames of this match from watching it (RULES.md, as of the
organisers' `f8f2629`, 2026-10-23).

---

## 4. What the data turned out to be (stage 0, 2026-10-23)

- **Alignment checks pass**: 199,319 ATD records numbered 1..199,319, and both tagged kickoffs
  contain the real ones.
- **The ball group is x, y, speed only.** No team or ball-state channel, so possession and dead ball
  have to be inferred, as planned.
- **The raw slot order matches the XML Player order** (checked against the
  `DataFormatSpecification`), so `team_of_slot` from `starter/baseline.py` is right.
- **Speed is in m/s**, despite the XML calling the channel `speed_kph`: players peak at 9.4. The
  ball's own speed field peaks at 7.3 and does not match its movement (99th percentile 24.6 m/s from
  positions), so it is ignored. 8% of tracked slot-frames have a position but no speed.
- **Origin is a pitch corner.** Positions span 0–1 with excursions beyond the lines (ball y up to
  1.34), so `deadball.out_of_bounds` has something to find.
- **One goalkeeper slot per team** is flagged in the XML (`position_type`), useful for goal kicks.
  Barcelona attack left to right in the first half (`attack_direction_first_half`).
- **The ball is usually not at anyone's feet.** When it is visible, the nearest player is a median
  1.9 m away, and both teams are within 2 m on only 5.6% of those frames.

### M1 on the real match

| version | public spellF1 [90% CI] | public possessions (true 17) | full match, per minute |
|---|---|---|---|
| baseline_smooth50 | 0.326 [0.21, 0.45] | 26 | 5.6 |
| M1, proximity only (2 m, 5 frames) | 0.233 [0.16, 0.31] | 38 | 6.2 |
| **M1 + ball moving with the player (≤ 3 m/s relative)** | **0.411 [0.26, 0.56]** | **18** | **2.5** |

Proximity alone counted a pass rolling past an opponent as that opponent's touch. Requiring the
ball to move with the player fixed it. The threshold comes from physics and the measured
distribution, not from a sweep on the public window. The intervals still overlap the baseline's, so
this is not yet evidence of beating the floor on the full match. 2.5 possessions per minute with no
dead balls at all is plausible, since every missing stoppage merges two possessions.

### M2 on the real match

How it works is in `possession/deadball.py`'s docstring. In short:
- **Candidates:** stretches where the players as a whole slow down.
- **Confirmation:** the ball crossing a line, or a set-piece tag.
- **Boundaries:** start at the crossing, end at the restart kick.
- **Restart team:** the player nearest the ball at the kick.

What the data said along the way:

- **Slowdown alone is not a dead-ball signal.** The full-match collective speed is unimodal; slow
  play is common. At 1.4 m/s for 3 s it proposes 135 candidates covering 50 of the 59 set-piece
  tags, so it only proposes.
- **Players can keep jogging up to 15 s after the ball goes out.** Then there is a gap: no crossing
  15–20 s before a tag-confirmed slowdown.
- **A restart is told from a pass by how long the ball rested before it.** With 0.5 s of rest, 602
  kick-like events fell in open play; with 2 s, 193, and the restart timing was unchanged.
- **No backward lag without a crossing.** Players are sometimes slow *before* a foul, and a dead ball
  that starts early eats a real possession.
- **Restart team:** nearest player and set-piece tag agree on 19 of the 21 stoppages where both
  name a team. Throw-ins are all tagged `N/A`, which is where the nearest player is needed.
- **The tracked ball is heavily smoothed.** A kick ramps up over ~20 frames, so restarts are found
  up to about a second early, which is the cheap side.

| version | public spellF1 [90% CI] | macroF1 | public possessions (true 17) | D precision / recall (public frames) |
|---|---|---|---|---|
| M1 | 0.411 [0.26, 0.56] | 0.512 | 18 | — |
| M2, first settings (0.5 s rest, 4 s restart search, 10 s crossing window, 1.7 s lag) | 0.487 [0.34, 0.63] | 0.711 | 20 | 0.74 / 0.41 |
| **M2, measured settings** | **0.522 [0.37, 0.67]** | **0.773** | **19** | **0.93 / 0.53** |

Full match: 55 stoppages (27 tag only, 8 crossing only, 19 both, plus the known pre-kickoff span),
median 21.6 s; 29.6% of graded frames dead; 2.65 possessions per minute.

**Honest caveats.** Each change was prompted by one of the two tagged stoppages in the public
window and then checked against a full-match distribution, but two stoppages cannot validate
anything. (Since the organisers' rule change, the public window may not steer a value at all. All
four were re-measured on the DFL calibration matches by rules written beforehand: the restart rest
moved to 1 s, the other three stayed. See "Re-checked on DFL", after M4.) The 29.6% dead share is likely *below* the truth, since a match typically has the ball out
of play for 35–40% of the time. The two public stoppages M2 misses (12851, 15889) have neither a tag
nor a crossing, most likely fouls, and that is the main recall gap left.

### M3: boundary timing

**The premise did not hold.** M2 had only 3 of the 20 public changes within ±25 frames. That is not
a shared lag of about 11 frames. We moved every M2 change onto the nearest true change of the same
kind, as an oracle: it reads the public truth, so it is a diagnosis and never a setting. Within
±25 frames this gains nothing:

| oracle window | 0 | ±25 | ±50 | ±100 | ±200 |
|---|---|---|---|---|---|
| public spellF1 | 0.522 | 0.522 | 0.556 | 0.644 | 0.844 |

Our changes are 1.5–8 s away from the true ones. The main reason is *which* touch fires a change, not
where on that touch the change is placed. Section 1's 12-frame cost applies to a perfect answer, not
to ours.

**What the truth means by "owning".** We read the public truth against the tactical video at the
boundaries, with the ATD drawn through the homography. Only public-window frames were used, to
understand the labels and never to make any; nothing from the video feeds the pipeline.

- **A change is stamped 0–8 frames after the new team's decisive touch, controlled or not.**
  - 14846: an Athletic tackle knocks the ball off a Barcelona dribbler at about 14839, and a
    team-mate plays the loose ball at about 14847. Truth: A from 14846. Our first Athletic
    controlled touch: 14886.
  - 15346: a Barcelona sliding tackle pokes the ball loose at about 15342. Truth: H from 15346 for
    19 frames, until an Athletic player kicks the loose ball at about 15362; truth A from 15365.
  - 13704: a Barcelona pass rolls among Athletic players for 30 frames, untouched, and the truth
    stays H. It flips when an Athletic player first touches the ball, with a one-touch clearance at
    about 13703.
  - 12665: an Athletic header meets a lofted ball in mid-flight at about 12658. This is the least
    clear of the four.
- **A touch that does not win the ball does not count.** An Athletic player dribbles for 2 s under
  pressure (11416–11466) and is dispossessed. Another holds the ball for about 0.7 s with a Barcelona
  player on his back (12433–12450) and loses it. The truth credits Athletic with neither. So a
  contested phase belongs to the team that comes out of it with the ball, starting from that team's
  decisive touch.
- **It is not a fixed lag after one kinematic event, and it is not the nearest player.** The offsets
  from kicks, receptions and nearest-team flips to the true change all vary by tens of frames.

**Why M2's changes miss.** The per-change offsets come from `experiments/boundary_timing.py`. Offsets
beyond about ±50 frames are a different event, not a mistimed one.

| cause | public true changes | what it does |
|---|---|---|
| control comes long after the decisive touch | 12665 (+42), 14846 (+40) | late. Stage 5 fixes this |
| a touch credited that the truth does not credit (contested phase) | 11351 (the 2 s Athletic dribble: +115); Athletic touches at 11710 and 12438 inside one long Barcelona possession; 15272–15290; 15656 | invented possessions, around which 11553, 15365 and 15766 are mismatched |
| short true possession with no controlled touch | 11553 (71 frames), 15346 (19), 15766 (33), 17293 (87) | missed |
| tracked ball is wrong | 13785: from 13731 to 13810 the ATD ball sits on the centre spot while the real ball is at the touchline | late (+66) |
| smoothed ball arrives early | 13704 (−12), 17838 (−11) | about 11 frames early. These are two of M2's three matches; the third is a line crossing at −2 |
| dead ball | 12851 and 15889, fouls with no tag or crossing, which also misplace their restarts 12989 and 16400; 17380 (a tag-only start at the slowdown, −54); 18389 (the truth restarts 28 frames after our kick) | M2's job |

**What the tracking can see of a touch.** A touch changes the ball's velocity, but frame by frame this
signal is useless here. Over ±5 frames of fully seen raw ball, the 90th percentile of the change is
6.3 m/s with a player within 1.5 m and 6.9 m/s with nobody within 5 m. The smoothed, sometimes
misplaced ball changes velocity as much on its own.

Across a whole short proximity run (up to 12 frames, nearest within 2 m) the signal does appear.
Measured from just before the run to just after it:
- 63% of 656 runs change by 4 m/s or more;
- so do 24% of 1126 equally long windows with nobody near (median 1.4 m/s, 75th percentile 3.8).

That is enough to filter candidates that proximity has already found, not enough to find touches.

**The rule** (`possession/boundary.py`):
- Detection stays on the controlled touch, which is good evidence that the ball changed hands.
- Each change moves back to the new team's first contact since the old team was last near the ball.
  A contact is a proximity run across which the ball's velocity changes by 4 m/s or more.
- The search stops at three bounds:
  - a ball gap the repair left open (over 1 s), which is M1's rule;
  - the start of the old team's spell;
  - the run that holds the controlled touch. Walking into it would only find the ball arriving,
    which on the smoothed track is already early.
- Changes at restarts stay on the kick.

The reason for each number is in `config.py`. Over the whole match, 38 of 251 changes move, by a
median of 48 frames (1.9 s), 90th percentile 112. Without the gap bound, 4 more would move by 11–17 s
across stretches where the ball was seen only 4–46% of the time.

| version | public spellF1 [90% CI] | spell F1 at IoU 0.3 / 0.4 / 0.5 / 0.6 / 0.7 | transition F1 ±12 / ±25 (matched of 20) | macroF1 | full match: possessions per min, under 1 s, under 2 s |
|---|---|---|---|---|---|
| M2 | 0.522 [0.37, 0.67] | 0.667 / 0.611 / 0.556 / 0.444 / 0.333 | 0.15 (3) / 0.15 (3) | 0.773 | 2.65, 14, 37 |
| **M3** | **0.544 [0.38, 0.69]** | 0.667 / 0.611 / 0.611 / 0.500 / 0.333 | 0.20 (4) / 0.25 (5) | 0.780 | 2.65, 14, 43 |

**Honest caveats.**
- **Two changes are not evidence.** On the public window the rule moves exactly two changes, both
  towards the truth: 12707 to 12647 against 12665, and 14886 to 14844 against 14846. The intervals
  overlap almost entirely. The case for the rule is the diagnosis; the threshold comes from the
  whole-match distribution, and no public number was used to set it.
- **The threshold is on a knife-edge for one case.** A pass rolling past at 13671 changes by
  3.96 m/s. A threshold of 3 m/s would take it and pull the 13704 change 21 frames early. That is
  exactly the kind of choice we refuse to make on the public window.
- **Spells under 2 s went from 37 to 43.** Moving a change back shortens the spell before it, and
  10 such spells fell under 2 s. One example: a 9-frame Athletic spell at 64875, between an Athletic
  touch and a Barcelona contact. Most look like contested phases the truth gives to the eventual
  winner, which is a segmentation problem.
- **The rule cannot touch what dominates the public error:** touches credited in contested phases,
  missed short possessions, the stuck ball and the fouls.

**What comes next, by size.**
1. **Segmentation.** Give a contested phase to the team that wins it. The table's second row is
   five of the public errors, and the diagnosis now defines the target the semi-Markov model was
   meant to hit.
2. **Fouls with neither a tag nor a crossing.** These are 12851 and 15889, two of the four public
   stoppages, and each also misplaces its restart.
3. **Ball repair.** A ball that sits still in open play with no player within reach, as on the
   centre spot from 13731 to 13810, is a false detection.
4. **Restart timing.** On the two stoppages we find, the truth restarts 11–28 frames after our kick.
   Where the kick is seen (18389), smoothing does not explain it: the centre of the kick's speed ramp
   is no later than our kick frame.

### M3b: contested phases

**What the touches the truth refuses actually are.** We read every case on the tactical video with
the ATD drawn through the homography, public-window frames only, as in M3. Nothing from the video
feeds the pipeline.

| public case (our spell) | what the tracking credits | what the video shows | truth |
|---|---|---|---|
| 11710 (A, 103 frames) | away keeper slot, 10 frames controlled | **Barcelona's** keeper passing to a team-mate: the away keeper slot stands in Barcelona's goal | H |
| 15656 (H, 116) | home keeper slot, 23 frames controlled | **Athletic's** keeper, sitting in the home keeper slot | A |
| 15272 (H, 18) | Barcelona controlled for 18 frames, Athletic on the ball on the frames either side | an Athletic player shields the ball with a Barcelona player on him; their tracked positions overlap | A |
| 12438 (A, 47) | Athletic receives a Barcelona pass, 12 frames controlled; Barcelona on the ball 1.4 s later | the receiver is closed down from behind and the ball pops back to a Barcelona player | H |
| 12798 (H, from 53 frames before a missed foul) | Barcelona, 6 frames controlled | a Barcelona player deflects an Athletic pass on to an Athletic player; the smoothed ball slows through the change of direction | A |

The short true possessions we miss:

| true possession | what the video shows | what the tracking has |
|---|---|---|
| 15346 (H, 19) | Barcelona sliding tackle pokes the ball loose; Athletic kicks it 19 frames later (M3) | Athletic controlled until 15353; Barcelona near the ball 15357–15367. The tracked ball runs 15–30 frames behind |
| 15766 (H, 33) | a Barcelona poke knocks the ball off an Athletic player, who chases it down about 35 frames later | nobody within 2 m of the tracked ball, which stays 2–3 m from every player; the chasing Athletic player then reads as in control |
| 11553 (A, 71) | an Athletic interception sends the ball into a crowded box; Barcelona controls it at about 11622 | one 7-frame uncontrolled Athletic contact, with a Barcelona player 0.9 m away |
| 17293 (H, 87) | Barcelona's last touches before the ball goes out | Barcelona near the ball for 4 and 12 frames, never controlled; our dead ball starts 54 frames early (M2's tag-only start) |

**What the truth means, refined.** M3 concluded that a contested phase belongs to the team that comes
out of it with the ball. The video says that holds only where the ball never leaves the duel:
- A touch that takes the ball away is credited even when the old team recovers the loose ball:
  the tackles at 15346 and 15766, and M3's clearance at 13704.
- What the truth refuses are touches that never took the ball away:
  - a keeper read with the wrong team;
  - a flip of the nearest team inside a duel;
  - a ball received under pressure and lost straight back;
  - a deflection.

**What the tracking can separate.** Two of the four kinds. Both checked over the whole match without
the truth (`experiments/contested_phases.py`):
- **Keeper slots.** The tracker hands each keeper slot to whichever keeper it sees.
  - Of the graded frames, the away keeper slot is in Barcelona's half on 16,392 and the home slot in
    Athletic's half on 1,610.
  - Changing keeper, the home slot sweeps across the pitch: 238 frames faster than 12 m/s, where
    outfield slots reach 8.9 m/s at the 99.9th percentile.
- **Duel flips.** Take the 65 single controlled touches between two touches of the other team with
  no stoppage between.
  - In 15 of them the other team is back on the ball on the very frame the touch ends. The next
    shortest wait is 3 frames, and above that the waits spread out over several seconds.
  - Look at the 27 touches the other team takes over that directly. In 22 its nearest player stays
    within 3 m of the ball throughout, and in none does it get 3–4 m away at most. For the 189 it
    takes over later, the same distance has a median of 5.6 m.

The other two kinds cannot be separated with this tracking:
- **Received under pressure and lost.** At 12438 Barcelona is back 1.4 s later and nobody is
  credited. The truth does credit the clearances at 12707 and 13692, though the old team was back
  3.1 and 6.1 s later. Over the whole match those waits have no gap to put a threshold in, and the
  pressure is not constant through the touch (Barcelona's nearest player is within 3 m on 58% of its
  frames).
- **Deflection.** The deflection at 12798 shows 6 frames of control, against 7 for the clearance at
  13692 that the truth credits.

Nor can the short possessions we miss. None has a controlled touch. Two are tackles the tracked ball
does not show, and one ends in a stoppage that starts early. Crediting uncontrolled contacts would
find them, but the team not in possession makes 330 contacts over the match (3.1 per minute, against
the truth's 3.4 possessions per minute on the public window) with nothing to tell the real ones apart.

**Why no semi-Markov model.** It would segment the same evidence. Where the evidence cannot tell a
credited touch from a refused one, a duration model can only trade them against each other by
length. That is the "keep a change only if the team holds the ball N seconds" rule that 15346 and
15766 rule out. The model becomes worth building once some evidence separates them: team shape (M4)
or a better ball (M5).

**The rules.**
- **`possession/keepers.py`** splits each keeper slot into a home slot and an away slot.
  - On every frame the position goes to the team that defends the half it is in. The attack
    direction comes from the XML, switched at the second graded block.
  - Frames where the slot moves faster than 12 m/s are dropped.
  - Every stage then sees the repaired tracking, including M2's restart team.
- **`possession/contest.py`** makes a controlled touch void when two things hold:
  - the other team's nearest player was within 3 m of the ball on every frame of the touch;
  - the other team's next controlled touch begins within 2 frames of it ending.

  The ball never left the two players, so the touch was a flip on tracking noise.

Each number's reason is in `config.py`. Over the whole match the result does not move for any gap
from 0 to 4 frames and any radius from 2.5 to 4 m.

**How it interacts with M2 and M3.**
- **Dead balls.** The 55 stoppages are unchanged, and touches inside them still count for nothing.
  One restart team changes, the stoppage at 50262, from away to home: the player nearest its kick is
  Barcelona's keeper, 1.1 m away, who was sitting in the away slot.
- **Boundary placement.** M3 still moves every change that remains, but a void touch takes its
  walk-back with it. One of them (Athletic, 132465) had been walked back 8.6 s, over loose ball, to an
  Athletic contact at 132249, so that whole 8.8 s spell goes.

**Over the whole match.**
- **Keepers split:** 22 fewer changes of possession (251 to 229). Every change it removes, adds or
  moves lies within 10 s of a stretch where a keeper slot stands in the other team's half, and 32 of
  the 33 lie inside one.
- **Duels resolved:** 10 spells removed and none added (229 to 209 changes). Spells under 1 s go
  from 14 to 7.

| version | public spellF1 [90% CI] | spell F1 at IoU 0.3 / 0.4 / 0.5 / 0.6 / 0.7 | transition F1 ±12 / ±25 (matched of 20) | macroF1 | public possessions (true 17) | full match: possessions per min, under 1 s, under 2 s, D |
|---|---|---|---|---|---|---|
| M3 | 0.544 [0.38, 0.69] | 0.667 / 0.611 / 0.611 / 0.500 / 0.333 | 0.20 (4) / 0.25 (5) | 0.780 | 19 | 2.65, 14, 43, 29.6% |
| + keepers split | 0.588 [0.40, 0.75] | 0.688 / 0.688 / 0.688 / 0.500 / 0.375 | 0.22 (4) / 0.28 (5) | 0.802 | 15 | 2.44, 14, 42, 29.6% |
| duels alone, for reference | 0.518 [0.35, 0.67] | 0.647 / 0.588 / 0.588 / 0.471 / 0.294 | 0.21 (4) / 0.26 (5) | 0.782 | 17 | 2.43, 7, 28, 29.6% |
| **M3b: keepers split + duels** | **0.547 [0.36, 0.73]** | 0.667 / 0.667 / 0.600 / 0.467 / 0.333 | 0.24 (4) / 0.29 (5) | 0.804 | 13 | 2.25, 7, 28, 29.6% |

**Why we ship the version with the lower public number.** On the public window each step removes
exactly the spells it was built for, and nothing else:
- the keeper split removes 11710 and 15656;
- the duel rule removes 15272.

The drop from 0.588 to 0.547 is one merge. Without the false Barcelona spell at 15272, Athletic's
14844–15272 and 15290–16052 join into 14844–16052. That matches neither true Athletic possession
well (IoU 0.41 with 14846–15346), for two reasons:
- the true split at 15346 is the 19-frame tackle the tracking cannot see;
- the foul at 15889 is missed.

The false spell was scoring by accident: it sat 56 frames before a real split it had nothing to do
with. So the choice rests on the diagnosis and the whole-match distribution, as the rules require.
The intervals overlap almost entirely either way.

**Honest caveats.**
- **The evidence is small.** Five public cases are all there is for what the truth refuses, and two
  of them are one data error.
- **The keeper split assumes a keeper never plays the ball in the other half.** A keeper going up for
  a late corner would be read as the other team. We have not looked for one, because checking it
  would mean reading the video outside the public window.
- **Possessions per minute fell from 2.65 to 2.25**, against the public truth's 3.40. The deficit was
  there before. Its main source is the stoppages M2 misses: each one merges two possessions, and M2's
  29.6% dead share is likely below the truth. Removing false changes lowers the count further, but a
  false change is wrong at any count.
- **The 132249 spell may have been real.** If the truth credits that contact the way it credits the
  clearance at 13704, the duel rule removed a real 8.8 s possession along with the duel at its end.
- **What still dominates the public error:**
  - 12438 and 12798, which the tracking cannot separate;
  - the four short possessions we miss;
  - the stuck ball at 13785;
  - the fouls at 12851 and 15889;
  - the restarts at 17380 and 18389.

**What comes next, by size.**
1. **Fouls with neither a tag nor a crossing** (12851, 15889). The 15889 miss is also what makes the
   merged Athletic spell above run on to 16052.
2. **Ball repair.**
   - The stuck ball (13731–13810) is a false detection in open play.
   - At 15346 and 15766 the tracked ball runs behind or beside the real one. Only a better ball
     (M5) can make those short possessions visible.
3. **Restart timing**, as in M3.

### M2b: foul stoppages

**What M2 leaves.** M2 confirms a slowdown candidate only with a line crossing or a set-piece tag, and
68 of the 135 candidates have neither. Smart Tagging tags few fouls: its 59 set pieces are 26
throw-ins, 12 corners, 10 goal kicks, 6 kick-offs and only 5 free kicks. The DFL
feasibility audit (a research note made before the dev set, kept out of this line) finds about 20
stoppages per match with no crossing, mostly fouls, and about 100 stoppages in all, against our 55.
The dev set's own truth confirms the total: 92–113 stoppages per match over all seven
(`solution/devset/README.md`, Truth).

In the public window, each of the two missed fouls has a candidate: 12894–13017 for 12851, and
15945–16429 for 15889. The window's other three candidates with no evidence (12006, 12425, 14439) are
3–4 s lulls inside Barcelona possessions. So a third confirmation must take the first two of these
five and leave the last three. (That framed the goal in public-window cases, which the rules no
longer allow. The test's thresholds were measured on the DFL matches alone, below, and the held-out
five confirmed them.)

**What can tell a stoppage from a lull.** Every signal below was measured without the ground truth.

- **The players: deeper or longer than open play (shipped).** This match has no labels to measure
  that on. The two open DFL matches of the audit do: their `BallStatus` marks every dead frame, and
  player speeds are unaffected by DFL freezing the ball during dead time. We ran M2's candidate stage
  on DFL player speeds.
  - **The scale carries over.** Collective speed has 10th, 50th and 90th percentiles of
    0.71 / 1.54 / 3.06 m/s here and 0.67 / 1.65 / 3.15 m/s on DFL.
  - **DFL candidates.** There are 223. Of the 84 with a crossing, only 1 lies wholly in open play,
    so M2's crossing rule holds up there. Of the 139 without one, 103 lie mostly in dead time and 36
    wholly in open play.
  - **Open play has hard limits.** Among those 36, collective speed never falls below 0.79 m/s, and
    no slowdown lasts longer than 14.5 s; the next longest is 11.6 s.
  - **The test.** Falling below 0.75 m/s, or staying under 1.4 m/s for 15 s or more, passes 58 of
    the 103 dead candidates and none of the 36 open-play ones. 96% of the frames it confirms are dead.
  - **The rest look like open play.** The 45 dead candidates it leaves are short and shallow. On
    every player feature measured they overlap the open-play ones: length, lowest collective speed,
    the fastest players (90th percentile), the share standing still, the longest run below 1 m/s,
    and the speed before the slowdown.
  - **On this match**, the test also passes 45 of the 67 candidates M2 confirms by crossing or tag,
    an independent check on stoppages we already know. It passes 18 of the 68 that M2 leaves.
- **The restart: a long rest with the ball unseen (measured, not shipped).** Take a kick after at
  least 6 s of rest, with the ball unseen for at least 80% of that rest and a non-keeper taking it.
  That describes 16 of the 35 kicks that end M2's stoppages, and 11 of the 114 kicks in fast open
  play (no slowdown in the 8 s before), a likelihood ratio of about 5. It would add 12894–12980,
  77877–78123 and 115677–115985. We do not ship it, for three reasons:
  - **It cannot be checked anywhere.** DFL freezes the ball in dead time. The dev set will
    synthesise its dead-ball ball track, so it cannot check a ball-based signal either.
  - **The one public case sits on the threshold.** 12894 rests 152 frames, against 150.
  - **It would not help the public window anyway.** Our Barcelona spell 12798–13692 already matches
    the truth's possession after the foul (IoU 0.78). Splitting it would leave a false 96-frame
    Barcelona spell before the stoppage: the 12798 deflection from M3b.

  (The last two reasons read the public window, which may no longer decide anything. The first
  reason stands alone and is the one `TECHNICAL.md` gives.)
- **Keepers.** Keeper slots take 10 of the 34 kicks that end candidates M2 leaves. A keeper holding
  the ball after a catch looks exactly like a restart, so any restart rule has to exclude keepers.
- **Opponents 9.15 m away (Law 13).** We took the median distance from the kick spot to the nearest
  opponent over the 2 s before the kick.
  - Free kicks, corners and kick-offs that M2 confirms: 8–26 m.
  - Throw-ins: from 3.4 m up.
  - Kicks after 2 s of rest in open play: a median of 5.9 m, with a fifth at 9 m or more (defenders
    with time on the ball).
  - The 12851 foul was taken quickly, with opponents 3 m from the spot.

  Not used.
- **A stationary ball (Law 13 again).** Not usable: the smoothed tracked ball wobbles at 0.5–2 m/s
  even before seen restarts.
- **The ball vanishing.** The tracker drops the ball in long blocks: 133 of the 150 raw gaps last
  4 s or more, with a median of 10.7 s. It is missing on 77% of M2's dead frames and 25% of the other
  graded frames. On the public window it is missing on 34% of the dead frames, 15% of Athletic's and
  none of Barcelona's. It leans towards dead time but is common in open play, so it is at most
  supporting evidence.
- **A defensive wall.** Not pursued. Only free kicks within shooting range get one, and those are
  the free kicks Smart Tagging is most likely to tag already.
- **`GOALS` tags.** These run from the scoring move to the kick-off, so they cannot pin the goal
  itself. After two of the four goals, part of the celebration still reads as possession (the
  follow-ups below).

**The rules.**
- **`deadball.beyond_open_play`** confirms a candidate when collective speed falls below
  `DEEP_SLOW_SPEED` (0.75 m/s), or the slowdown lasts `LONG_SLOW_FRAMES` (15 s) or longer. Its
  evidence is `slow`.
  - It starts at the slowdown, with no lag, as M2 does for tag-only stoppages. DFL backs that: for
    the 58 dead candidates the test confirms, the slowdown starts a median 23 frames after the
    stoppage. But 17 of them start before it, by up to 8.6 s, so any backward lag would eat real
    possessions.
  - It ends at the restart kick, as in M2.
- **`deadball.join`** joins two stoppages into one under three conditions:
  - the first ends with no restart kick, where its slowdown ends;
  - the next starts less than `JOIN_GAP` (3 s) later;
  - nothing is kicked in between.

  Then the ball never came back into play. On DFL, where the first of two consecutive slowdowns ends
  with play still dead, 6 of 6 gaps under 3 s were dead throughout, and 5 of 8 of 3–6 s. Without the
  join, the jog back after two of this match's goals became possessions of 2.6 s and 1.4 s.

Each number's reason is in `config.py`. `python solution/experiments/foul_stoppages.py` reproduces this
match's numbers, and with `--dfl` the audit's two files, which live outside the repository, the DFL
ones.

**Over the whole match.**
- **Stoppages: 55 to 71.** 16 are wholly new, with a median of 22.3 s; only one is under 10 s
  (64019, 6.6 s, players down to 0.61 m/s). Two existing kick-off stoppages grow by joining the
  goal celebration before them (1,610 frames).
- **None sits in fast play.** In the new stoppages the players get down to 0.30–1.13 m/s, and over
  the 10 s before each they average 1.4–3.2 m/s: play winding down, never a sprint. 14 of the 16 end
  on a restart kick.
- **Dead share: 29.6% to 37.1%**, against 41.5% and 44.3% on the two DFL matches.
- **Possessions: 238 to 246 (2.25 to 2.32 per minute)**, eight more for 16 new stoppages, for three
  reasons:
  - Five of the new stoppages cover a change of possession M3b had placed inside dead time, either a
    touch while the ball was dead or the restart itself. In two of them (15945, 118618) a spell made
    only of such a touch goes.
  - Three restarts now give the ball to the other team. One is 133947: 19.5 s with the ball unseen
    near Athletic's goal after a Barcelona attack, restarted by Athletic.
  - The two goal celebrations only lengthen stoppages that were already there.
- **Spells under 1 s: 7 to 8; under 2 s: 28 to 30.** One spell is cut short by a new stoppage
  (Athletic, 33204, 1.6 s before a whistle). The other is a restart answered by an Athletic touch
  within a second (143078).
- **Restart team unknown: 9 to 12**, for stoppages that end with no kick or nobody near the ball.

| version | public spellF1 [90% CI] | spell F1 at IoU 0.3 / 0.4 / 0.5 / 0.6 / 0.7 | transition F1 ±12 / ±25 (matched of 20) | macroF1 | D precision / recall (public frames) | public possessions (true 17) | full match: stoppages, D, possessions per min, under 1 s, under 2 s |
|---|---|---|---|---|---|---|---|
| M3b | 0.547 [0.36, 0.73] | 0.667 / 0.667 / 0.600 / 0.467 / 0.333 | 0.24 (4) / 0.29 (5) | 0.804 | 0.933 / 0.532 | 13 | 55, 29.6%, 2.25, 7, 28 |
| + slowdown test | 0.566 [0.37, 0.74] | 0.690 / 0.690 / 0.621 / 0.483 / 0.345 | 0.29 (5) / 0.35 (6) | 0.898 | 0.957 / 0.841 | 12 | 73, 37.1%, 2.34, 8, 31 |
| **M2b: + join** | **0.566 [0.37, 0.74]** | 0.690 / 0.690 / 0.621 / 0.483 / 0.345 | 0.29 (5) / 0.35 (6) | 0.898 | 0.957 / 0.841 | 12 | 71, 37.1%, 2.32, 8, 30 |

**On the public window**, the test confirms exactly one of the five candidates with no evidence:
15945–16429, the 15889 foul.
- **Its boundaries.** It is now dead from 15945 to 16399, against the truth's 15889–16400: 56 frames
  late at the start (no lag), and on the restart.
- **One false spell goes.** The false Barcelona spell 16052–16283 inside it, a Barcelona player
  kicking the ball back during the stoppage, now counts for nothing. That is why the public
  possessions go from 13 to 12.
- **One merge remains.** Athletic's 14844–15945 still covers three true possessions, split by the
  15346 tackle and the 15766 poke that the tracking cannot see (M3b).
- **12851 is still missed**, along with the three lulls, as intended.

The join changes nothing on the public window.

**Honest caveats.**
- **The thresholds rest on 36 open-play candidates from two Bundesliga matches.** 15 s sits only
  0.5 s above the longest of them. This match has Barcelona, a side that keeps the ball, so a long,
  slow spell of passing at the back is more likely here than there.
  - A rough extrapolation of DFL's open-play length tail (it halves about every 2.5 s beyond 6 s)
    puts the 15 s test at about half a false stoppage per match.
  - 0.70 m/s and 20 s would have kept 45 of DFL's 103 dead candidates instead of 58.
  - Section 1's numbers put a missed stoppage and a false one at about the same cost per event, so
    the looser setting is the better bet. It is a bet, not a measurement on this match.
- **The public gain is one stoppage and one false spell removed.** The intervals overlap almost
  entirely. The case rests on DFL and the whole-match statistics, not on 0.547 to 0.566.
- **Two new stoppages end before their restart.** They have no kick in the restart search window, so
  they end where the players speed up. At 53498, a ball resting near Athletic's box is not kicked
  until 54157, 94 frames after the stoppage ends. That is restart timing, which stays out of scope.
- **Short, shallow stoppages are still missed.** On DFL that is 45 of 103, about 22 per match. Here
  that includes 12851, and 77877 and 115677 look like it (13–15 s, ball unseen, then a kick). M2b
  finds 71 stoppages against the roughly 100 that DFL suggests.

**What to re-check on the dev set** (`solution/devset/`, once it exists):
- `DEEP_SLOW_SPEED` and `LONG_SLOW_FRAMES`, first of all, against many more open-play candidates.
  The question is whether open play ever goes deeper than 0.79 m/s or longer than 15 s.
- `JOIN_GAP`.
- `SLOW_LAG` = 0 for stoppages confirmed by the slowdown test.
- The DFL part of `experiments/foul_stoppages.py`, which belongs there.
- Not the restart rule. No dev set with a synthetic dead-ball track can check it.

**M2b on held-out matches** (2026-10-27; `solution/devset/README.md` has the tables). The dev set now
has seven DFL matches. The thresholds were set on two of them (J03WMX, J03WPY); the other five were
downloaded afterwards and used for nothing before this check. No threshold was changed.
- **The test on the five held-out matches**, measured exactly as it was calibrated (DFL's own player
  speeds, `python solution/devset/foul_check.py`). This reproduces the calibration counts above to the
  candidate.
  - **False stoppages: 3 of 100** open-play candidates without a crossing pass, about 0.6 per match.
    That is close to the half per match extrapolated above. All three pass on length (15.8, 16.6 and
    24.5 s), and one is also deep (0.68 m/s). Open play does last longer than 15 s sometimes; it
    went deeper than 0.75 m/s only in that one long slowdown.
  - **Caught: 199 of 347** mostly dead candidates (57%, against 56% on calibration). 95% of the frames
    the test passes are dead (96% on calibration).
  - **The alternative named above** (0.70 m/s or 20 s) gives 2 false stoppages and 168 caught on the
    held-out five. That is 31 fewer stoppages for one fewer false one, so the held-out matches do
    not argue for a change. They show the cost of the current setting: a false stoppage now and
    then, from a long lull in open play.
  - **`JOIN_GAP`:** of consecutive slowdowns whose first ends with play dead, 25 of 27 gaps under 3 s
    were dead throughout (6 of 6 on calibration). From 3 s to 6 s, 21 of 30 were.
- **On the degraded dev-set tracking** the same test passes 0.5 open-play candidates per held-out
  match (13 over 5 matches and 5 degradation seeds). They come from 3 open-play stretches, each 14–21 s
  long.
- **Scores on the held-out five** (5 degradation seeds, paired bootstrap over matches and
  possessions). M2b against the same pipeline with the slowdown test switched off
  (`foul_stoppages=False`; the join stays):
  - spellF1 +0.024 [+0.018, +0.030] raw and +0.032 [+0.025, +0.041] merged at 3 s, with tags;
  - +0.036 and +0.048 without tags;
  - on calibration: +0.022 / +0.030 and +0.034 / +0.048.

  Every interval excludes zero, and M2b is ahead on each of the five matches. Dead-ball recall goes
  from 0.52 to 0.74 with tags (0.39 to 0.72 without) at precision 0.96. Stoppages go from 60 to 78
  per match, against a truth of 103. M2b against smooth50: +0.097 raw, +0.178 merged at 3 s.

**What comes next, by size.**
1. **Short fouls (12851).** Only the ball could separate them from lulls, and only a better ball
   (M5) or the dev set's evidence on how often open play produces long unseen rests could make that
   safe.
2. **Goal celebrations.** After two of the four goals, the frames between the goal and the kick-off
   stoppage still read as the scoring team's possession, for example from 46494 to 47081.
3. **Ball repair and restart timing**, as before.

### M2c: when a stoppage starts, and who restarts it

Everything in this section up to "Held-out check" was measured on the two DFL calibration matches
(J03WMX, J03WPY), degraded as the dev set degrades them, 2 seeds, with synthetic tags. The held-out
five and the public window set nothing. `python solution/experiments/stoppage_ceiling.py --dfl-dir
<dfl>` reproduces the ceilings and pair tables, and `python solution/experiments/restart_rules.py
--dfl-dir <dfl>` the rule tables and every number next to the M2c settings in `config.py`.

**Where the stoppages we find lose score.** We replaced one thing at a time with the DFL truth (the
truth goes in through `deadball.detect`; the restart team is the truth's on the frame after the dead
ball). SpellF1:

| if this were perfect | raw | merged2s | merged3s |
|---|---|---|---|
| nothing (M2b) | 0.334 | 0.432 | 0.448 |
| restart team, on our stoppages | 0.344 | 0.447 | 0.466 |
| start only, on our stoppages | 0.358 | 0.463 | 0.483 |
| stop only, on our stoppages | 0.346 | 0.447 | 0.464 |
| start and stop, on our stoppages | 0.370 | 0.477 | 0.496 |
| start, stop and team: the truth stoppages we find | 0.400 | 0.518 | 0.539 |
| every truth stoppage, no restart team | 0.389 | 0.497 | 0.518 |
| every truth stoppage, with its team | 0.440 | 0.567 | 0.591 |
| team in open play, with our stoppages | 0.869 | 0.644 | 0.567 |

The stoppages we already find are worth more fixed than the ones we miss are found. Of the start's
+0.035 (merged3s), +0.026 is in stoppages with no crossing and +0.018 in starts 10 s or more late.
Our 307 stoppages that overlap a true one:
- **Start, ours minus truth, median by evidence:** crossing −0.0 s; slowdown only +1.0 s; slowdown
  and tag +4.8 s; tag only +8.6 s. Stoppages confirmed without a crossing start late, and the
  possession before them runs into the dead ball.
- **Restart team:** 159 right, 18 wrong, 130 unknown. 116 of the unknowns end with no restart kick,
  so the nearest-player rule has no ball to look at. Throw-ins alone: 45 / 14 / 69.

**Who touched it last cannot be read from this tracking.** The laws decide a restart by the last
touch: a throw-in to the other team, a corner or a goal kick by which side touched it. Before a
touchline crossing, the team nearest the smoothed ball gives the right throw-in team 48 times and
the wrong one 54 times. Velocity-change contacts (`boundary.contacts`) and controlled touches do no
better. Before a goal-line crossing, the nearest team is as bad, 33 / 34.

**Where the players stand at the restart can be read, and the laws fix it.**
- **Throw-in (Law 15):** the thrower stands on or behind the touchline, and opponents keep 2 m away.
  At calibration throw-ins the throwing team's player nearest that line, in the 2 s before our
  restart, stands a median 0.43 m behind it, 90% of them less than 0.5 m inside it, and a median
  4.3 m along it from where the ball went out (90% within 15 m).
- **Corner or goal kick (Laws 16, 17):** at a corner the attackers crowd the penalty area for the
  cross; at a goal kick they must stay out of it until the ball is in play. Attackers in that area
  at our restart: 3–7 at all 29 corners, 0–2 at all 37 goal kicks.
- **Fouls with no crossing:** nothing as clean. Opponents keep 9.15 m from a free kick, but with the
  ball unseen nothing says where the kick is. No rule, except that a player standing on a touchline
  at a restart is its taker (a free kick by the line, a corner, a throw-in whose crossing the tracker
  missed).

**When play stopped.**
- **The ball went out long before the slowdown.** After a goal kick or a goal, players can jog on
  for 15–40 s before they slow down, longer than M2's 15 s crossing window. The ball was then last
  seen beyond a line and never seen back inside it. Open play stays under 3 m/s collective speed
  in only 36% of 10 s stretches; after their first 2 s, 98% of stoppages of 10 s or more stay under
  it throughout.
- **The players slow down after the whistle.** The candidate starts once collective speed is down
  to 1.4 m/s, a second or more after the players react. At the true start of the 83 no-crossing
  stoppages that began before our slowdown, collective speed has a lower quartile of 1.56 m/s and a
  median of 1.86 m/s: play is often slow already when the referee stops it.
- **Not used:** the ball's last sighting heading out near a line, for crossings the dropped ball
  hides. Before the 107 no-crossing slowdowns the ball was last seen a median 4.8 s earlier, but only
  14 times heading out within 10 m of a line at 2 m/s or more. On 3 of those 14 the straight line
  crosses more than 3 s before the true start, by up to 7 s: the ball was played on unseen. Too few
  to help, with its errors on the expensive side.
- Nothing keys on how the ball moves inside a stoppage, which the dev set invents
  (`degrade.dead_ball_track`).

**The rules** (`possession/deadball.py`, behind two `label_frames` flags; on by default since the
held-out check below). Each number's reason is in `config.py`.
- **`restart_laws`:** the restart team comes from the laws first, then the nearest player at the
  kick, then the tag.
  - After a goal-line crossing (`goal_line_team`): 3 or more attackers in that penalty area at the
    restart is a corner to them, fewer a goal kick to the defenders.
  - After a touchline crossing (`thrower_team`): the team of the player nearest that touchline,
    within 15 m along it of the crossing, in the 2 s up to the restart, if within 0.5 m of the line.
  - With no crossing: the same with either touchline and no limit along it.
- **`stoppage_starts`:**
  - `out_since`: a stoppage starts at the earliest crossing in the 60 s before it since which the
    ball has never been seen inside the lines and collective speed has stayed under 3 m/s.
  - `whistle`: one with no crossing starts after the last frame in the 10 s before its slowdown with
    collective speed at or above 1.6 m/s, if there is one.

**What each rule recovers on calibration** (M2b's own stoppages, right / wrong / unknown against
the truth's restart team):

| crossing | pairs | M2b | law rule alone | law rule, then M2b's | where M2b is unknown: team carried / law rule |
|---|---|---|---|---|---|
| touchline (105 throw-ins, 11 free kicks, 1 corner) | 117 | 48 / 13 / 56 | 90 / 4 / 23 | 99 / 10 / 8 | 33 / 23 / 0 vs 41 / 4 / 11 |
| goal line (29 corners, 37 goal kicks, 16 other) | 82 | 53 / 3 / 26 | 77 / 5 / 0 | 76 / 5 / 1 | 6 / 20 / 0 vs 23 / 3 / 0 |
| none (41 free kicks, 22 throw-ins, 13 goal kicks, 32 other) | 108 | 58 / 2 / 48 | 44 / 2 / 62 | 82 / 4 / 22 | 30 / 18 / 0 vs 22 / 1 / 25 |
| all | 307 | 159 / 18 / 130 | 211 / 11 / 85 | 257 / 19 / 31 | 69 / 61 / 0 vs 86 / 8 / 36 |

Where M2b and the law rule both name a team, they agree on 115 and differ on 13: the law rule is
right on 12 of those.

- On the throw-ins where no thrower is found, the nearest player is right only 2 times in 8. The
  throw-in's taker is usually the one the tracker loses, and the nearest player is then a marker.
  We keep the fallback anyway: 8 cases are too few to set a rule on.
- Two of the five goal-line "wrong" have no restart team in the truth (the half ended dead).

The start rules, each alone on M2b's stoppages:

| rule | starts moved | moved by, q10 / median / q90 | truth dead / play frames added | starts > 1 s / > 3 s early (before) |
|---|---|---|---|---|
| `out_since` | 10 | 14.8 / 23.1 / 45.4 s | 6554 / 309 (95% dead) | 3 / 3 (0 / 0) |
| `whistle` | 102 | 0.4 / 1.0 / 5.5 s | 4050 / 851 (83% dead) | 19 / 5 (8 / 3) |

- `out_since`'s three early starts each jump a few seconds of play the tracker did not see. The
  crossing belonged to the stoppage before, and in each case that stoppage had ended with no restart
  kick, exactly like the three moves that were right. So no guard was added.
- `whistle` makes 11 more starts over a second early, the expensive side. It still adds dead frames
  4.8 times as often as play ones.

Both rules on, against M2b on its own stoppages (299 pairs instead of 307: earlier starts merge 8
stoppages into their neighbours):
- restart team 248 / 17 / 34, from 159 / 18 / 130;
- start, ours minus truth, q25 / median / q75: −0.2 / −0.0 / +1.0 s, from −0.2 / +0.0 / +3.1 s; by
  evidence the medians go from +1.0 / +4.8 / +8.6 s to +0.2 / +0.6 / +2.3 s (slow / slow and tag /
  tag only);
- starts more than 1 s early: 28, from 15; more than 3 s early: 15, from 10.

Scores on calibration (`run_devset.py --matches J03WMX J03WPY`, 5 seeds), paired against M2b, raw /
merged2s / merged3s. These are biased: the rules were built on these matches.

| variant | tags | no tags |
|---|---|---|
| `restart_laws` | +0.003 / +0.007 / +0.010 | +0.004 / +0.009 / +0.014 |
| `stoppage_starts` | +0.005 / +0.007 / +0.007 | +0.002 / +0.003 / +0.003 |
| both | +0.008 / +0.014 / +0.016 | +0.007 / +0.013 / +0.018 |

**Pre-registered, before the held-out matches were scored** (2026-10-27). One run of
`python3 -I solution/devset/run_devset.py --dfl-dir <dfl> --seeds 5 --bootstrap 1000`, which scores
all seven matches. Reported: the held-out five's paired differences against M2b for three
variants, `restart_laws`, `stoppage_starts` and both, under raw, merged2s and merged3s, with and
without tags; their D precision / recall, stoppages per match and possessions per minute; and the
per-match spellF1. A rule goes on by default if, on the held-out five, its own paired difference
against M2b is above zero in all six cells (three truths, two tag modes) and its 90% interval
excludes zero under merged3s in both tag modes. Otherwise it stays behind its flag, off. "Both" is
reported to show whether the two add up; it decides nothing. A negative result is reported as it
comes. Nothing is re-tuned afterwards.

**Held-out check** (the one run, 2026-10-27; `solution/devset/README.md` has every table). Paired
differences in spellF1 against M2b on the five held-out matches, 5 seeds, 90% intervals:

| variant | tags | raw | merged2s | merged3s |
|---|---|---|---|---|
| `restart_laws` | tags | +0.009 [+0.005, +0.013] | +0.011 [+0.007, +0.017] | +0.013 [+0.008, +0.017] |
| `restart_laws` | no tags | +0.009 [+0.006, +0.013] | +0.013 [+0.009, +0.017] | +0.014 [+0.010, +0.018] |
| `stoppage_starts` | tags | +0.006 [+0.004, +0.008] | +0.007 [+0.005, +0.009] | +0.007 [+0.005, +0.010] |
| `stoppage_starts` | no tags | +0.005 [+0.003, +0.007] | +0.006 [+0.004, +0.008] | +0.006 [+0.005, +0.009] |
| both | tags | +0.014 [+0.010, +0.019] | +0.018 [+0.012, +0.023] | +0.019 [+0.014, +0.026] |
| both | no tags | +0.014 [+0.011, +0.018] | +0.018 [+0.014, +0.023] | +0.020 [+0.015, +0.025] |

- **Both rules pass the pre-registered test, so both are on by default.** Every point estimate is
  above zero, every interval excludes zero, and 100% of the bootstrap draws are above zero.
- **They add up.** Both together gain about the sum of the two alone.
- **The held-out matches gain more than the calibration ones did.** The restart laws give +0.009 raw
  on the held-out five, against +0.003 on calibration. So nothing points to overfitting the two
  calibration matches.
- **M2c is ahead on each of the five held-out matches**, with and without tags, under the raw and
  merged 3 s truths. Held-out spellF1 with tags: 0.330 / 0.383 / 0.411 (raw / merged2s / merged3s)
  for M2b, 0.344 / 0.401 / 0.430 for M2c.
- **D precision / recall**, with tags: 0.96 / 0.74 to 0.95 / 0.77. Stoppages per match: 77.6 to
  75.5, against a truth of 102.6. The earlier starts merge some stoppages into their neighbours.
- **Possessions per minute**, with tags: 2.40 to 2.35, against a truth of 4.49 raw and 2.91 merged
  at 3 s. Where a restart team is now named and the same team makes the next touch, the short spell
  the carried team used to get after the stoppage disappears. Merged stoppages lose the spell between
  them too.
- **Against the ceilings** (calibration, 2 seeds, merged3s): M2c takes M2b's 0.448 to 0.468. Perfect
  restart teams alone on our stoppages would give 0.466, perfect starts alone 0.483, and perfect
  starts, stops and teams 0.539.

**On the real match** (`python solution/run.py`, then `python solution/evaluate.py
evaluation/results/submission.csv --bootstrap 1000`). This is a sanity check, not the deciding number.

| version | public spellF1 [90% CI] | macroF1 | D F1 (public) | public possessions (true 17) | full match: stoppages, restart team unknown, D, possessions per min, under 1 s, under 2 s |
|---|---|---|---|---|---|
| M2b | 0.566 [0.37, 0.74] | 0.898 | 0.895 | 12 | 71, 12, 37.1%, 2.32, 8, 30 |
| **M2c** | **0.566 [0.37, 0.74]** | **0.901** | **0.897** | 12 | 71, 7, 38.6%, 2.34, 9, 31 |

- **Restart teams.** Five of the 12 unknown teams are now named. One more changes: at 169071, a
  stoppage with no crossing, from Athletic (nearest player) to Barcelona (a player on the touchline).
  That removes a 0.6 s Athletic spell at 170179.
- **Starts.** 40 starts with no crossing move earlier by the `whistle` rule, by 0.2–9.6 s. `out_since`
  fires once: the stoppage at 58578 now starts 20.2 s earlier, at a crossing the ball never came
  back from.
- **On the public window.**
  - The 15889 foul now starts at 15929, 16 frames nearer the truth.
  - The 17380 stoppage starts 12 frames earlier, at 17314, so it is now 66 frames early instead of
    54. Its players slowed before the stoppage: the expensive case the `whistle` rule's level was
    set to keep rare.
  - Neither changes which possessions match, so spellF1 is unchanged.
- **Short spells.** One of the possessions the earlier starts shorten, Athletic's at 33204, drops
  under 1 s (1.6 s to 0.4 s). It was already cut short by a stoppage in M2b. Two new spells under
  2 s follow restart teams that are now named (25483, 132017), each ended by the other team within
  2 s.

**Honest caveats.**
- **One dev set.** Seven DFL matches, one provider, and FC Barcelona's possession style is not
  represented. The thrower and penalty-area rules rest on the laws of the game and on where players
  stand, which do not depend on the league. The 3 m/s and 1.6 m/s levels rest on how fast players
  move, which the M2b check found distributed alike on this match and on DFL.
- **The restart-team rules read player positions at our restart frame.** On the dev set the players
  are real (degraded) and the restart frame comes from the real kick. A kick taken far from where
  we place the restart could fool them. The 2 s window covers our usual lateness (q75 1.0 s).
- **`out_since` reads the ball's absence.** It fires only when the tracker never sees the ball back
  inside the lines. The dev set's dropout in dead time is fitted to this match's, but the dead-ball
  ball's path is invented: carried straight to the restart spot. On the real match it fires once.
- **The whistle rule makes more early starts:** more than 1 s early, 15 to 28 over the calibration
  pairs with both rules. The scores say the trade is worth it. On this match's possession-heavy play
  slow spells before a foul may be commoner, and those are the cases it gets wrong (17380).
- **Throw-ins whose thrower the tracker loses** fall back to the nearest player, who is then usually
  a marker (2 right of 8 on calibration). We left this alone rather than set a rule on 8 cases.
- **Kick-offs still start late** (a median 24 s on calibration). After a goal the ball is carried
  back to the centre spot in sight, so `out_since` never fires.

**What comes next, by size** (calibration ceilings, merged3s):
1. **Possession changes in open play.** Giving our stoppages the truth's team on every frame of play
   is worth +0.12 (0.448 to 0.567), far more than anything left at the stoppages. That needs pass
   chains or the team-shape classifier (M4).
2. **Stoppages we miss**, mostly short and shallow (every truth stoppage with its team: 0.591, against
   0.539 for the ones we find). As M2b found, the players cannot separate them; only a better ball
   (M5) can.
3. **What is left at our stoppages:** starts still late after goals, starts the `whistle` rule makes
   early, and restart timing (the end edge, at most +0.016, and not measurable on the dev set).

### M4: a learned team in open play

Everything in this section up to the held-out test was measured on the two DFL calibration matches
(J03WMX, J03WPY), degraded as the dev set degrades them, with synthetic tags. The held-out five and
the public window set nothing.

**Where open play loses score.** `python3 -I solution/experiments/open_play.py --dfl-dir <dfl> --sweep`
(2 seeds, tags) reproduces every number here. M2c with the merged 3 s truth's team put on some of its
play frames (an upper bound, since the merged truth's own team goes in):

| if the team were perfect… | raw | merged2s | merged3s | poss/min, merged3s (truth 2.91) |
|---|---|---|---|---|
| nowhere (M2c) | 0.346 | 0.449 | 0.468 | 2.25 |
| on play frames with the ball unseen (22% of play; M2c right on 75%) | 0.420 | 0.524 | 0.546 | 3.05 |
| on play frames with the ball seen (78% of play; M2c right on 87%) | 0.503 | 0.657 | 0.736 | 2.83 |
| on every play frame | 0.569 | 0.734 | 0.821 | 3.00 |

Most of what is left is on frames where the ball is seen. Three cheap routes did not reach it:
- **Pass chains.** Two consecutive contacts by one team at most 3 s apart, with the ball seen between,
  agree with the truth on 88% of 1329 chains made by two different track slots. But on the 130 where
  M2c names the other team, the truth sides with the chain only 37% of the time. Handing the rest of
  M2c's spell to the chain's team scores 0.348 / 0.440 / 0.454.
- **A team-shape classifier for unseen-ball frames.** Gradient boosting on the team centroids along
  the attack, their spreads, speeds and tracked counts, how long since the ball was seen and where,
  trained on one calibration match and tested on the other. It is right on 81–82% of unseen-ball
  play frames, against M2c's 75%. Pasted onto unseen stretches of at least 2 s or 5 s after a 3 s
  smoothing, merged3s goes to 0.452–0.465, against 0.468. (The script fits it on one thread, so a
  rerun repeats these rows exactly; a first, multi-threaded run gave 0.449–0.466.)
- **Looser touch thresholds.** Radius 2 / 2.5 / 3 m, relative speed 3 / 5 / 8 m/s and 3 or 5 frames
  (seed 0): raw rises to 0.383 at most, but merged3s falls. M2c's (2 m, 3 m/s, 5 frames) has the
  best merged3s of the 18 (0.480).

So each rule has used what its own evidence separates, as M3b found for the duel rule. What is left
is to weigh all of it together, and to segment the result into possessions rather than paste it
frame by frame.

**What M4 is** (`label_frames(learned_model=...)`, on by default since the held-out check below):
- **`possession/learned.py`: P(home has the ball) on every frame.** 63 features, the same 27 for
  each team from its own end, plus 9 shared:
  - what the touch rule reads: distance and relative speed of the nearest player; the share of frames
    near the ball and in control over 1 s and 3 s before and after; the length of the run near the
    ball;
  - time since and until the team's last and next controlled touch, contact and frame near the ball;
  - M2c's owner, and the team that took the restart this stretch of play follows;
  - team shape: centroid along the team's attack, spreads, mean speed, tracked count, how fast the
    centroid moves towards the other goal; where the ball was last seen along that attack, and how
    far from the centroid;
  - shared: ball seen, time since and until it is seen, its speed and velocity change, time since
    the restart and until the next stoppage, and the age and remaining length of M2c's current
    possession.

  The model is symmetric between the teams by construction: a tree ensemble trained on every frame
  and its copy with the teams swapped, whose log-odds are the half-difference of the two scores; or
  a logistic regression on home-minus-away differences with no intercept. Nothing it learns can be
  about being at home.
- **`possession/segment.py`: possessions.** Inside each stretch of play between M2c's stoppages, a
  two-state Viterbi path collects each frame's log-odds for the team it names and pays a switch cost
  per change. Optionally every possession but a stretch's first and last lasts a minimum length.
  M2c's stoppages and restart teams are kept: a known restart team has the ball on the stretch's
  first frame.
- **Training** (`devset/train_model.py`): the calibration matches only, degraded with seeds 10–13
  (never the dev set's scoring seeds 0–4), with tags; every 5th play frame where the truth names a
  team. Switch cost and minimum length are tuned on the training match itself, with probabilities
  that each half gets from a model trained on the other half. The objective is the mean of raw and
  merged3s spellF1. Fixed settings, no early stopping, one thread: a rerun writes the same
  `possession/model.json`. The trees are exported node by node and evaluated in numpy; `run.py`
  needs neither scikit-learn nor the DFL data.
- **Target: the truth merged at 2 s.** The DFL flag credits touches lost straight back, and the
  challenge does not ("a simple touch of the ball without control" does not change possession,
  CHALLENGE.md), so the raw truth is the wrong thing to learn. Merged at 2 s, J03WMX has 3.50
  possessions per minute (raw 5.25, merged at 3 s 2.97). The gate below chose 2 s over 3 s. (This
  sentence first called 2 s "the closest of the three to the public truth's 3.4"; that comparison
  set nothing and is withdrawn, see "Re-checked on DFL".) The segmentation is tuned on raw and
  merged3s, the two truths the gate reads.

**The gate, written before it was run** (2026-10-27). `python3 -I solution/experiments/learned_gate.py
--dfl-dir <dfl>` runs it.
- **Folds.** Train on J03WMX and test on J03WPY, then the other way round. Everything the variant
  sets (model, switch cost, minimum length) comes from the training match alone.
- **Test.** The test match degraded with seeds 0–4, as `run_devset.py` does, with and without tags.
  M2c runs on the same tracking. Both folds' test matches are pooled, and M4 − M2c is the paired
  difference of seed-averaged spellF1 (`score.paired`).
- **Go** only if a variant's difference is at least **+0.010** under raw *and* merged3s, with tags
  *and* without (four cells, point estimates). merged2s and the 90% intervals are reported and decide
  nothing.
- **Variants** allowed inside the gate, each reported with its numbers: the model (trees or
  logistic), the target (merged at 2 s or 3 s), the segmentation (minimum length tuned, or none).
  The first run is trees, merged at 2 s, minimum length tuned. If several pass, the one with the
  highest mean over the four cells goes on, and the held-out test judges that one alone.
- **No-go:** stop; write the negative result here; the flag stays off and the held-out five stay
  untouched.

**Gate result** (one run, 2026-10-27, with all four variants in it; nothing else was tried). Both
folds' test matches pooled, 5 seeds, paired against M2c on the same tracking, 90% intervals:

| variant | tags | raw | merged2s | merged3s | poss/min (truth 4.78 raw / 2.91 merged3s) |
|---|---|---|---|---|---|
| M2c (spellF1) | tags | 0.342 | 0.441 | 0.458 | 2.23 |
| **trees, merged2s** | tags | **+0.062** [+0.033, +0.096] | +0.095 [+0.072, +0.118] | **+0.095** [+0.077, +0.111] | 2.42 |
| trees, merged2s, no minimum length | tags | +0.058 [+0.033, +0.087] | +0.095 [+0.072, +0.118] | +0.099 [+0.080, +0.116] | 2.32 |
| logistic, merged2s | tags | +0.027 [+0.004, +0.054] | +0.055 [+0.023, +0.088] | +0.066 [+0.035, +0.096] | 2.29 |
| trees, merged3s | tags | +0.044 [−0.001, +0.091] | +0.070 [+0.038, +0.100] | +0.086 [+0.067, +0.103] | 2.34 |
| M2c (spellF1) | no tags | 0.329 | 0.425 | 0.442 | 2.18 |
| **trees, merged2s** | no tags | **+0.064** [+0.034, +0.098] | +0.097 [+0.073, +0.123] | **+0.100** [+0.080, +0.116] | 2.38 |
| trees, merged2s, no minimum length | no tags | +0.058 [+0.034, +0.087] | +0.097 [+0.073, +0.122] | +0.102 [+0.082, +0.119] | 2.28 |
| logistic, merged2s | no tags | +0.030 [+0.005, +0.059] | +0.059 [+0.026, +0.096] | +0.072 [+0.038, +0.105] | 2.25 |
| trees, merged3s | no tags | +0.046 [−0.001, +0.095] | +0.072 [+0.038, +0.106] | +0.088 [+0.069, +0.105] | 2.31 |

- **Every variant passes; trees on the 2 s truth go on** (mean of the four cells +0.080, against
  +0.079 without a minimum length, +0.066 on the 3 s truth and +0.049 for the logistic model). M2c's
  row is the dev set's calibration row to the third decimal, so the comparison is against the
  pipeline as it stands.
- **The gain is the team, not more possessions.** Possessions per minute rise from 2.23 to 2.42,
  still below the merged truth's 2.91. On play frames (tags, against the merged 3 s truth) the team is
  right on 89.9% with the ball seen (M2c 86.8%) and 86.3% with it unseen (M2c 72.9%). Frame by frame,
  before segmentation, the model is right on 89.7% and 85.7%: the segmentation adds little accuracy
  but decides which spells exist.
- **The segmentation is not fragile.** The tuned settings differ between folds (switch cost 20 and
  no minimum when trained on J03WPY; 10 and 50 frames when trained on J03WMX), and without a
  minimum length the result is within 0.006 everywhere.
- **The trees gain about 60% more than the linear model,** whose only interactions are the shared
  features times M2c's owner. So combinations of the evidence matter, which is what the hand-written
  rules could not weigh.
- **Training on the 2 s truth beats training on the 3 s truth** even when scored on the 3 s truth.
- **What this does not measure.** The dev set's ball has no 15–30-frame lag, no identity switches
  inside a slot and no stuck centre-spot ball, and the DFL flag stamps a change by its own convention.
  A model keyed to fine ball–player timing could lose part of this on the real match, which is what
  the drift check and the real-match numbers below look at.

**Pre-registered, before the held-out matches were scored** (2026-10-27).
- **The model:** `python3 -I solution/devset/train_model.py --dfl-dir <dfl>`. Trees on the merged 2 s
  truth, trained on both calibration matches (seeds 10–13, tags), switch cost and minimum length tuned
  on them out of half; written to `possession/model.json` once, before the run.
- **The run:** one run of `python3 -I solution/devset/run_devset.py --dfl-dir <dfl> --seeds 5
  --bootstrap 1000`, with `adapter.PIPELINES` holding one new variant, "M4: learned team"
  (`learned_model=True`), scored against M2c.
- **Reported:** the held-out five's paired differences M4 − M2c under raw, merged2s and merged3s,
  with and without tags; D precision / recall, stoppages per match and possessions per minute; the
  per-match spellF1.
- **Pass:** on the held-out five, the paired difference is above zero in all six cells (three truths,
  two tag modes), and its 90% interval excludes zero under raw and under merged3s, in both tag modes.
  Then `learned_model` goes on by default. Otherwise it stays off. Nothing is re-tuned afterwards, and
  a negative result is reported as it comes.

**The shipped model.** Trained once before the run: switch cost 20, minimum possession 50 frames
(2 s), 100 trees; a rerun of `train_model.py` writes a byte-identical `model.json`. The tuning surface
and what each number means are in `config.py`.

**Held-out check** (the one run, 2026-10-27; `solution/devset/README.md` has every table). Paired
differences in spellF1 against M2c on the five held-out matches, 5 seeds, 90% intervals:

| | tags | raw | merged2s | merged3s |
|---|---|---|---|---|
| M4 − M2c | tags | **+0.070** [+0.057, +0.082] | +0.101 [+0.088, +0.115] | **+0.112** [+0.096, +0.129] |
| M4 − M2c | no tags | **+0.071** [+0.058, +0.082] | +0.103 [+0.091, +0.116] | **+0.115** [+0.099, +0.132] |

- **M4 passes the pre-registered test, so `learned_model` is on by default.** All six cells are
  above zero, every interval excludes zero, and every bootstrap draw is above zero.
- **It holds on every held-out match**, raw and merged 3 s, with tags and without. With tags, raw /
  merged3s: J03WN1 0.292 / 0.377 → 0.368 / 0.504; J03WOH 0.350 / 0.419 → 0.404 / 0.546; J03WOY
  0.316 / 0.396 → 0.395 / 0.507; J03WQQ 0.378 / 0.477 → 0.455 / 0.574; J03WR9 0.397 / 0.491 →
  0.459 / 0.588.
- **The held-out gain is larger than the gate's** (+0.062 / +0.095 with one training match), as a
  model trained on two matches should be. Nothing points to having fitted the calibration matches.
- **Held-out spellF1 with tags:** 0.344 / 0.401 / 0.430 (raw / merged2s / merged3s) for M2c, 0.414 /
  0.502 / 0.542 for M4. macroF1 0.767 → 0.801. smooth50 is at 0.233 / 0.243 / 0.233.
- **Dead balls are unchanged** (D precision / recall 0.95 / 0.77, 75.5 stoppages per match), as they
  must be. Possessions per minute rise from 2.35 to 2.54, against a truth of 4.49 raw and 2.91 merged
  at 3 s.
- **M2c's rows reproduce the M2c run to the third decimal**, so the comparison is against the
  pipeline as it was.

**Does the real match look like the training data?** `python3 -I solution/experiments/learned_drift.py
--dfl-dir <dfl>` compares every feature on the real match's play frames (97,523) with the degraded
calibration matches' (seeds 0 and 1, 364,541), by quantiles and by the population stability index
(PSI) against DFL's deciles. No labels are involved. 55 of the 63 features have a PSI under 0.10.
The other eight:

| feature | splits in the model (of 1,400) | real q5 / q50 / q95 | DFL q5 / q50 / q95 | PSI |
|---|---|---|---|---|
| spread along the pitch, home (Barcelona) | 58 | 8.4 / 15.8 / 24.1 m | 9.4 / 14.7 / 19.9 m | **0.27** |
| spread along the pitch, away (Athletic) | 65 | 7.2 / 13.2 / 23.6 m | 7.7 / 15.0 / 20.6 m | **0.29** |
| players tracked, home | 5 | 9 / 11 / 12 | 9 / 10 / 11 | **0.40** |
| players tracked, away | 6 | 9 / 11 / 11 | 9 / 10 / 11 | 0.23 |
| until the next contact, home | 9 | log(1 + s): 0.00 / 1.20 / 4.11 | 0.00 / 1.95 / 4.11 | 0.10 |
| until the next controlled touch, away | 23 | 0.00 / 2.70 / 4.11 | 0.00 / 2.80 / 4.11 | 0.12 |
| since / until the last / next frame near the ball, away | 28 / 9 | 0.00 / 1.50 / 3.64 | 0.00 / 1.70 / 4.08 | 0.11 / 0.13 |

- **The drift is mostly the match-up and the ball.** Barcelona stretch the pitch further than the
  DFL sides, and Athletic sit more compactly. The since / until features are shorter on the real
  match because its ball is seen on more play frames (83%, against 76–80% on the degraded DFL). The
  real match also tracks an eleventh player per team more often: the degradation matches the tracked
  count over all graded frames (10.2 per team), not over play alone.
- **The team-shape features it drifts on are among those the model splits on most.** The worry is a
  model that hands the ball to whichever team is spread out. It does not: Barcelona's share of play
  frames is 60.4% with M2c and 60.1% with M4.
- **The ball–player timing features hardly drift** (distance, relative speed, the near and control
  windows, all PSI ≤ 0.07). But no marginal can show the 15–30-frame lag of the real ball, identity
  switches inside a slot or the stuck centre-spot ball, which the dev set does not reproduce.

**On the real match** (`python solution/run.py`, then `python solution/evaluate.py
evaluation/results/submission.csv --bootstrap 1000`; `--rules-only` writes M2c). A sanity check, not
the deciding number:

| version | public spellF1 [90% CI] | macroF1 | D F1 (public) | public possessions (true 17) | full match: possessions, per min, under 1 s, under 2 s, D, Barcelona's share of play |
|---|---|---|---|---|---|
| M2c | 0.566 [0.37, 0.74] | 0.901 | 0.897 | 12 | 248, 2.34, 9, 31, 38.6%, 60.4% |
| **M4** | **0.622 [0.39, 0.81]** | **0.906** | 0.897 | 10 | 214, 2.02, 5, 10, 38.6%, 60.1% |

- **On the public window M4 changes one spell and four edges.**
  - It removes the false Athletic spell at 12438–12485: the ball received under pressure and lost
    straight back, which M3b found the rules could not separate.
  - The Athletic spell at 13704 runs 13705–13795 instead of 13692–13851, against the truth's
    13704–13785 (the stuck-ball case): both edges move towards the truth.
  - Two edges move away from it. The Athletic spell at 12665 starts at 12713 instead of 12647, and
    the Barcelona spell after 13785 ends at 14882 instead of 14844 (truth 14846).
  - The stoppages are untouched. The gain is the removed spell and the 13704 edges; with 17
    possessions the interval is as wide as before.
- **Possessions per minute fall from 2.34 to 2.02** on the full match, away from the public truth's
  3.4, while on the dev set M4 raised them. Spell lengths per minute:

  | | M2c: under 2 s / 2–5 s / 5 s or more | M4 |
  |---|---|---|
  | real match | 0.29 / 0.42 / 1.63 | 0.09 / 0.32 / 1.61 |
  | J03WMX, seed 0 | 0.30 / 0.47 / 1.58 | 0.15 / 0.40 / 1.78 |
  | J03WPY, seed 0 | 0.15 / 0.35 / 1.65 | 0.09 / 0.39 / 1.88 |
  | public truth (5 min) | 0.40 / 0.80 / 2.20 | |

  On both kinds of match M4 removes most of M2c's spells under 2 s, which the 2 s merged truth says
  are mostly touches lost straight back. On DFL it also finds new long possessions; on the real
  match it finds none. The model is as confident there as on DFL (median |log-odds| 3.0 against
  2.9–3.0; it agrees with M2c on 86% of play frames against 86–87%). So the part of the gain that
  comes from finding changes M2c misses may not carry over. The part that comes from removing false
  spells does, as far as the public window shows.

**Honest caveats.**
- **Two training matches.** Both are Bundesliga, and the held-out five are 2. Bundesliga, four of
  them Düsseldorf's. Nothing in them plays like Barcelona, which is what the drift table shows.
- **The dev set's ball is better than the real one** in ways the drift check cannot see: no lag, no
  stuck ball, no identity switches inside a slot. M4 reads timing between ball and players more
  finely than any rule, so it is more exposed to these than M2c is.
- **The DFL flag times a change by its own convention**, and M4 learned it. The public truth stamps a
  change 0–8 frames after the decisive touch (M3). On the public window M4's edges are as close as
  M2c's or closer, but that is four edges.
- **Fewer possessions on the real match** (above). A pipeline that under-counts possessions loses
  spellF1 on recall, and M4 widens a gap M2c already had (2.34 against the public truth's 3.4).
- **The model replaces the rules' team on every frame of play**, so M3's boundary placement, the
  duel rule and the keeper split now act only through the features they feed it (`owner`, the
  touches). Their switches in `adapter.PIPELINES` measure that, from the next run on.
- **`run.py` now reads `possession/model.json`** (105 KB, committed). It is derived from the DFL data
  alone, so it carries the CC BY credit; `run.py` needs neither scikit-learn nor the DFL files, and
  `train_model.py` rebuilds the file exactly.

**What comes next, by size.**
1. **Stoppages we miss**, mostly short fouls (M2c ceilings: every truth stoppage with its team is
   worth +0.05 merged3s). M4 does not touch them, and every missed stoppage still merges two
   possessions.
2. **The ball defects the dev set lacks.** The real ball lags, swaps identity and sticks in ways the
   degradation does not reproduce, and M4 reads ball–player timing finely. These were seen by
   watching public-window frames, so no number from there may go into the degradation (RULES.md).
   Only a measurement made by an algorithm on the whole match, or on other games, could.

(This list was rewritten on 2026-10-28. Its first version measured the ball's lag on the real
match's video and looked to the switch cost for the possession count; both are withdrawn, see
"Re-checked on DFL".)

### Re-checked on DFL: what the public window steered (2026-10-28)

**Why.** On 2026-10-23 the organisers tightened the rules (commit `f8f2629` on `origin/main`). The
public 5 minutes are "not training data and not a tuning set" (RULES.md, rule 2). Watching the video
is fine, but "turning what you saw into labels, corrections, or rules and constants tied to specific
frames of the game is" not (rule 1). M1–M3b were built before the dev set existed, and the public
window steered some of their choices. This section re-measures what it steered, on the two DFL
calibration matches alone. The held-out five and the public window are not read.

**Decision rules, written before anything below was measured.**

1. **`KICK_REST_FRAMES` (2 s).** Candidates 0.5, 1, 2 and 3 s.
   - Measured: (A) kick-like events in open play on the degraded calibration matches (seeds 0–1, the
     repaired ball as the pipeline sees it): kicks on a frame the truth marks alive, not within 1 s
     of a true restart, per match. (B) The share of the calibration dead balls (DFL's real
     `BallStatus` runs) shorter than the rest: a restart after a shorter dead ball cannot have
     rested that long.
   - DFL freezes its ball in dead time, and the dev set invents it, resting exactly 2 s before every
     restart (`degrade.REST`). So DFL bounds how long a real ball rests before a restart only from
     above, by the dead ball's length. It can argue for a shorter rest, never for a longer one.
   - 2 s stays if (i) at most 5% of calibration dead balls are shorter than 2 s, and (ii) 2 s gives
     at least a quarter fewer open-play kick-like events than 1 s. If (i) fails: the longest shorter
     candidate with at most 5% shorter. If (ii) fails: 1 s, and the same test between 0.5 s and 1 s.
     3 s is measured and reported, and cannot be chosen.
   - Reported, deciding nothing: where our stoppages end against the true restart with each rest.
     The invented ball sets how long the restart itself rests.
2. **`RESTART_BACK` (8 s).** Candidates 4, 6, 8, 10, 12 and 15 s.
   - Measured on the degraded calibration matches (seeds 0–1, tags): for each truth dead ball that
     one of our stoppages overlaps, d = the end of the latest slowdown candidate overlapping it minus
     the true restart (DFL's first live frame). The backward search has to reach the restarts with
     d ≥ 0.
   - A window too long costs little: the latest kick wins, and an earlier one is taken only when
     there is no later one, which ends the dead ball early, the cheap side. A window too short
     loses restarts.
   - 8 s stays if it reaches at least 90% of the restarts with d ≥ 0. Otherwise the smallest
     candidate that does. It is not shortened.
3. **`OOB_BEFORE` (15 s).** Candidates 10, 15, 20, 25 and 30 s.
   - Measured on the degraded calibration matches (seeds 0–1): (a) coverage: the truth dead balls
     whose start has a line crossing of the repaired ball from 1 s before to 2 s after it, and a
     slowdown candidate overlapping them; the share whose first such candidate starts at most W after
     the crossing. (b) False confirmations: candidates lying wholly in open play by the truth with a
     crossing from W before them to their end, per match.
   - 15 s stays if its coverage is at least 90% and it confirms at most one open-play candidate per
     match. Otherwise the smallest candidate that meets both; if none does, the one with the highest
     coverage among those that confirm at most one per match.
4. **`MIN_DEAD_FRAMES` (3 s).** 3 s stays if at most 5% of calibration dead balls are shorter than
   3 s. Otherwise the longest of 1 s and 2 s with at most 5% shorter.
5. **M4's target: spells next to a dead ball.** The merged truths absorb a spell under 2 s (3 s) that
   lies between two spells of the other team, so a spell with a dead ball on either side is never
   absorbed.
   - *After* a dead ball, the challenge's definition settles it: "Whenever play restarts after a
     stoppage, there is a clear possessor" (CHALLENGE.md). Not tested.
   - *Before* one, it does not: "a simple touch of the ball without control" does not change
     possession, but nothing says whether the last touch before the ball goes out is one. The only
     reason given so far was a public-window case (`truth.py`, "17293"). So it is tested by leaving
     one calibration match out, as `learned_gate.py` does: trees, two variants that differ only in
     the training target, the 2 s truth as now and the 2 s truth with each such spell absorbed into
     the other team's. Segmentation tuning and scoring truths as in the gate. Scored under raw and
     under the 3 s truth built both ways, with and without tags.
   - The variant replaces the target only if its paired difference against the current one is above
     zero in all six cells (point estimates). Otherwise the target stays. If it does replace it, only
     the training target changes: the scoring truths stay as `truth.VARIANTS` builds them,
     `model.json` is retrained and the leave-one-out check against M2c is rerun.
6. **`SLOW_LAG` (0).** Not re-measured. M2b's DFL measurement is the reason: 17 of the 58
   slowdown-confirmed calibration stoppages start before the true start, by up to 8.6 s. Since M2c,
   `whistle`, set on calibration, decides where a stoppage with no crossing starts; `SLOW_LAG` only
   anchors `out_since`'s search.

If any value moves, M4 is retrained on the calibration matches, the leave-one-out check confirms it
still beats M2c there (not on the held-out five), and the submission is regenerated with a new hash.

**Results** (one run each, 2026-10-28, calibration only: `python3 -I
solution/experiments/dead_ball_windows.py --dfl-dir <dfl>` and `python3 -I
solution/experiments/target_dead_ball.py --dfl-dir <dfl>`).

1. **`KICK_REST_FRAMES` moves from 2 s to 1 s.**

   | rest | kick-like events in open play, per match | at a true restart, per match | true restarts with one within 1 s |
   |---|---|---|---|
   | 0.5 s | 88.0 | 30.0 | 30.6% |
   | 1 s | 50.0 | 28.2 | 28.8% |
   | 2 s | 39.5 | 24.8 | 25.3% |
   | 3 s | 34.8 | 23.0 | 23.5% |

   - (i) holds: none of the 196 calibration dead balls lasts under 2 s (3 under 3 s, 10 under 4 s;
     median 18.2 s).
   - (ii) fails: 2 s gives 21% fewer open-play events than 1 s, short of a quarter. 1 s gives 43%
     fewer than 0.5 s, so the rule stops at 1 s.
   - Reported, deciding nothing: our stoppages end within 1 s of the true restart on 60.8% of 296
     pairs with 1 s of rest and 60.0% of 295 with 2 s, and more than 1 s late on 25.3% and 26.1%.
2. **`RESTART_BACK` stays at 8 s.** Our stoppages overlap 316 truth dead balls. For 211 the restart
   comes before the slowdown over it ends, a median 1.0 s before (q90 2.5 s); for 96 after it; 9 have
   no slowdown over them. 4 s back reaches 93.8% of the 211, 6 s 97.6%, 8 s 98.1%, 12 s 99.1%.
3. **`OOB_BEFORE` stays at 15 s.** 212 truth dead balls start with a crossing, 180 of them with a
   slowdown over them, which starts a median 3.0 s after the crossing (q90 11.1 s).

   | window | crossing dead balls reached | open-play candidates confirmed, per match |
   |---|---|---|
   | 10 s | 85.0% | 0.8 |
   | 15 s | 96.1% | 0.8 |
   | 20 s | 97.2% | 1.2 |
   | 25 s | 99.4% | 1.8 |
   | 30 s | 100% | 2.0 |

4. **`MIN_DEAD_FRAMES` stays at 3 s:** 1.5% of calibration dead balls are shorter.
5. **M4's target stays.** Absorbing a short spell before a dead ball changes 43 and 35 spells at 2 s
   (1,039 and 876 frames). Paired against the current target, with tags: −0.004 raw, −0.004 under the
   3 s truth, −0.001 under the 3 s truth built the other way; without tags −0.003, −0.003, −0.001. So
   a spell next to a dead ball stays unabsorbed, and `truth.py` now gives this result as its reason.
6. **`SLOW_LAG`:** no measurement, as written above.

**What changed.** One value: `KICK_REST_FRAMES`, 50 frames to 25.
- **M4 retrained** (`train_model.py`), with the same settings (switch cost 20, minimum 50 frames).
  `model.json` was then SHA-256 `ba6ec802…`, and a rerun rebuilt it byte for byte. Renaming the
  folder to `solution/` (2026-10-29) changed only the path in its credit line, and it is now
  `6ba3f99919e01a21d18d0d692366eee1eb05420efc691dead137bef3e92c4ef7`, again rebuilt byte for byte. Retrained without the video rules on 2026-11-12, it is
  `08f745837633dcaa33af04560da18062dba243eff56238771776be5ee3c00bb8`.
- **The leave-one-out gate, rerun on calibration** with the new rules (`learned_gate.py --variants
  "trees, merged2s"`): M4 − M2c is +0.057 [+0.034, +0.082] raw and +0.101 [+0.081, +0.117] under the
  3 s truth with tags, +0.058 and +0.104 without; mean +0.080, as before the change. It passes the
  +0.010 gate. M2c's calibration row is unchanged to the third decimal (0.342 / 0.441 / 0.458).
- **The held-out five were not re-scored.** Their numbers, here and in `solution/devset/README.md`, are
  for the pipeline with the 2 s rest.
- **The submission** is now SHA-256 `c77d0ae0bcf5556bf0aca67b82e930b938230652da6f0fe6c6e3504fd5cbffb1`.
  1,195 of its 158,857 graded rows differ from `534bf8d0…`: 966 change team and 229 become `D`.
  `--rules-only` writes `cf791a5e4ec33f3388507b77727cf2d10bbcc2b16b27d3300ec5d83e0960f41b`.
- **On the real match**, as a sanity check only: public spellF1 0.6222 before and after, macroF1
  0.906 to 0.902; over the full match, 214 to 222 possessions (2.02 to 2.10 per minute), `D` 38.6% to
  38.8%.

**Statements corrected in this record.** M4's target was described as "the closest of the three to
the public truth's 3.4" possessions per minute; the gate chose it, and that sentence now says so.
The M2 and M2b sections now say which of their public-window reasons no longer stand.
`config.py`, `truth.py`, `TECHNICAL.md` and both READMEs say what justifies each value now.

**Withdrawn plans.** Four next steps written earlier in this record would turn public-window
observations into constants. They are withdrawn. M4's "What comes next" and the build order are
rewritten without them. M3's and M3b's older lists keep them as written, as the record of what was
planned then.
1. The real ball's "15–30-frame lag", read off the video at a public-window tackle, as a degradation
   parameter (M4, "What comes next").
2. A restart lag swept over 0 / 12 / 25 frames from the public window's 11–28-frame restarts (M3's
   and M3b's "Restart timing", and the dev-set audit).
3. A stuck-ball repair shaped on frames 13731–13810 (M3, M3b, "Ball repair").
4. A switch cost moved towards the public truth's 3.4 possessions per minute (M4, "Possessions per
   minute on the real match").

### Withdrawn: the rules from the video (2026-11-12)

**Why.** The organisers sent a clarification of the rules that takes precedence over everything
before it. It does not allow "checking your output against the video", nor training, fitting or
calibrating on the ground truth or the video; parameters "MUST" be tuned on external data. M3 and
M3b did check our output against the video: the M3 boundary cases and the M3b tables above compare
our spells with what the video shows at public-window frames. Three rules came out of that work:
- the decisive-touch placement (`boundary.place_changes`, M3);
- the keeper split (`keepers.split`, M3b);
- the duel rule (`contest.resolve`, M3b).

Their thresholds were set on whole-match distributions, never on a frame, but the rules exist
because of that comparison. On the held-out five each was worth +0.000 to +0.004 (M4, above). They
are withdrawn: `pipeline.label_frames` runs without them by default, and they stay only as switches
for the experiments that measured them. Everything else in the pipeline was set on DFL, or on this
match's tracking and tags without labels, which the clarification allows.

**What runs, written before anything below was measured.**
1. M4 is retrained without the three rules (`train_model.py`, same settings and seeds).
2. **Gate**, as M4's: leave one calibration match out, both ways, 5 seeds, paired against the rules
   alone (M2c without the three rules). The retrained M4 ships if it gains at least +0.010 under raw
   and merged3s, with and without tags. Otherwise the rules alone ship (`--rules-only`).
3. **The held-out five**, scored once, as a report: the retrained M4, the rules alone, and the
   pipeline before the withdrawal. The decision is made in steps 1–2, so nothing here can change
   it; this is the five's fourth use.
4. The submission is regenerated by a person, not here: the clarification does not allow the
   provided data to go to an AI service.

**Results** (2026-11-12; `train_model.py`, `learned_gate.py --variants "trees, merged2s"` and
`run_devset.py` with one extra row for the model before the withdrawal).
1. **Retrained.** `model.json` is now SHA-256
   `08f745837633dcaa33af04560da18062dba243eff56238771776be5ee3c00bb8`, rebuilt byte for byte by a
   second run. The segmentation kept switch cost 20 and a 50-frame minimum.
2. **Gate: GO.** M4 − the rules alone, leave one calibration match out: +0.069 [+0.044, +0.099] raw
   and +0.102 [+0.080, +0.124] merged 3 s with tags; +0.070 and +0.107 without. Team on play frames
   against the merged 3 s truth: 90.0% with the ball seen (rules 87.0%), 86.8% unseen (rules 72.7%).
3. **The held-out five, once** (tags): M4 0.416 [0.385, 0.448] raw / 0.542 [0.511, 0.576] merged 3 s;
   the rules alone 0.339 / 0.422; M4 before the withdrawal 0.415 / 0.543. M4 − the rules alone:
   +0.077 [+0.065, +0.088] / +0.120 [+0.104, +0.137]. The withdrawal: −0.001 [−0.004, +0.002] /
   +0.001 [−0.002, +0.004]. Inside M4, M2c's restart team by the laws of the game is now worth
   +0.000 / +0.001, its earlier starts +0.007 / +0.010, the foul test +0.037 / +0.046.
4. **Not done here:** the regenerated submission and its hash.

---

## 5. Build order

| milestone | content | status |
|---|---|---|
| M0 | `solution/` package: loader + cache, features, evaluation harness, tests on synthetic data | done |
| M1 | last controlled touch, hold previous team through ball loss, known pre-kickoff `D` | done, see section 4 |
| **M2** | dead-ball detector + restart team | done, see section 4. Gap: fouls with no tag or crossing, partly closed by M2b |
| M3 | boundary timing: what "owning" means, changes moved to the decisive touch | done, see section 4. Small gain: the timing premise did not hold. **Withdrawn 2026-11-12** (came from comparing our output with the video) |
| M3b | contested phases: keeper slots split by half, duels the other team takes straight back void | done, see section 4. The rest of the contested-phase error is not separable with this tracking, so no semi-Markov model yet. **Withdrawn 2026-11-12** (same reason) |
| M2b | foul stoppages: a slowdown deeper or longer than open play is a stoppage on its own; stoppages with no restart joined | done, see section 4. Thresholds measured on two labelled DFL matches; short, shallow fouls (12851) still missed |
| M2c | stoppage starts (an earlier crossing the ball never came back from; where the slowdown set in) and restart team by the laws of the game (thrower on the touchline, attackers in the penalty area) | done, see section 4. +0.014 raw / +0.019 merged3s spellF1 on the five held-out DFL matches, every interval above zero; public window unchanged |
| **M4** | the team in open play: a tree ensemble trained on the DFL calibration matches weighs every rule's evidence and team shape per frame; a two-state Viterbi between the stoppages decodes possessions | done, see section 4. +0.070 raw / +0.112 merged3s spellF1 on the five held-out DFL matches, every interval above zero; public window 0.566 to 0.622, but full-match possessions per minute fall from 2.34 to 2.02 |
| re-check | what the public window steered, re-measured on the DFL calibration matches by rules written beforehand (organisers' rule change, `f8f2629`) | done, see section 4, "Re-checked on DFL". The restart rest moves from 2 s to 1 s, M4 retrained; the restart search, crossing window, minimum stoppage and M4's target stay |
| withdrawal | the three rules from comparing our output with the video (M3's decisive touch, M3b's duel rule and keeper split) switched off; M4 retrained | done, see section 4, "Withdrawn: the rules from the video". Held-out change −0.001 to +0.002, intervals across zero |
| **next** | short fouls; the ball defects the dev set lacks, measured only by an algorithm | not started (section 4, M4) |
| M5 | ball detection from the tactical video | if there is time. The only route to the short possessions the tracked ball does not show |

M1–M3 are mostly sequential. With 3–5 people, M4 and the dev-set harness can run in parallel from
the start.

---

## 6. Reproducibility

- One command regenerates the submission: `python solution/run.py`.
- Every tunable number lives in `solution/possession/config.py`.
- Deterministic: fixed seeds, no GPU nondeterminism in the final run, pinned dependencies in
  `requirements.txt`.
- **M4's trees compare time features as whole frame counts** (`learned.tree_scores`).
  - The 20 time features are log(1 + frames / 25), and 406 of the model's 429 thresholds on them sat
    exactly on such a value (395 of 415 in the model retrained on 2026-10-28).
  - With a float comparison, the last bit of log1p decided which side those frames fell on: nudging
    every log1p result up one float step changed 8 rows of the submission.
  - Both sides are now turned back into integer frame counts first; a threshold standing on a count
    keeps that count on the left, as before. Nudging by one or four float steps, up or down, now
    changes nothing; the retrained model was probed again the same way, with the same result.
  - `model.json` and the submission were byte for byte unchanged by that fix.
- No data, cache or submission file is ever committed.
