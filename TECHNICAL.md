# Ball possession from tracking: rules for dead balls, a learned team in open play

More than a Hack 2027, Phase 1 — technical document.

| | |
|---|---|
| Team | Tiki Taka Masala |
| Repository | <https://github.com/aaryan2811/tiki-taka-masala> — the pipeline is `solution/`, the command `python solution/run.py` |
| Pipeline | M4 without the rules that came from the video (2026-11-12), as published in the repository |
| Submission file | 199,319 rows; SHA-256 `8e93ae6ffd0967dcd06fa622c0ca4bac38fe42d25f401eb71e053cbbbae03b41` |

**In short.**
- **Dead balls and restarts** come from hand-written rules, built for precision because a false dead
  ball costs a whole possession.
- **The team in open play** comes from gradient-boosted trees trained on two open DFL matches. They
  give, for every frame, how likely each team is to have the ball. A two-state Viterbi path between
  the stoppages turns that into possessions, paying a cost for every change.
- **Validation.** On five other open DFL matches that shaped nothing in it, degraded to look like
  this match's tracking, the pipeline scores spellF1 **0.416** [0.385, 0.448]. The rules alone score
  0.339 and the organisers' `baseline_smooth50` 0.233. Against a truth that absorbs spells under 3 s
  lost straight back, the three score 0.542, 0.422 and 0.233.
- **Every parameter is set on external data** (the DFL matches), or read without labels from this
  match's tracking and tags. Nothing is trained, fitted or calibrated on the ground truth or the
  video (section 4).
- **A clean checkout** regenerates the file byte for byte (section 6).

Brackets are 90% bootstrap intervals. Every number comes from `solution/DESIGN.md` (the full
engineering record, section 4 by milestone), from `solution/devset/README.md` (the development set),
or from a command in section 6. CHALLENGE.md, RULES.md and FAQ.md are the organisers' documents, as
of their commit `f8f2629`.

---

## 1. The approach, and why

**What the metric punishes.** We damaged the public ground truth in controlled ways and scored the
result (`solution/experiments/metric_sensitivity.py`). This studies the metric only: no prediction of
ours is in it, and it sets no value.

| error put into a perfect answer | spellF1 |
|---|---|
| every possession change 5 / 12 / 25 frames late | 0.976 / 0.906 / 0.776 |
| possessions under 2 s merged into the previous one | 0.827 |
| no dead balls at all (previous team carried through) | 0.812 |
| one false 1 s dead ball inside every long possession | **0.644** |

Four design rules follow:
1. **A false dead ball costs more than a missed one**, so dead balls are detected for precision.
2. **Decide segments, then place boundaries.** Window smoothing moves every boundary, which is why
   the baseline cannot get far.
3. **A missing ball is not a dead ball.** Keep the previous team.
4. **A missed short possession costs less than an invented one.** Credit a change only on strong
   evidence.

**The pipeline** (`solution/possession/`, composed in `pipeline.label_frames`):

| stage | module | what it does |
|---|---|---|
| load | `atd.py` | parse the raw ATD into arrays in metres; cache them (gitignored) |
| ball | `ball.py` | drop spikes and runs under 3 frames; fill gaps up to 1 s with a line |
| evidence | `features.py`, `control.py` | nearest player per team, controlled touches, team shape |
| dead balls | `deadball.py` | slowdowns confirmed by a crossing, a tag or a slowdown beyond open play; start, restart kick, restart team |
| the rules' team | `control.py` | the last controlled touch's team (M2c; `--rules-only` writes it) |
| learned team (M4) | `learned.py`, `segment.py`, `model.json` | per-frame log-odds from 63 features; a Viterbi path between the stoppages |
| out | `labels.py` | `H`/`A`/`D` on graded frames, `X` outside them; validated |

Every rule threshold is in `config.py`, with its measurement next to it. M4's parameters are in
`model.json`, and its training settings in `solution/devset/train_model.py`.

**Rules for dead balls, a model for the team.**
- **Dead balls stay rules.** One false dead ball costs a whole possession, and a rule's failure mode
  can be read off and bounded.
- **The team in open play is learned, because the rules had run out of separate evidence.** Pass
  chains, a team-shape classifier on its own and looser touch thresholds all failed to beat them
  (section 2). What was left was to weigh all the evidence together, which rules cannot do. Trees
  gain about 60% more than a linear model on the same features (section 4), so the combinations
  matter.
- **There are no labels on this match, and none were made.** The model is trained on DFL matches,
  and it is symmetric between the teams by construction.

---

## 2. Assets, training data, and what we dropped

| asset | used? | how |
|---|---|---|
| **Tactical ATD** (`_atd.txt`) | the main input | Ball and 46 track slots per frame; the ball is seen on 56.9% of frames. Ball speed is derived from positions, because the ball's own speed field peaks at 7.3 m/s. Player speed is in m/s despite the name `speed_kph`. |
| **ATD metadata** (`_atd.xml`) | yes | Each slot's team, the goalkeeper slots, the attack direction in the first half. |
| **Smart Tagging** | set pieces only | A `SET PIECES` tag near a slowdown confirms a dead ball; its team is the last fallback for the restart team. 59 set pieces, only 5 of them free kicks; throw-ins carry no team. Phase tags are not used: a phase is not a possession. |
| **Public ground truth** (5 minutes) | automatic checks | Frame alignment, format, what the labels mean. No model is trained, tuned or validated on it. Before the DFL dev set existed it steered some early choices; every value it steered is now set on DFL (section 4). |
| **Tactical video and homography** | not in the pipeline | Inspected on public-window frames to understand what counts as possession. Early on we also compared our output with it; the three rules that came out of that are withdrawn (section 4). No label, correction, rule or constant comes from it. |
| **Broadcast source** | no | It can be aligned to the tactical timeline, but no labelled data could test what it adds. |
| **DFL / Sportec open data** (7 matches, CC BY 4.0) | training, calibration, validation | Per-frame truth for other matches. Two train M4 and set the DFL-measured thresholds; five validate. All of it runs on tracking degraded to look like ours. `run.py` reads only the exported `model.json`. |
| Organisers' code | yes | `games.py`, the `starter/` loaders, `evaluation/possession_metrics.py` (never reimplemented). |

### Training data

We hand-labelled nothing, on this game or any other, and used no human-produced data about this
game. Every labelled dataset is about other games, and its labels were made by the DFL.

| dataset | labels | who produced them, and how | how much | used for |
|---|---|---|---|---|
| DFL / Sportec open data (Bassek et al., 2025), positions | `BallStatus` (in or out of play) and `BallPossession` (home or away), every frame | The DFL's official match data. Positions are from TRACAB Gen5 optical tracking (16–20 cameras, 25 Hz). The dataset article defines possession as "defined by the player in control of the ball", and out of play as "when the match is interrupted by the referee". It does not say whether a person or the system sets the flags, and we could not verify which. | 7 matches, 1,002,644 frames (668 min). Calibration: J03WMX and J03WPY, 292,178 frames. Held-out: J03WN1, J03WOH, J03WOY, J03WQQ, J03WR9, 710,466 frames. | training M4 and tuning its segmentation; the DFL-measured rule values; the degradation's matched widths (calibration only); validation |
| The same dataset, events | set-piece events with time and team | Hand-collected by the DFL's trained operators (five per match, live, checked by a supervisor and after the match) | set pieces only: 195 on calibration, 496 on the held-out five | synthetic Smart Tagging on the dev set |

**M4's training set** (`solution/devset/train_model.py`):
- **Data:** the two calibration matches, each degraded with four seeds (10–13), never the scoring
  seeds (0–4).
- **Rows:** every 5th frame of play where the target names a team. That is 128,916 rows, each also
  used with the teams swapped.
- **Target:** DFL's possession flag with spells under 2 s absorbed when they sit between two spells
  of the other team. The raw flag credits a touch lost straight back; the challenge does not ("a
  simple touch of the ball without control", CHALLENGE.md).
  - The 2 s length was chosen over 3 s by the pre-registered gate (section 4).
  - A short spell next to a dead ball is kept. A leave-one-out test on the calibration matches
    favoured keeping it in all six cells.

This game enters the training without any label. The degradation is fitted to aggregates of its
tracking (`solution/devset/params.json`), and the synthetic tags copy its set-piece counts.
The clarification allows any generated data as input, as long as nothing is trained on the ground
truth or the video. The held-out five trained and tuned nothing.

### Tried and dropped

Measured on the DFL calibration matches unless stated. There, the rules alone (M2c) score 0.346 raw
and 0.468 merged at 3 s.

| idea | what we measured | outcome |
|---|---|---|
| Nearest player = possession | a pass rolling past an opponent counts as their touch: 6.2 possessions per minute on this match, against 2.5 once the ball must move with the player | dropped; the shipped 2 m, 3 m/s, 5 frames is the best of 18 settings on DFL |
| Window smoothing (the baseline) | moves every boundary; writes `D` through occlusions | the floor we measure against |
| Restart team by the last touch | before a touchline crossing, right 48 times and wrong 54 | replaced by where the players stand |
| Crediting uncontrolled contacts | 330 by the team out of possession over this match, with nothing to tell the real ones apart | not shipped |
| Pass chains (two contacts by one team within 3 s) | where they overrule the rules' team, the truth sides with them 37% of the time; −0.014 merged | dropped |
| A team-shape classifier alone, for unseen-ball frames | 81–82% of frames right against the rules' 75%, but spellF1 does not rise | dropped; team shape went into M4 |
| Looser touch thresholds | raw rises to 0.383, merged 3 s falls | kept the shipped values |
| A looser foul test (0.70 m/s or 20 s), named beforehand | held-out: 2 false stoppages and 168 caught, against 3 and 199 shipped | kept the shipped setting |
| A ball-based restart rule (long unseen rest) | cannot be checked: DFL freezes its dead-ball ball | not shipped |
| M4 alternatives in its gate: logistic regression, a 3 s target, no minimum length | mean gains +0.049, +0.066, +0.079, against +0.080 for the trees | the trees |
| The rules from comparing our output with the video: changes moved to the decisive touch, duels taken straight back void, keeper slots read by the half they stand in | held-out: M4 with them minus M4 without, −0.001 to +0.002 spellF1, every interval across zero | withdrawn (section 4) |

---

## 3. The possession rules

The rules decide dead balls and restart teams directly. In open play they produce the evidence M4
weighs; `--rules-only` writes their own team instead.

**Who has the ball.** A team is on the ball when its nearest player is within 2 m, nearer than any
opponent, and the ball moves with that player (at most 3 m/s relative). A **controlled touch** is
5 or more such frames in a row (0.2 s), with the ball seen on each.

**Loose balls and a missing ball.** Possession stays with the last team to control the ball, through
loose balls, deflections and frames with no ball, until the other team makes a controlled touch. A
ball that disappears does not make play dead.

**Challenges, where a change goes, and goalkeeper slots.** The rules put a change on the new team's
controlled touch and read each slot with the team the metadata gives it. Where the line falls inside
a challenge, M4 places it: it learned when a change happens from DFL's labels.

**Dead balls.**
1. **Candidate:** all tracked players' mean speed below 1.4 m/s for at least 3 s.
2. **Confirmed by any one of:**
   - a line crossing up to 15 s before;
   - a set-piece tag nearby;
   - a slowdown beyond open play (below 0.75 m/s, or 15 s long). On DFL, open play never went
     below 0.79 m/s or past 14.5 s on calibration.
3. **Start:**
   - at an earlier crossing the ball never came back from;
   - else at the crossing;
   - else where the slowdown set in.

   No lag is added: an early start eats a real possession.
4. **End:** at the restart kick, else where the slowdown ends.
5. **Clean-up:** dead balls under 3 s are dropped, and a stoppage with no kick is joined to the next
   if it starts within 3 s.
6. **The known span:** frames 110,808–113,741 are dead, as `FAQ.md` publishes. This is the
   pipeline's only frame constant, and it comes from the organisers.

**Restarts.**
- **The kick:** the ball faster than 4 m/s for 3 frames after at least 1 s without, searched from
  8 s before the slowdown ends.
- **The team,** from the first of these that gives one:
  1. the laws of the game, from where players stand:
     - 3 or more attackers in the penalty area after a goal-line crossing means a corner, fewer a
       goal kick;
     - after a touchline crossing, the team of the player standing on that line near where the ball
       went out;
  2. the nearest player at the kick;
  3. the tag's team.

  Otherwise the previous team carries on.

**The team in open play (M4).** Between two stoppages:
1. **Per frame,** the log-odds that the home team has the ball, from 63 features. Each team gets the
   same 27, read from its own end:
   - what the touch rule reads (distance, relative speed, time near and in control);
   - time since and until its touches and contacts;
   - the rules' team and the restart team;
   - team shape (centroid along the attack, spreads, speed, how fast it advances).

   Nine shared features describe the ball's visibility, speed and the stretch of play. Every frame
   is also scored with the teams swapped, and the log-odds are half the difference, so nothing
   learned can be about being at home.
2. **Per stretch,** a two-state Viterbi path pays 20 for each change of team, keeps possessions to
   at least 2 s, and starts with the known restart team. A short spell survives only if its evidence
   outweighs the two changes it needs. The line on loose balls and challenges is therefore set by
   the training target (touches lost back within 2 s are not possessions), the switch cost and the
   minimum length.

---

## 4. Validation

### How we used the public window and the video

The organisers' clarification of the rules takes precedence over everything before it. It allows
automatic checks on the provided data, and inspecting the data and the video "to better understand
what counts as possession". It does not allow checking one's output against the video, or training,
fitting or calibrating on the ground truth or the video; parameters must be tuned on external data.
This is everything we did with the public window and the video:
- **Automatic checks:** frame alignment, the file format, the validator and the scorer.
- **What counts as possession.** We read the window's truth against the video, at public-window
  frames, to understand the labels.
- **Our output against the video.** Early on (M3 and M3b, before the clarification), we also
  compared our own output with the video at public-window frames. Three rules came out of it:
  changes moved to the decisive touch, duels taken straight back made void, and keeper slots read
  by the half they stand in. Their values came from whole-match distributions, not from any frame,
  but the clarification does not allow how they were found, so on 2026-11-12 we withdrew all three
  and retrained M4 without them. On the held-out DFL matches the withdrawal changes spellF1 by
  −0.001 to +0.002, every interval across zero.
- **Where the window steered a choice.** M1–M3b were built before the DFL dev set existed.
  - Four of M2's dead-ball settings were revised after looking at the window's stoppages, and the
    two versions were scored on it: the restart rest, the restart search, the crossing window and
    the start lag.
  - One comment and one rationale for M4's target cited it.
  - On 2026-10-28 we re-measured each of these on the DFL calibration matches, by decision rules
    written down before measuring (DESIGN.md section 4, "Re-checked on DFL"). The restart rest
    moved from 2 s to 1 s; everything else stayed, each value now justified by its DFL measurement.
- **No version is chosen on the window.** Every choice since the dev set existed was made on DFL by
  a test written beforehand.

### The development set

`solution/devset/` scores the pipeline, unchanged, on seven open DFL matches.
- **Truth.** `D` where `BallStatus` is 0, otherwise `H`/`A` from `BallPossession`. DFL credits a
  touch lost straight back, so we also score against truths that absorb such spells under 2 s or
  3 s ("merged"). A change counts only if it holds under all three truths.
- **Degradation.** Each DFL match is degraded to look like this match's tracking:
  - ball smoothing, offset and dropout in six phases;
  - player smoothing;
  - anonymous tracklets;
  - keeper-slot swaps;
  - missing speed;
  - synthetic set-piece tags.

  Each is fitted to aggregates of this match (`params.json`). DFL freezes its ball in dead time, so
  the ball *during* a stoppage is invented, not fitted.
- **Split.** Two calibration matches set every DFL-measured value and train M4. The five held-out
  matches were downloaded afterwards and set nothing.
- **Pre-registration.** M2c's rules and M4 were each written down with an acceptance test before the
  held-out matches were scored for them. Each was scored once.
- **Intervals.** 5 degradation seeds; a two-level bootstrap over matches, then possessions; paired
  differences on the same draws.

### Results on the five held-out matches

The shipped pipeline (M4 without the video rules, 1 s restart rest), with synthetic tags, scored
once on 2026-11-12.

| pipeline | raw | merged 2 s | merged 3 s | `D` precision / recall | possessions per min (truth 4.49 raw, 2.91 merged 3 s) |
|---|---|---|---|---|---|
| **M4 (submitted)** | **0.416** [0.385, 0.448] | **0.504** | **0.542** [0.511, 0.576] | 0.96 / 0.77 | 2.56 |
| M2c: the rules alone | 0.339 [0.308, 0.372] | 0.393 | 0.422 [0.387, 0.463] | 0.96 / 0.77 | 2.42 |
| M2b, before M2c | 0.325 | 0.376 | 0.403 | 0.96 / 0.74 | 2.48 |
| M4 before the withdrawal | 0.415 | 0.503 | 0.543 | 0.95 / 0.77 | 2.54 |
| `baseline_smooth50` | 0.233 [0.216, 0.247] | 0.243 | 0.233 | 0.76 / 0.59 | 5.15 |

Paired differences, with tags (raw / merged 3 s):
- **M4 − the rules alone:** **+0.077** [+0.065, +0.088] / **+0.120** [+0.104, +0.137]. Without
  tags: +0.076 / +0.121.
- **What each dead-ball rule adds inside M4:** the foul test +0.037 / +0.046; M2c's earlier
  stoppage starts +0.007 / +0.010; M2c's restart team by the laws of the game +0.000 / +0.001.
- **The withdrawal** (M4 before it − M4 now): −0.001 [−0.004, +0.002] / +0.001 [−0.002, +0.004].

Earlier, each milestone was pre-registered and passed its test on these five: M2b's foul test,
M2c's rules (+0.014 / +0.019 over M2b) and M4 (+0.070 / +0.112 over M2c). This is their fourth use;
the decision to withdraw was made before it, so it could change nothing.

**M4's gate.** Before any held-out test, M4 must pass a gate on the calibration matches alone: leave
one match out, both ways, paired against the rules alone, 5 seeds, at least +0.010 under raw and
merged 3 s, with and without tags. Retrained without the video rules, it gains +0.069 / +0.102 with
tags and +0.070 / +0.107 without.

**Where the gain comes from.** It is the team, not more possessions. On play frames, against the
merged truth, the team is right:
- with the ball seen, on 90.0% (rules 87.0%);
- with it unseen, on 86.8% (rules 72.7%).

(Leave-one-out on the calibration matches, with tags.)

**Does this match look like the training data?** No labels are needed to compare the features
(`solution/experiments/learned_drift.py`):
- 55 of 63 features have a population stability index under 0.10.
- Barcelona's spread and the number of tracked players drift (0.23–0.40).
- The model did not follow that drift: before the withdrawal, Barcelona's share of play was 60.4%
  under the rules and 59.8% under M4.

---

## 5. Limitations, and what we would fix next

- **Fewer possessions on this match.** Before the withdrawal, M4 found 2.10 per minute on this match,
  against 2.33 for the rules, while on DFL it found more (2.54 against 2.35). On both, it removes the rules' spells under
  2 s, which DFL's truth says are mostly touches lost straight back. On DFL it also finds new long
  possessions, and on this match it finds none. So part of its gain may not carry over, and we have
  no label-free way to measure how much.
- **The dev set's ball is better than the real one.** The real ball lags the play, once sticks on the
  centre spot, and can swap identities inside a slot; the degradation reproduces none of these. M4
  reads ball–player timing more finely than the rules, so it is more exposed. Only a size measured
  by an algorithm could test it, and we have none.
- **Two training matches**, both Bundesliga; nothing in them plays like Barcelona. The model also
  learned DFL's convention for when a change happens: the first touch, close to but not the same as
  the challenge's decisive touch.
- **Short fouls.** We find about 75 of 103 stoppages per DFL match. Short, shallow fouls look exactly
  like lulls in play on every player feature we measured, and every missed stoppage merges two
  possessions. Only a better ball could separate them.
- **The ball** is missing on 41% of graded frames and heavily smoothed: a kick ramps up over about
  20 frames, so restarts are found up to a second early. *Next:* ball detection on the tactical
  video.
- **Stoppage edges.**
  - The dev set cannot measure how the challenge's truth times a restart relative to the kick.
  - Kick-off starts are late, because the ball is carried back to the centre in sight.
  - The no-crossing start is early more often than before (over 1 s early on 28 calibration
    stoppages, against 15), and early is the expensive side.
- **Assumptions about this match.**
  - The 15 s foul test sits just above the longest open-play slowdown on calibration, and the
    held-out matches showed 0.6 false stoppages per match. Barcelona's long spells of passing may
    produce more.
  - The tracker sometimes hands a keeper slot to the other team's keeper. With the keeper split
    withdrawn, only M4's features see this, through where the slot stands.
- **The dev set.**
  - Seven matches from one provider.
  - DFL's possession definition differs, so absolute scores are not comparable with the leaderboard;
    we read differences.
  - The degradation matches marginals only.
  - The held-out five have now been used four times: M2b, M2c, M4 and the withdrawal.

---

## 6. Reproduction

From the repository root, with Python 3.14 and the data bundle (`README.md` has the details):

```bash
python3.14 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, threadpoolctl 3.5.0, pytest 9.0.3
export MTAH_DATA_DIR=/path/to/bundle # the folder holding tactical/
python solution/run.py               # -> evaluation/results/submission.csv, validated
shasum -a 256 evaluation/results/submission.csv
```

- **`run.py`** needs numpy alone. It reads three files of the bundle (`tactical/BAR-ATH_tactical_atd.txt`,
  `_atd.xml`, `_smart_tagging.xml`), `solution/possession/model.json` and its own cache. It reads no
  DFL data. `--rules-only` writes the rules alone.
- **Retraining** `model.json` needs the DFL data and scikit-learn:
  `python3 -I solution/devset/train_model.py --dfl-dir <dfl>` rebuilds it byte for byte (SHA-256
  `08f745837633dcaa33af04560da18062dba243eff56238771776be5ee3c00bb8`).
- **The exact-reproduction check** (2026-10-28, macOS 26.6.2 arm64, Python 3.14.6, before the
  withdrawal) gave the same file in every case:
  - from an empty copy with no cache, cold, warm and with `--refresh`;
  - from another working directory, and under `python -I`;
  - with `PYTHONHASHSEED` 0 and 4242, and with 1 and 8 threads.

  The withdrawal changed no code path that could break this. The file it gives is validated on every
  run; its SHA-256 is in the header, and `--rules-only` gives
  `25f58f566b48171b824dc10a55023343b9cf3cfb8a8612903a874394739a8f5e`.
- **Determinism.**
  - The pipeline has no randomness and no order dependence.
  - The model is read from JSON and evaluated in a fixed order.
  - Its time features are compared as whole frame counts, so the last bit of `log1p` cannot move a
    frame across a split: nudging every `log1p` result by one float step changes 0 rows.
  - The file is written with `\n` line endings on every platform.
  - Only macOS arm64 has been run.
- **Tests:** `pytest solution/tests solution/devset/tests`: 73 pass without the data, and one more
  needs it.

---

## Credits

The development set and M4's model use the DFL / Sportec open data, CC BY 4.0
(<https://creativecommons.org/licenses/by/4.0/>), provided with authorization from the DFL:

> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and
> event data in elite soccer* [Dataset]. figshare. <https://doi.org/10.6084/m9.figshare.28196177.v1>.
> Article: *Scientific Data*, <https://doi.org/10.1038/s41597-025-04505-y>.

We changed it as follows:
- positions are shifted to a pitch-corner origin;
- the ball is replaced during dead balls and degraded;
- players are smoothed and cut into anonymous track slots;
- set-piece events are subsampled into synthetic tags.

`solution/possession/model.json` and `solution/devset/params.json` carry this attribution. `model.json`
holds tree splits on features of the degraded tracking and leaf values learned from the DFL
possession flag. No position or frame of the data is in it. Nothing else of the DFL data, and none of
the competition data, is in the repository.
