# solution/devset/ — a development set from the open DFL matches

The public window has 17 possessions and 4 stoppages. It shows the pipeline is correct, and the rules
make it only that: "not training data and not a tuning set" (RULES.md), so it never chooses between
two versions of the pipeline. This folder scores the pipeline, unchanged, on
open German league matches that carry a per-frame ground truth. The DFL tracking is first degraded
to look like the competition's.

All seven open DFL matches are in:

| group | matches | why |
|---|---|---|
| **calibration** | J03WMX (1. FC Köln – FC Bayern München), J03WPY (Fortuna Düsseldorf – 1. FC Nürnberg) | Downloaded first, for the audit. `params.json`'s matched widths and M2b's slowdown thresholds were set on them, so their scores are biased. |
| **held-out** | J03WN1 (VfL Bochum – Bayer 04 Leverkusen), J03WOH (Düsseldorf – Jahn Regensburg), J03WOY (Düsseldorf – Hansa Rostock), J03WQQ (Düsseldorf – FC St. Pauli), J03WR9 (Düsseldorf – 1. FC Kaiserslautern) | Downloaded on 2026-10-27, after both of those were set, and used to set nothing. **These are the real test.** Keep it that way: tuning anything on them spends them. |

## Run it

```bash
python solution/devset/run_devset.py --dfl-dir /path/to/mtah2027-external/dfl   # ~20 min for 7 matches x 5 seeds
python solution/devset/foul_check.py --dfl-dir /path/to/mtah2027-external/dfl   # M2b's slowdown test, re-checked
python -m pytest solution/devset/tests                                          # synthetic data; no download needed
```

`--dfl-dir` holds one folder per match with its three XML files from figshare. `run_devset.py` does
not need the competition data: the degradation reads only the committed `params.json`. Refitting the
parameters does need it:

```bash
MTAH_DATA_DIR=/path/to/data python solution/devset/fit.py --dfl-dir /path/to/dfl     # rewrites params.json
```

The fit uses only the calibration matches and the stoppages `params.json` was first fitted on: M2's,
with no foul rule, no M2c starts and a 2 s restart rest, pinned in `fit.py` (`FIT_KICK_REST_FRAMES`)
so that later changes to the dead-ball stage do not move the degradation or M4's training data. It
is deterministic: a rerun writes the same file. Parsed matches are cached in `cache/` and the
per-run CSV goes to `results/`. Both folders are gitignored.

### Training M4's model

M4's per-frame model (`possession/learned.py`, DESIGN.md section 4, "M4") is trained here, on the two
calibration matches only, and needs neither the competition data nor anything outside this folder
but the DFL files:

```bash
pip install -r solution/requirements.txt                                        # scikit-learn 1.9.0
python3 -I solution/devset/train_model.py --dfl-dir /path/to/dfl                # ~1 min; rewrites possession/model.json
python3 -I solution/experiments/learned_gate.py --dfl-dir /path/to/dfl          # ~6 min; the leave-one-match-out gate
```

- **What goes in.** J03WMX and J03WPY, degraded with seeds 10–13 (never the scoring seeds 0–4) and
  run with synthetic tags. Every 5th frame the rules call play, where the truth merged at 2 s names
  a team, and its copy with the teams swapped.
- **What comes out.** A gradient-boosted tree ensemble (100 trees of at most 15 leaves), exported
  node by node to `possession/model.json` (about 100 KB). Its switch cost and minimum possession
  length are tuned on the same two matches, with probabilities each half gets from a model trained on
  the other half.
- **Reproducible.** Fixed settings, no early stopping, one thread: a rerun writes a byte-identical
  file. `solution/run.py` reads the JSON with numpy and needs neither scikit-learn nor the DFL data.
- **Licence.** The parameters are derived from the DFL data alone, so `model.json` carries the CC BY
  credit in its `credit` field ([Dataset and licence](#dataset-and-licence)).
- **Cost.** With M4 on, a dev-set call takes about 2 s instead of 0.25 s, so `run_devset.py`, which
  runs M4 in seven of its nine variants, takes about 20 minutes.

## Results

Every number is averaged over 5 degradation seeds. Brackets are 90% intervals from a two-level
bootstrap: matches, then possessions inside them (see `score.py`). "merged2s" and "merged3s" are the
truth with brief lost-back spells absorbed (see [Truth](#truth)). "M2c" is the rules alone, the
pipeline before M4: M2b with its two stoppage rules, earlier starts (`stoppage_starts`) and the restart team by the laws
of the game (`restart_laws`; DESIGN.md section 4, "M2c"). "M2b" is the same with both off
(`1bdf0ec`). "no foul stoppages" is M2b with `foul_stoppages=False`: the slowdown test is off but
M2b's join stays, so it is M3b's confirmation rule plus the join.

**"M4" is now the current pipeline:** M2c's stoppages, with the team in open play from a model
trained on the calibration matches (`learned_model`; DESIGN.md section 4, "M4"). Its rows come from a
second run, made while M2c was still the default, with M4 scored as one variant switched on, as
pre-registered. M2c's rows in that run reproduce the ones below to the third decimal.

The other rows come from the first run, made while M2b was still the default: M2c's rules were scored
as variants switched on, as pre-registered in DESIGN.md. So the rows that switch off an older stage
("no foul stoppages", "no boundary placement", ...) are M2b with that stage off. Since M4 is now the
default, `run_devset.py` prints "M2c: no learned team" and "M2b: no M2c rules, no learned team"
against M4, and every other stage switched off under M4, where it also changes what the model
reads.

### M4 on the held-out matches

**Verdict: the learned team holds up on matches it never saw, by more than in the gate. It is on by
default.**
- **The pre-registered test** (DESIGN.md section 4, "M4"): on the held-out five, M4 − M2c above zero
  under all three truths and both tag modes, with the interval excluding zero under raw and merged
  3 s in both. Every interval is above zero and every draw is above zero.
- **The gain:** +0.070 / +0.071 raw and +0.112 / +0.115 merged 3 s (tags / no tags). The
  leave-one-match-out gate on the calibration matches, with one training match per fold, gave
  +0.062 / +0.095 with tags.
- **On every held-out match**, raw and merged 3 s, with tags and without.
- **What moves besides spellF1.** Dead balls not at all (the stoppages are the rules'). macroF1 0.767
  to 0.801. Possessions per minute 2.35 to 2.54 (truth 4.49 raw, 2.91 merged 3 s).
- **Training** (`train_model.py`, [above](#training-m4s-model)): J03WMX and J03WPY only, seeds
  10–13. The calibration rows for M4 are in-sample and biased.

| pipeline | tags | spellF1 raw | spellF1 merged2s | spellF1 merged3s | macroF1 raw | D precision / recall | stoppages per match (truth 102.6) | poss/min (truth 4.49 raw / 2.91 merged3s) |
|---|---|---|---|---|---|---|---|---|
| **M4** | tags | **0.414** [0.383, 0.445] | **0.502** [0.477, 0.528] | **0.542** [0.513, 0.576] | 0.801 | 0.95 / 0.77 | 75.5 | 2.54 |
| M2c | tags | 0.344 [0.313, 0.377] | 0.401 [0.371, 0.430] | 0.430 [0.397, 0.470] | 0.767 | 0.95 / 0.77 | 75.5 | 2.35 |
| **M4** | no tags | **0.405** [0.376, 0.435] | **0.492** [0.468, 0.518] | **0.532** [0.505, 0.565] | 0.792 | 0.96 / 0.74 | 69.2 | 2.50 |
| M2c | no tags | 0.334 [0.306, 0.365] | 0.389 [0.362, 0.417] | 0.418 [0.387, 0.455] | 0.757 | 0.96 / 0.74 | 69.2 | 2.31 |

| difference | tags | raw | merged2s | merged3s |
|---|---|---|---|---|
| **M4 − M2c** | tags | **+0.070** [+0.057, +0.082] 100% | **+0.101** [+0.088, +0.115] 100% | **+0.112** [+0.096, +0.129] 100% |
| **M4 − M2c** | no tags | **+0.071** [+0.058, +0.082] 100% | **+0.103** [+0.091, +0.116] 100% | **+0.115** [+0.099, +0.132] 100% |
| M2c − smooth50 | tags | +0.111 [+0.085, +0.138] 100% | +0.158 [+0.132, +0.183] 100% | +0.197 [+0.164, +0.234] 100% |
| M2c − smooth50 | no tags | +0.102 [+0.078, +0.125] 100% | +0.147 [+0.123, +0.169] 100% | +0.185 [+0.153, +0.219] 100% |

Calibration (in-sample for M4) and all seven, M4 − M2c, raw / merged 3 s:

| | calibration (2) | all seven |
|---|---|---|
| M4 with tags: spellF1 raw / merged2s / merged3s | 0.423 / 0.571 / 0.596 | 0.417 / 0.521 / 0.558 |
| M4 − M2c, tags | +0.081 [+0.060, +0.105] / +0.138 [+0.116, +0.159] | +0.073 [+0.063, +0.085] / +0.119 [+0.104, +0.134] |
| M4 − M2c, no tags | +0.084 [+0.062, +0.108] / +0.143 [+0.119, +0.166] | +0.075 [+0.064, +0.086] / +0.122 [+0.107, +0.138] |
| possessions per minute, M4 tags (M2c) | 2.36 (2.23) | 2.49 (2.31) |

### Held-out matches (5) — the real test

| pipeline | tags | spellF1 raw | spellF1 merged2s | spellF1 merged3s | macroF1 raw | D precision / recall | stoppages per match (truth 102.6) | poss/min (truth 4.49 raw / 2.91 merged3s) |
|---|---|---|---|---|---|---|---|---|
| **M2c** | tags | **0.344** [0.313, 0.377] | **0.401** [0.371, 0.430] | **0.430** [0.397, 0.470] | 0.767 | 0.95 / 0.77 | 75.5 | 2.35 |
| M2b + restart laws only | tags | 0.338 [0.309, 0.370] | 0.395 [0.366, 0.424] | 0.424 [0.391, 0.461] | 0.756 | 0.96 / 0.74 | 77.6 | 2.39 |
| M2b + stoppage starts only | tags | 0.335 [0.304, 0.369] | 0.390 [0.361, 0.419] | 0.418 [0.384, 0.456] | 0.762 | 0.95 / 0.77 | 75.5 | 2.37 |
| **M2b** | tags | **0.330** [0.300, 0.361] | **0.383** [0.355, 0.411] | **0.411** [0.379, 0.447] | 0.751 | 0.96 / 0.74 | 77.6 | 2.40 |
| no foul stoppages | tags | 0.305 [0.274, 0.339] | 0.354 [0.324, 0.385] | 0.379 [0.345, 0.417] | 0.661 | 0.95 / 0.52 | 59.8 | 2.26 |
| no boundary placement | tags | 0.327 | 0.381 | 0.409 | 0.750 | 0.96 / 0.74 | 77.6 | 2.40 |
| no duel rule | tags | 0.329 | 0.380 | 0.407 | 0.751 | 0.96 / 0.74 | 77.6 | 2.48 |
| no keeper split | tags | 0.327 | 0.380 | 0.408 | 0.748 | 0.96 / 0.74 | 77.7 | 2.41 |
| **M2c** | no tags | **0.334** [0.306, 0.365] | **0.389** [0.362, 0.417] | **0.418** [0.387, 0.455] | 0.757 | 0.96 / 0.74 | 69.2 | 2.31 |
| M2b + restart laws only | no tags | 0.330 [0.302, 0.359] | 0.384 [0.357, 0.411] | 0.412 [0.381, 0.447] | 0.748 | 0.96 / 0.72 | 70.5 | 2.34 |
| M2b + stoppage starts only | no tags | 0.325 [0.296, 0.355] | 0.377 [0.350, 0.404] | 0.404 [0.375, 0.438] | 0.750 | 0.96 / 0.74 | 69.2 | 2.35 |
| **M2b** | no tags | **0.320** [0.292, 0.350] | **0.371** [0.345, 0.397] | **0.398** [0.369, 0.431] | 0.742 | 0.96 / 0.72 | 70.5 | 2.37 |
| no foul stoppages | no tags | 0.284 [0.255, 0.316] | 0.328 [0.299, 0.359] | 0.350 [0.318, 0.387] | 0.601 | 0.96 / 0.39 | 42.2 | 2.16 |
| baseline_smooth50 | n/a | 0.233 [0.216, 0.247] | 0.243 [0.229, 0.256] | 0.233 [0.221, 0.246] | 0.616 | 0.76 / 0.59 | 122.5 | 5.15 |

Paired differences in spellF1 on the held-out five, under the same draws. The share of draws above
zero follows each interval.

| difference | tags | raw | merged2s | merged3s |
|---|---|---|---|---|
| **M2c − M2b** | tags | **+0.014** [+0.010, +0.019] 100% | **+0.018** [+0.012, +0.023] 100% | **+0.019** [+0.014, +0.026] 100% |
| **M2c − M2b** | no tags | **+0.014** [+0.011, +0.018] 100% | **+0.018** [+0.014, +0.023] 100% | **+0.020** [+0.015, +0.025] 100% |
| restart laws only − M2b | tags | +0.009 [+0.005, +0.013] 100% | +0.011 [+0.007, +0.017] 100% | +0.013 [+0.008, +0.017] 100% |
| restart laws only − M2b | no tags | +0.009 [+0.006, +0.013] 100% | +0.013 [+0.009, +0.017] 100% | +0.014 [+0.010, +0.018] 100% |
| stoppage starts only − M2b | tags | +0.006 [+0.004, +0.008] 100% | +0.007 [+0.005, +0.009] 100% | +0.007 [+0.005, +0.010] 100% |
| stoppage starts only − M2b | no tags | +0.005 [+0.003, +0.007] 100% | +0.006 [+0.004, +0.008] 100% | +0.006 [+0.005, +0.009] 100% |
| **M2b − no foul stoppages** | tags | **+0.024** [+0.018, +0.030] 100% | **+0.029** [+0.021, +0.037] 100% | **+0.032** [+0.025, +0.041] 100% |
| **M2b − no foul stoppages** | no tags | **+0.036** [+0.029, +0.043] 100% | **+0.043** [+0.033, +0.053] 100% | **+0.048** [+0.039, +0.058] 100% |
| M2b − smooth50 | tags | +0.097 [+0.071, +0.122] 100% | +0.140 [+0.116, +0.163] 100% | +0.178 [+0.146, +0.211] 100% |
| M2b − smooth50 | no tags | +0.087 [+0.064, +0.110] 100% | +0.128 [+0.106, +0.149] 100% | +0.165 [+0.137, +0.195] 100% |
| M2b − no boundary placement | tags | +0.002 [−0.001, +0.006] 87% | +0.003 [−0.001, +0.007] 86% | +0.002 [−0.003, +0.006] 76% |
| M2b − no duel rule | tags | +0.000 [−0.001, +0.002] 62% | +0.003 [+0.002, +0.004] 100% | +0.004 [+0.003, +0.006] 100% |
| M2b − no keeper split | tags | +0.002 [+0.001, +0.004] 100% | +0.003 [+0.002, +0.004] 100% | +0.003 [+0.002, +0.005] 100% |

(For the older stages the runner printed "variant − M2b"; those rows are flipped here so that every
row reads as what the stage adds. The no-tags rows for the last three stages are within 0.001 of the
tags rows.)

### Calibration matches (2) and all seven

| | calibration (2) | all seven |
|---|---|---|
| M2c with tags: spellF1 raw / merged2s / merged3s | 0.342 / 0.441 / 0.458 | 0.343 / 0.412 / 0.438 |
| M2c − M2b, tags: raw / merged3s | +0.008 [+0.003, +0.015] / +0.016 [+0.011, +0.021] | +0.013 [+0.009, +0.017] / +0.018 [+0.014, +0.023] |
| M2c − M2b, no tags: raw / merged3s | +0.007 [+0.001, +0.013] / +0.018 [+0.013, +0.024] | +0.012 [+0.008, +0.016] / +0.019 [+0.016, +0.024] |
| restart laws only − M2b, tags: raw / merged3s | +0.003 [+0.000, +0.007] / +0.010 [+0.006, +0.014] | +0.007 [+0.004, +0.011] / +0.012 [+0.008, +0.016] |
| stoppage starts only − M2b, tags: raw / merged3s | +0.005 [+0.002, +0.009] / +0.007 [+0.004, +0.010] | +0.006 [+0.004, +0.008] / +0.007 [+0.005, +0.009] |
| D precision / recall, M2c vs M2b (tags) | 0.95 / 0.83 vs 0.96 / 0.81 | 0.95 / 0.79 vs 0.96 / 0.76 |
| stoppages per match, M2c tags / no tags | 75.0 / 68.9 | 75.4 / 69.1 |
| possessions per minute, M2c tags | 2.23 | 2.31 |
| M2b with tags: spellF1 raw / merged2s / merged3s | 0.333 / 0.427 / 0.442 | 0.331 / 0.396 / 0.420 |
| M2b − no foul stoppages, tags: raw / merged3s | +0.022 [+0.014, +0.032] / +0.030 [+0.019, +0.041] | +0.023 [+0.019, +0.029] / +0.032 [+0.026, +0.038] |
| M2b − no foul stoppages, no tags: raw / merged3s | +0.034 [+0.023, +0.048] / +0.048 [+0.034, +0.063] | +0.035 [+0.029, +0.042] / +0.048 [+0.040, +0.056] |
| M2b − smooth50, tags: raw / merged3s | +0.098 / +0.197 | +0.097 / +0.183 |
| D precision / recall, M2b vs no foul stoppages (tags) | 0.96 / 0.81 vs 0.95 / 0.58 | 0.96 / 0.76 vs 0.95 / 0.53 |
| D precision / recall, M2b vs no foul stoppages (no tags) | 0.96 / 0.79 vs 0.95 / 0.45 | 0.96 / 0.74 vs 0.95 / 0.40 |
| stoppages per match, M2b tags / no tags (truth) | 77.6 / 70.4 (100.0) | 77.6 / 70.5 (101.9) |
| possessions per minute, M2b tags (truth raw / merged3s) | 2.30 (4.78 / 2.91) | 2.37 (4.58 / 2.91) |
| M2b − no boundary placement, tags: merged3s | −0.006 [−0.012, −0.002] | −0.000 [−0.005, +0.004] |

**Per match** (spellF1, raw / merged 3 s, with tags):

| | J03WMX\* | J03WPY\* | J03WN1 | J03WOH | J03WOY | J03WQQ | J03WR9 |
|---|---|---|---|---|---|---|---|
| M4 | 0.404 / 0.585 | 0.445 / 0.608 | 0.368 / 0.504 | 0.404 / 0.546 | 0.395 / 0.507 | 0.455 / 0.574 | 0.459 / 0.588 |
| M2c | 0.335 / 0.458 | 0.350 / 0.458 | 0.292 / 0.377 | 0.350 / 0.419 | 0.316 / 0.396 | 0.378 / 0.477 | 0.397 / 0.491 |
| M2b | 0.330 / 0.443 | 0.337 / 0.440 | 0.275 / 0.356 | 0.342 / 0.407 | 0.305 / 0.382 | 0.364 / 0.456 | 0.374 / 0.461 |
| no foul stoppages | 0.314 / 0.419 | 0.309 / 0.404 | 0.252 / 0.327 | 0.317 / 0.378 | 0.272 / 0.336 | 0.348 / 0.429 | 0.351 / 0.430 |
| smooth50 | 0.226 / 0.239 | 0.247 / 0.252 | 0.232 / 0.240 | 0.238 / 0.238 | 0.205 / 0.209 | 0.256 / 0.243 | 0.235 / 0.237 |

\* calibration (in-sample for M4). M4 beats M2c, M2c beats M2b, and M2b beats "no foul stoppages"
and smooth50, on every match, with tags and without.

### M2c on the held-out matches

**Verdict: both of M2c's rules hold up on matches they never saw, by more than on the two they were
built on. Both are now on by default.**
- **The pre-registered test** (DESIGN.md section 4, "M2c"): a rule goes on if its paired difference
  against M2b on the held-out five is above zero under all three truths and both tag modes, and its
  interval excludes zero under merged 3 s in both. Both rules pass with every interval above zero
  and all draws above zero.
- **Restart laws** (the thrower on the touchline; attackers in the penalty area at a corner, kept
  out of it at a goal kick): +0.009 raw, +0.013 / +0.014 merged 3 s. On calibration it gained only
  +0.003 / +0.004 raw.
- **Stoppage starts** (an earlier crossing the ball never came back from; where the slowdown set in):
  +0.005 / +0.006 raw, +0.006 / +0.007 merged 3 s, about what calibration showed.
- **Together** +0.014 raw and +0.019 / +0.020 merged 3 s, about the sum of the two. M2c is ahead of
  M2b on each of the five held-out matches, raw and merged 3 s, with tags and without.
- **What moves besides spellF1.**
  - Dead-ball recall rises from 0.74 to 0.77 with tags, and precision falls from 0.96 to 0.95.
  - Stoppages per match fall from 77.6 to 75.5: earlier starts merge some into their neighbours.
  - Possessions per minute fall from 2.40 to 2.35: a restart team now named removes the short spell
    the carried team used to get after the stoppage, when the same team then touches the ball.
- **The rule tables, start offsets and ceilings** behind M2c are calibration-only:
  `solution/experiments/stoppage_ceiling.py` and `solution/experiments/restart_rules.py`.

### M2b on the held-out matches

**Verdict: the foul-stoppage rule holds up on matches it never saw. The held-out results do not
argue for changing any threshold.**
- **It works as well on held-out matches as on calibration ones.**
  - spellF1 gain: +0.024 / +0.032 (raw / merged 3 s) with tags, against +0.022 / +0.030 on
    calibration. Without tags: +0.036 / +0.048, against +0.034 / +0.048.
  - Every interval excludes zero, and the rule helps on all five held-out matches.
- **Recall improves; precision holds.**
  - Dead-ball recall rises from 0.52 to 0.74 with tags, and from 0.39 to 0.72 without.
  - Precision stays at 0.96.
  - It finds 78 of ~103 stoppages per match with tags (60 without the rule), and 71 without tags
    (42).
- **Its thresholds, re-checked on the held-out five** (`foul_check.py`, report only). This is the
  calibration's own measurement: DFL's player speeds and BallStatus. On the two calibration matches
  it reproduces M2b's numbers to the candidate.

  | | calibration (2) | held-out (5) |
  |---|---|---|
  | slowdown candidates without a line crossing | 139 | 457 |
  | … wholly in open play / mostly dead / mixed | 36 / 103 / 0 | 100 / 347 / 10 |
  | **open-play candidates the test passes (false stoppages)** | **0 of 36** | **3 of 100** (0.6 per match) |
  | **dead candidates it catches** | **58 of 103 (56%)** | **199 of 347 (57%)** |
  | dead share of the frames it passes | 96% | 95% |
  | longest open-play slowdown / lowest collective speed in open play | 14.5 s / 0.79 m/s | 24.5 s / 0.68 m/s |
  | the 0.70 m/s / 20 s alternative DESIGN.md named beforehand: false / caught | 0 / 45 | 2 / 168 |
  | `JOIN_GAP`: gaps under 3 s that are dead throughout | 6 of 6 | 25 of 27 |

  - **All three false stoppages pass on length:** 15.8, 16.6 and 24.5 s of open play. The 16.6 s one
    also goes below 0.75 m/s.
  - **Open play does sometimes last longer than 15 s,** which the calibration matches never showed.
    DESIGN.md extrapolated "about half a false stoppage per match"; the held-out matches show 0.6.
  - **The trade-off favours the current setting.** The pre-named looser alternative would save one
    false stoppage and lose 31 caught ones. DESIGN.md section 1 costs a missed and a false stoppage
    about the same.
- **On the degraded tracking** the pipeline actually sees, the same test passes 0.5 open-play
  stretches per held-out match (13 over 5 matches × 5 seeds). They come from three stretches, each
  14–21 s long.

### What else the held-out matches change

- **Boundary placement (M3) is not harmful after all.** On the calibration matches, removing it
  scored +0.006 under merged 3 s labels. On the held-out five it costs −0.002 [−0.006, +0.003], and
  pooled over all seven it is 0.000. The earlier "surprise" was a property of the two calibration
  matches.
- **The duel rule and keeper split replicate.** Both keep their sign. The duel rule gains
  +0.003–0.004 under the merged truths on the held-out matches, a little less than on calibration
  (+0.005–0.007), and still nothing under the raw truth. The keeper split gains +0.002–0.003 on both
  groups.
- **smooth50 stays far below**, by +0.09–0.18 on the held-out matches.

### How noisy is it?

- **Possessions.** Absolute pooled spellF1 has a 90% half-width of about ±0.03 on the five held-out
  matches, and ±0.024 on all seven. Paired differences are much tighter (±0.002–0.010), because both
  variants miss the same hard possessions. That is the number to read when ranking a change.
- **Degradation seeds.** One seed alone moves pooled spellF1 over all seven matches by sd 0.005–0.007
  (0.008–0.009 on two matches). Every number averages 5 seeds.
- **Matches.**
  - Match-to-match spread is real: M2b scores 0.275 on J03WN1 (red card) and 0.374 on J03WR9.
  - The ranking of M2b against "no foul stoppages" and smooth50 never changes between matches.
  - With five held-out matches, the match level of the bootstrap now does real work.
- **Rough guide:** a change worth more than ~0.01 spellF1 that keeps its sign under all three truths,
  both tag modes and the held-out group is probably real for matches like these. Changes of ~0.003
  that are consistent everywhere (duel rule, keeper split) are detectable now, but small.

## Truth

`truth.py` builds the dev truth the way the competition builds its truth: D where DFL's `BallStatus` is 0,
otherwise H or A from `BallPossession`. The two halves are the graded blocks, with X between them.
The counts match the audit exactly: 511 and 421 spells raw, 341 / 307 merged at 2 s, 289 / 277 at 3 s.
Across all seven matches the raw truth has 3.8–5.3 possessions per minute, 2.7–3.3 merged at 3 s,
92–113 stoppages and 41–52% dead frames.

The audit found one real difference of definition. DFL credits a touch that is lost straight back to
the other team; our truth does not. The merged variants absorb any H or A spell shorter than 2 s or
3 s that sits between two spells of the other team, with no dead ball in between. A spell next to a
dead ball is kept, because "touched it last before it went out" is a possession our truth keeps. The
threshold is an assumption, and a rule that itself merges short spells cannot be validated against a
truth merged the same way. **Trust a change only if it holds under all three truths.**

## Degradation and how each parameter was fitted

Every number is in `params.json`, written by `fit.py`. Only aggregates leave the competition data:
quantiles of stretch lengths (21 points), shares, smoothing widths and tag counts. No frame-level
mask or position is committed. The matched widths use the two calibration matches only; the held-out
five were degraded with the same file and shaped nothing in it.

A few definitions used below:
- **Fitted target.** Most targets are measured on the competition tracking over graded frames.
- **Dead.** The pipeline's own stoppages on the real match.
- **Settled play.** Graded frames outside those stoppages and outside every slowdown, so that missed
  stoppages do not count as play.
- **Matched parameters.** A smoothing width or offset is chosen as the value at which the DFL matches
  show the same statistic as the competition.

| step | parameter | value | how it was fitted | target → result on DFL |
|---|---|---|---|---|
| 1. dead-ball surgery | run-out, rest before kick | 25, 50 frames | **Not fitted** (see below). The ball rolls on with its last velocity and slows to rest over 1 s. It is carried in a straight line to the restart spot and placed there 2 s before the kick. The kick stays where DFL has it. | — |
| 2. ball smoothing | Gaussian σ | 8 frames | Matched: the median frames a kick takes to ramp from 20% to 80% of its peak speed | 14 → 14 frames |
| 3. ball misplacement | AR(1) offset sd, time constant | 0.2 m, 25 frames | Matched: lower quartile of the distance from the ball to the nearest player in settled play. The time constant is assumed. | 0.90 → 0.91 m |
| 4. ball dropout | per-frame lose / find rates in 6 phases | e.g. settled play 0.0012 / 0.0036, long stoppage 0.0045 / 0.0009 | Counted directly: transitions per seen or unseen frame. The phases are play, its first 3 s and last 3 s, and the same three for dead balls. | ball seen 59.3% of graded frames → 55.7% / 57.4% |
| 5. player smoothing | Gaussian σ | 1.25 frames | Matched: robust sd of each track's second difference. The competition's tracks are smooth (1.6 mm), and DFL's raw positions are rounded to 1 cm. | 1.56 → 1.36 mm |
| 5. track slots | tracklet lifetimes | median 873 frames | Quantiles of outfield slot runs (1143) | — |
| | re-acquisition gap | 66% within 10 s (median 87 frames); the rest 250 + 71 frames on average | A run that ends is matched to the next run of the same team that starts within reach (2 m + 7 m/s). The late mean reproduces the untracked share. | tracked per team 10.24 → 10.28 / 10.20 |
| | keeper slot shows the other keeper | 6.1% of its frames, 16 stretches (median 440 frames); own keeper then untracked | Keeper slot standing in the other team's half for ≥ 1 s | — |
| 6. speed | missing | every tracklet's first 8–21 frames, short gaps elsewhere (2.5% of outfield frames); keeper slots 64% in long stretches | Counted directly | 8.0% of tracked slot-frames → 6.7% |
| tags | per type, team, window | 26 throw-ins, 12 corners, 10 goal kicks, 6 kick-offs, 5 free kicks; throw-ins and kick-offs `N/A`; windows 35 s (kick-offs 5 s) starting a median 5 s before the restart | Counted on the 59 real set-piece tags. Window starts are offsets from the pipeline's restart for the 36 tags within 10 s of one. | 59 → 56 / 55 tags |

Checks of quantities that were **not** fitted directly:

| | seen stretch, median / q90 frames | unseen stretch, median / q90 frames | ball speed p99 |
|---|---|---|---|
| competition | 344 / 1562 | 268 / 873 | 24.6 m/s |
| J03WMX degraded | 354 / 1156 | 281 / 884 | 19.0 m/s |
| J03WPY degraded | 345 / 1153 | 310 / 838 | 19.9 m/s |
| held-out five, degraded | 338–407 / 1093–1276 | 226–308 / 899–970 | 18.9–20.6 m/s |

Where the ball is seen, by true state (competition figures are on dead = its own stoppages and on
settled play):

| | graded | dead | play |
|---|---|---|---|
| competition | 59.3% | 22.6% | 83.0% |
| J03WMX / J03WPY | 55.7% / 57.4% | 27.2% / 29.5% | 75.9% / 79.5% |
| held-out five | 53.0–58.7% | 28.8–31.7% | 74.7–82.4% |

**Why the dropout is phased.**
- The first version drew whole seen and unseen stretch lengths per state. That ties each stretch to
  the state it began in, and left the DFL ball seen as often in dead balls (57%) as in play (54%).
- A chain with one rate per state still blurred the contrast on the competition's own stoppages: 34%
  seen in dead balls and 72% in play, against the real 23% and 83%.
- The real ball is lost in the seconds *before* a stoppage (56%) and found again quickly after the
  restart (80%). Adding those edge phases brings the same check to 27% and 76%.

## Dataset and licence

The DFL / Sportec open data, CC BY 4.0 (<https://creativecommons.org/licenses/by/4.0/>), provided with
authorization from the DFL. Credit the DFL and cite:

> Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated dataset of spatiotemporal and
> event data in elite soccer* [Dataset]. figshare. <https://doi.org/10.6084/m9.figshare.28196177.v1>.
> Article: *Scientific Data*, <https://doi.org/10.1038/s41597-025-04505-y>.

Changes made to the data, as CC BY asks:
- Positions are shifted to a pitch-corner origin.
- The ball is replaced during dead balls and degraded.
- Players are smoothed and cut into anonymous track slots.
- Set-piece events are subsampled into synthetic tags.

None of the data, parsed arrays or results is committed. The competition data is used only for
`fit.py`, which publishes aggregates.

`solution/possession/model.json`, M4's trained model, is derived from the two calibration matches
alone (`train_model.py`): tree splits on features of the degraded tracking, and leaf values learned
from the DFL possession flag. No position or frame of the data is in it, but it is derived from the
data, so the credit above applies wherever it is used; its `credit` field carries it.

## Caveats

- **Seven matches, one provider, one country, one team in most.** Two are Bundesliga, five are
  2. Bundesliga. Fortuna Düsseldorf plays in five of the seven, including four of the five held-out
  matches.
  - The pooled intervals describe possessions in these matches, not the population of matches.
  - FC Barcelona's possession-heavy play is not represented. A long, slow spell of passing at the
    back, which is what triggers a false stoppage by length, may be more common in the real match.
  - J03WN1 has a red card in the first half, and J03WQQ has a team at 10 for part of the second
    half. Our pipeline does not care, but the tracked count per team is lower there.
- **Held-out means held-out once.** This check spends the five matches for M2b's thresholds, the
  stages already built, M2c's two rules and M4's model (each scored once, against a test written
  down beforehand). Re-tuning anything on them and quoting them again would turn them into
  calibration matches.
- **The truth's definition differs** (see [Truth](#truth)). Absolute spellF1 here is not comparable with
  the public window's 0.547 or the leaderboard. Read differences between variants, under all three
  truths.
- **Dead-ball edges.**
  - DFL flags the restart at the kick. How the competition's truth times a restart is not something
    the dev set can measure, and the public window is a correctness check, not a tuning set
    (RULES.md), so no restart lag is tuned on it. The audit's earlier suggestion of a lag sweep taken
    from the public window's restarts is withdrawn.
- **The synthetic dead-ball ball is invented.** DFL does not record where the ball goes during a
  stoppage, so the run-out, the carry and the 2 s rest are assumptions, not fits. A dead-ball
  detector that keys on how the ball moves inside a stoppage will be measured against our model of
  it. Detection of crossings and kicks is less exposed, because those come from the real ball before
  and after the stoppage.
- **The degradation is matched on marginals.** There is no clean/degraded pair of the same match, so
  nothing checks the joint behaviour. Known gaps:
  - Ball speed peaks are lower than the competition's (p99 19–20 vs 24.6 m/s). The Gaussian that
    matches the kick ramp also flattens short kicks.
  - Seen and unseen stretches are geometric, so their 90th percentile is shorter for seen stretches
    (1150 vs 1560 frames).
  - The ball is seen slightly more in dead balls and less in play than on the competition's selected
    frames, and 55–57% of graded frames overall against 59%. DFL matches have 42–44% dead frames.
  - Not reproduced at all: identity switches inside a slot, the ball stuck on the centre spot, the
    ball running 15–30 frames behind the real one, any positional bias of the tracker, and any false
    positives among the tags.
- **The "dead" and "settled play" frames used for fitting come from our own pipeline.** Its stoppages
  are its precision-first half of the real ones, so the dead-ball figures are measured on the
  stoppages it can find. The ones it misses (mostly fouls) may look different.
- **Tags.** Per-type counts are copied from the one real match, against ~100 set pieces in a DFL
  match. Tag window starts were measured against the pipeline's restart, which smoothing makes about
  1 s early, and are applied to DFL's true restart.
- **The adapter changes one setting for the length of each call.** `config.KNOWN_DEAD_SPANS` (the
  competition's published pre-kick-off span, 110808–113742) is emptied, because those frame numbers
  fall in the middle of a dev match's second half. It is restored afterwards. Nothing in
  `solution/possession/` is edited.

## Layout

```
run_devset.py   the one command: degrade, run, score, print the tables by group
foul_check.py   M2b's slowdown test re-checked on every match, clean and degraded; reports only
fit.py          fit params.json from the competition tracking (needs MTAH_DATA_DIR and --dfl-dir)
train_model.py  train M4's model on the calibration matches; writes possession/model.json
params.json     the fitted degradation parameters -- aggregates only
dfl.py          parse a DFL match (positions streamed with iterparse), cache arrays in cache/;
                CALIBRATION names the two matches anything was set on
truth.py        H / A / D truth from BallStatus and BallPossession; merged variants
degrade.py      dead-ball surgery, smoothing, dropout, track slots, speed -> atd.Tracking
tags.py         Smart Tagging stand-in from DFL set-piece events
adapter.py      label_frames on a dev match (empties KNOWN_DEAD_SPANS for the call); variants scored
baselines.py    baseline_smooth50 on a Tracking (reproduces the published file frame for frame)
score.py        possession_metrics pooled over matches; two-level, seed-averaged, paired bootstrap
tests/          synthetic-data tests; one uses the real match data when present and skips otherwise
```

To measure a change to the pipeline, merge this branch into that tree and run `run_devset.py`. It
imports `possession` from the same checkout. Add the change as a variant in `adapter.PIPELINES` (a
`label_frames` flag), then read its paired row against the current pipeline. Read it under all three
truths and both tag modes, and on the held-out group first.

The seven matches come from figshare article 28196177. Each match folder holds its positions,
events and match-information XML, with names and MD5s as listed by
<https://api.figshare.com/v2/articles/28196177/files?page_size=50>. About 2.6 GB in all.
