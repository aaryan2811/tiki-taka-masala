# Ball possession from tracking

Team Tiki Taka Masala's entry to
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) (Metrica Sports
and Barça Innovation Hub), Phase 1. It labels each of the 199,319 frames of FC
Barcelona – Athletic Club as `H` (home in possession), `A` (away), `D` (dead ball) or `X` (not
graded), from the tracking data and the Smart Tagging alone.

- **Dead balls and restarts** come from rules, built for precision because a false dead ball costs a
  whole possession.
- **The team in open play** comes from gradient-boosted trees trained on two open DFL matches. A
  two-state Viterbi path between the stoppages turns their per-frame output into possessions.
- **Validated** on five other DFL matches that shaped nothing in it, degraded to look like this
  match's tracking: spellF1 0.414, against 0.344 for the rules alone and 0.233 for the organisers'
  `baseline_smooth50`.

**The technical document is [TECHNICAL.md](TECHNICAL.md):** the approach, the assets and training
data, the possession rules, the validation and the limitations. The full engineering record, with
every measurement, is [solution/DESIGN.md](solution/DESIGN.md).

## Reproduce the submission

Every command below runs from the repository root.

**1. Python 3.14 and the pinned packages.** Tested with Python 3.14.6 on macOS (arm64).

```bash
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

What each package is for:
- **numpy** is all the file depends on. `run.py` evaluates the trained model with numpy alone.
- **pandas** validates and scores the file.
- **scikit-learn and threadpoolctl** are only for retraining the model and for the experiments
  ([below](#the-model-the-development-set-and-the-experiments)).
- **pytest** runs the tests.

To regenerate and check the file without the rest, `pip install numpy==2.4.6 pandas==3.0.3` is
enough.

**2. The data bundle.** It is gated and not in this repository (the organisers'
[DATA.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/DATA.md)
says how to get it). The pipeline reads three files of it, all from the tactical source; the video,
the homography and the broadcast source are not read:

```
data/tactical/BAR-ATH_tactical_atd.txt             tracking, players and ball
data/tactical/BAR-ATH_tactical_atd.xml             tracking metadata
data/tactical/BAR-ATH_tactical_smart_tagging.xml   Smart Tagging
```

Unpack the bundle into `data/`, or leave it anywhere and point `MTAH_DATA_DIR` at the folder that
holds `tactical/` (`--data-dir` does the same per run):

```bash
export MTAH_DATA_DIR=/path/to/bundle
```

**3. Run.**

```bash
python solution/run.py --name <team_name>
```

It writes `evaluation/results/<team_name>.csv` (`submission.csv` without `--name`) and runs the
official format checks on it before it exits. Among its last lines:

```
team in open play: M4's learned model (100 trees, trained on DFL J03WMX and J03WPY, CC BY 4.0), switch cost 20.0, possessions of at least 50 frames
dead balls: 71 (1 known+oob+slow+tag, 4 oob, 4 oob+slow, 14 oob+slow+tag, 5 oob+tag, 16 slow, 24 slow+tag, 3 tag), median 22.2 s, restart team unknown for 7
```

It takes a few seconds. The first run parses the raw tracking and caches it as arrays in
`solution/cache/` (gitignored); later runs read the cache, which is rebuilt whenever the raw file's
size or modification time changes. `--refresh` re-parses regardless. The name changes the file name,
not its bytes.

**4. Check it.**

```bash
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
shasum -a 256 evaluation/results/<team_name>.csv          # Linux: sha256sum
```

The validator must print `Valid.` The SHA-256 must be

```
c77d0ae0bcf5556bf0aca67b82e930b938230652da6f0fe6c6e3504fd5cbffb1
```

We got that hash in every way we tried (TECHNICAL.md, "Reproduction"):
- from a fresh clone: cold run, warm run and `--refresh`;
- from another working directory, and under `python -I`;
- with different `PYTHONHASHSEED`s and thread counts;
- from a bundle holding only the three files above.

Any change to the pipeline or to `solution/possession/model.json` changes it, so it is re-recorded
with every change.

If your hash differs:
- **Check the inputs first.** The bundle we ran on:

  ```
  f130e350d2bfdf734b7e32f072931fccc6d015c1e837d082064875fe0a700431  BAR-ATH_tactical_atd.txt
  327e35e32e46a647389b88f883d4f9a2ab931e3e6148aef2fcf8f22b1e93c43a  BAR-ATH_tactical_atd.xml
  0594d35e3df01b9c5002372d8615e2e8ef5379ac8f4e84c5ef7a53f4304c9019  BAR-ATH_tactical_smart_tagging.xml
  ```

- **Line endings** are `\n` on every platform, Windows included (`labels.write`).
- **On another platform** float rounding in C libraries could matter. The model's trees compare
  their time features as whole frame counts, so `np.log1p`'s last bit cannot move a frame
  (TECHNICAL.md, "Reproduction"). Only macOS arm64 has been checked.

**5. Score it** (the public 5 minutes only; the ranking uses the private full match):

```bash
python solution/evaluate.py evaluation/results/<team_name>.csv --bootstrap 1000   # public window + full-match statistics
python evaluation/run_evaluation.py                                               # the official board, with the baselines
```

Expected on the public window, as a correctness check: spellF1 0.6222 [0.3852, 0.8058], macroF1
0.9023.

**The rules alone.** `python solution/run.py --rules-only --name rules_only` writes the pipeline
without the learned model: its team in open play is the rules'. Its SHA-256 is
`cf791a5e4ec33f3388507b77727cf2d10bbcc2b16b27d3300ec5d83e0960f41b`, and it scores 0.5655
[0.3717, 0.7448] / 0.8962 on the public window.

## Without the match data

These run on a fresh clone with no data at all:

```bash
python -m pytest solution/tests solution/devset/tests   # the pipeline, the model and the dev set on synthetic data
python solution/experiments/metric_sensitivity.py       # what each kind of error costs on spellF1
python solution/evaluate.py evaluation/results/baseline_smooth50.csv
```

## The model, the development set and the experiments

`solution/devset/` scores the pipeline on seven open DFL / Sportec matches (CC BY 4.0) with
per-frame truth, degraded to look like the competition tracking. It needs those matches, downloaded
separately; see [solution/devset/README.md](solution/devset/README.md). `run.py` never reads them.

**Retraining the model.** `solution/possession/model.json` is trained on two of those matches, and
the trained file is committed:

```bash
python3 -I solution/devset/train_model.py --dfl-dir /path/to/dfl     # about a minute; rewrites possession/model.json
```

- **What it needs:** the DFL files and scikit-learn 1.9.0 with threadpoolctl 3.5.0 (both in
  `requirements.txt`). `run.py` needs neither.
- **It is deterministic:** one thread, no early stopping, so a rerun writes the same file, SHA-256
  `6ba3f99919e01a21d18d0d692366eee1eb05420efc691dead137bef3e92c4ef7`.
- **To compare without overwriting it,** pass `--out /tmp/model.json` and compare the two files.

Run it with `python3 -I`: the DFL files come from outside this repository.

**The experiments** feed DESIGN.md. What each needs:

| needs | experiments |
|---|---|
| nothing | `metric_sensitivity.py` |
| the match data | `boundary_timing.py`, `contested_phases.py` |
| the DFL matches (`--dfl-dir`) | `restart_rules.py`, `stoppage_ceiling.py`, `open_play.py`, `learned_gate.py`, `dead_ball_windows.py`, `target_dead_ball.py` |
| both | `learned_drift.py`; `foul_stoppages.py`, which needs the DFL files for its DFL part only |

## Layout

```
README.md                 this file
TECHNICAL.md              the technical document
requirements.txt          pinned dependencies
solution/                 our pipeline
  run.py                  regenerate the submission; --rules-only for the rules alone
  evaluate.py             score one file: public window with bootstrap, full-match statistics
  possession/             the pipeline, one module per stage (see possession/__init__.py)
    config.py             every tunable number, with the reason for its value
    model.json            the trained model, from DFL data alone (devset/train_model.py)
  devset/                 the DFL development set: degradation, truth, scoring, training
  experiments/            one-off studies whose results feed DESIGN.md
  tests/                  synthetic-data tests
  DESIGN.md               the engineering record
  LICENSE                 MIT, for our code and documents
  cache/                  parsed tracking -- gitignored, never commit it

from the organisers' starter kit (MIT, see LICENSE):
games.py                  match constants and data paths
evaluation/               the official scorer and validator, the public ground truth, the baselines
starter/                  the two loaders the pipeline reuses
data/                     where the gated bundle is unpacked (empty in git)
```

The challenge's own documents (CHALLENGE.md, RULES.md, FAQ.md, DATA.md, EVALUATION.md,
SUBMISSION.md) are cited by name throughout. They are in the
[organisers' repository](https://github.com/metrica-sports/more-than-a-hack-2027/tree/f8f2629ff1e0985f81d7e59d8413678514ee1530),
at the version this entry follows.

## Rules we hold ourselves to

The organisers'
[RULES.md](https://github.com/metrica-sports/more-than-a-hack-2027/blob/f8f2629ff1e0985f81d7e59d8413678514ee1530/RULES.md)
(as of their commit `f8f2629`, 2026-10-23) is the authority; these are how we apply it.
- No hand labels on this match, in any amount or for any purpose, and no human-produced data about
  it from any source. Nothing derived from the private ground truth.
- The model is trained on DFL matches only. This match enters only as unlabelled tracking and Smart
  Tagging: the features `run.py` computes, and the aggregates the dev set's degradation is fitted to.
- The public window is a correctness check, "not training data and not a tuning set". Never choose
  between two versions on it, and never validate a model on it. Some early values were chosen after
  looking at public-window cases; DESIGN.md section 4 ("Re-checked on DFL") says which, and what now
  justifies each.
- Watching the video is for understanding the data. No label, correction, rule or constant refers to
  a frame of this match, except the dead span FAQ.md publishes before the second-half kick-off.
- No data, cache or submission file in git. `solution/.gitignore`, `solution/devset/.gitignore` and
  the root `.gitignore` cover them.

## Credits

The development set and the model use the DFL / Sportec open data, CC BY 4.0, provided with
authorization from the DFL: Bassek, M., Rein, R., Weber, H., & Memmert, D. (2025). *An integrated
dataset of spatiotemporal and event data in elite soccer* [Dataset]. figshare.
<https://doi.org/10.6084/m9.figshare.28196177.v1>. TECHNICAL.md ("Credits") says how we changed it.

`games.py`, `evaluation/`, `starter/` and `data/README.md` are from the organisers' starter kit,
© Metrica Sports, under the MIT licence in [LICENSE](LICENSE).

## Licence

Our code and documents (`solution/`, this README and TECHNICAL.md) are under the MIT licence in
[solution/LICENSE](solution/LICENSE), © 2026 Tiki Taka Masala. The organisers' files keep their own
MIT licence, [LICENSE](LICENSE).
