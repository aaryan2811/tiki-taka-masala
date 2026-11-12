"""Score the possession pipeline on the DFL dev set. The one command that reproduces README.md.

    python solution/devset/run_devset.py --dfl-dir /path/to/dfl
    python solution/devset/run_devset.py --dfl-dir /path/to/dfl --seeds 3 --bootstrap 1000

For every DFL match under --dfl-dir (or --matches): parse it (cached), build its three truth
variants (truth.py), degrade it to look like the competition's tracking (degrade.py, with the
committed params.json), make a Smart Tagging stand-in (tags.py), and run each pipeline variant with
and without tags (adapter.py) plus baseline_smooth50 (baselines.py). Then score everything with the
competition's metric, pooled over matches with a two-level bootstrap (score.py), and print the
tables as Markdown. Nothing here needs the competition data: the degradation reads params.json only.

Tables, in order:
  data       per match: the truth's shape, and how the degraded tracking compares with the
             competition's on the numbers the degradation was fitted to
  per match  spellF1 per match, raw truth and merged 3 s
  then, for each group of matches -- calibration (dfl.CALIBRATION, on which params.json's matched
  widths and M2b's slowdown thresholds were set, so biased), held-out (every other match: the real
  test), and all of them pooled:
    headline pooled spellF1 [90% CI] under each truth variant, the per-frame view, stoppages per
             match and possessions per minute against the truth
    paired   every variant against the current pipeline, and the current pipeline against
             smooth50, under the same draws
  seeds      (with --seeds > 1) how much the degradation's randomness alone moves the score

The CSV of every number goes to --out (default solution/devset/results/, gitignored).
"""

import argparse
import csv
import os
import sys
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from devset import adapter, degrade, dfl, score, tags, truth  # noqa: E402  (needs the path above)
from possession import ball as ball_stage  # noqa: E402
from possession.runs import true_runs  # noqa: E402

TAG_MODES = (("tags", True), ("no tags", False))


def match_seed(match_id, seed):
    """A fixed seed per match and repeat, so every run degrades a match the same way."""
    return zlib.crc32(match_id.encode("ascii")) + 7919 * seed


def predict(tracking, blocks, tag_list):
    """{(pipeline, tag mode): codes} for every variant on one degraded match."""
    out = {}
    for mode, use_tags in TAG_MODES:
        for name, flags in adapter.PIPELINES:
            out[(name, mode)] = adapter.label(tracking, blocks, tag_list if use_tags else (),
                                              **flags)[0]
        out[(adapter.SMOOTH50, mode)] = adapter.smooth50(tracking, blocks)
    return out


def data_row(match, tracking, blocks, tag_list, truths):
    """The numbers in the data table for one degraded match."""
    graded = np.zeros(tracking.n_frames, dtype=bool)
    for start, stop in blocks:
        graded[start:stop] = True
    raw = truths["raw"]
    dead = graded & (raw == score.pm.D)
    alive = graded & (raw != score.pm.D)
    seen = ~np.isnan(tracking.ball[:, 0])
    present = ~np.isnan(tracking.players[:, :, 0])
    minutes = graded.sum() / 25.0 / 60.0
    runs = score.pm.valid_runs(raw)
    seen_runs = [b - a for a, b in zip(*true_runs(seen)) if graded[a] and graded[b - 1]]
    unseen_runs = [b - a for a, b in zip(*true_runs(~seen & graded))
                   if a > 0 and graded[a - 1] and b < len(seen) and graded[b]]
    with np.errstate(invalid="ignore"):
        speed = ball_stage.jumps(tracking.ball)
    row = {"match": match.match_id, "teams": " - ".join(match.team_names),
           "seen_runs": np.percentile(seen_runs, (50, 90)),
           "unseen_runs": np.percentile(unseen_runs, (50, 90)),
           "speed_p99": float(np.nanpercentile(speed[graded], 99)),
           "minutes": minutes, "dead_share": float(dead.sum()) / graded.sum(),
           "stoppages": sum(code == score.pm.D for _, _, code in score.pm.segments(raw, runs)),
           "set_pieces": len(match.set_pieces), "tags": len(tag_list),
           "seen": float(seen[graded].mean()), "seen_dead": float(seen[dead].mean()),
           "seen_alive": float(seen[alive].mean()),
           "tracked_per_team": float(present[graded].sum(axis=1).mean() / 2.0)}
    for name, codes in truths.items():
        row["per_min_" + name] = len(score.pm.possession_spells(codes, runs)) / minutes
    return row


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dfl-dir", required=True, help="folder holding the DFL match folders")
    parser.add_argument("--matches", nargs="+", help="match ids (default: every one found)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="degradation repeats per match, averaged (one alone moves spellF1 by "
                             "about 0.008)")
    parser.add_argument("--bootstrap", type=int, default=1000, help="bootstrap draws")
    parser.add_argument("--out", default=os.path.join(HERE, "results"), help="where the CSV goes")
    args = parser.parse_args()

    params = degrade.load_params()
    match_ids = args.matches or dfl.available_matches(args.dfl_dir)
    if not match_ids:
        sys.exit("no DFL positions files under {}".format(args.dfl_dir))

    # scored[(pipeline, tag mode, truth variant)][seed] = [score.Scored per match]
    scored = {}
    data = []
    for match_id in match_ids:
        match = dfl.load(args.dfl_dir, match_id)
        truths = truth.variants(match)
        for seed in range(args.seeds):
            tracking, blocks = degrade.to_tracking(match, params, match_seed(match_id, seed))
            tag_list = tags.synthetic(match, blocks, params, match_seed(match_id, seed))
            if seed == 0:
                data.append(data_row(match, tracking, blocks, tag_list, truths))
            for (name, mode), codes in predict(tracking, blocks, tag_list).items():
                for variant, gt in truths.items():
                    by_seed = scored.setdefault((name, mode, variant),
                                                [[] for _ in range(args.seeds)])
                    by_seed[seed].append(score.Scored(gt, codes))
        print("scored {} ({})".format(match_id, " - ".join(match.team_names)), file=sys.stderr)

    variants = [name for name, _ in truth.VARIANTS]
    print(data_table(data, params, variants))
    print(per_match_table(scored, match_ids))
    for title, indices in groups(match_ids):
        names = ", ".join(match_ids[i] for i in indices)
        group = subset(scored, indices)
        print("# {} ({})\n".format(title, names))
        print(headline_table(group, variants, len(indices), args.seeds, args.bootstrap))
        print(paired_table(group, variants, args.seeds, args.bootstrap))
    if args.seeds > 1:
        print(seeds_table(scored, args.seeds))
    path = write_csv(scored, match_ids, args.out)
    print("\nwrote {}".format(os.path.relpath(path)), file=sys.stderr)


def groups(match_ids):
    """[(title, match indices)]: calibration, held-out and all, leaving out an empty group (and
    "all" when it would repeat the only other one)."""
    calibration = [i for i, m in enumerate(match_ids) if m in dfl.CALIBRATION]
    held_out = [i for i, m in enumerate(match_ids) if m not in dfl.CALIBRATION]
    out = [(title, indices) for title, indices in
           (("Calibration matches", calibration), ("Held-out matches", held_out)) if indices]
    if len(out) > 1:
        out.append(("All matches", list(range(len(match_ids)))))
    return out


def subset(scored, indices):
    """scored restricted to the matches at `indices`, every seed alike."""
    return {key: [[by_match[i] for i in indices] for by_match in by_seed]
            for key, by_seed in scored.items()}


def rows_to_report():
    """(pipeline, tag mode, label) for every row of the tables: smooth50 reads no tags, so it is
    run in both modes but reported once."""
    out = []
    for mode, _ in TAG_MODES:
        out += [(name, mode, mode) for name, _ in adapter.PIPELINES]
    return out + [(adapter.SMOOTH50, TAG_MODES[0][0], "n/a")]


def data_table(data, params, variants):
    ball = params["ball"]
    rows = [[row["match"] + (" (calibration)" if row["match"] in dfl.CALIBRATION else ""),
             "{:.1f}".format(row["minutes"]),
             " / ".join("{:.2f}".format(row["per_min_" + v]) for v in variants),
             "{:.1%}".format(row["dead_share"]), row["stoppages"],
             "{} / {}".format(row["set_pieces"], row["tags"]),
             "{:.1%} / {:.1%} / {:.1%}".format(row["seen"], row["seen_dead"], row["seen_alive"]),
             "{:.2f}".format(row["tracked_per_team"])] for row in data]
    # The competition's tracking, which fit.py matches the degradation to. It has no truth here, so
    # it has no truth poss/min.
    rows.append(["competition (tracking fit.py matches)", "", "no truth", "", "", "~100 / 59",
                 "{:.1%} / {:.1%}* / {:.1%}*".format(ball["seen_share_graded"],
                                                      ball["seen_share"]["dead"],
                                                      ball["seen_share"]["alive"]),
                 "{:.2f}".format(params["slots"]["tracked_per_team"])])
    checks = [[row["match"], "{:.0f} / {:.0f}".format(*row["seen_runs"]),
               "{:.0f} / {:.0f}".format(*row["unseen_runs"]), "{:.1f}".format(row["speed_p99"])]
              for row in data]
    checks.append(["competition", "{:.0f} / {:.0f}".format(ball["seen_frames"][10],
                                                           ball["seen_frames"][18]),
                   "{:.0f} / {:.0f}".format(ball["unseen_frames"][10], ball["unseen_frames"][18]),
                   "{:.1f}".format(ball["speed_p99"])])
    return ("## Data (seed 0)\n\n{}\n\n\\* on the pipeline's own stoppages and on settled play, "
            "not on a ground truth.\n\nBall stretches and speed, which were not fitted "
            "directly:\n\n"
            "{}\n".format(table(
                ["match", "minutes", "truth poss/min raw / 2 s / 3 s", "dead share", "stoppages",
                 "set pieces / tags", "ball seen: graded / dead / play",
                 "players tracked per team"], rows),
                table(["", "seen stretch frames, median / q90",
                       "unseen stretch frames, median / q90", "ball speed p99 m/s"], checks)))


def headline_table(scored, variants, n_matches, n_seeds, n_draws):
    rows = []
    for name, mode, label in rows_to_report():
        cells = [name, label]
        for variant in variants:
            by_seed = scored[(name, mode, variant)]
            low, high = score.bootstrap(by_seed, n_draws)
            cells.append("{:.3f} [{:.3f}, {:.3f}]".format(
                score.mean_summary(by_seed)["spell_f1"], low, high))
        raw = score.mean_summary(scored[(name, mode, "raw")])
        merged = score.mean_summary(scored[(name, mode, variants[-1])])
        cells += ["{:.3f}".format(raw["macro_f1"]),
                  "{:.2f} / {:.2f}".format(raw["dead_precision"], raw["dead_recall"]),
                  "{:.1f} ({:.1f})".format(raw["pred_stoppages"] / n_matches,
                                           raw["gt_stoppages"] / n_matches),
                  "{:.2f} ({:.2f} / {:.2f})".format(raw["pred_per_min"], raw["gt_per_min"],
                                                    merged["gt_per_min"])]
        rows.append(cells)
    return "## Headline: pooled over {} matches, mean of {} degradation seeds\n\n{}\n".format(
        n_matches, n_seeds, table(["pipeline", "tags"] + ["spellF1 {}".format(v) for v in variants]
                                  + ["macroF1 raw", "D precision / recall",
                                     "stoppages per match (truth)",
                                     "poss/min (truth raw / {})".format(variants[-1])], rows))


def per_match_table(scored, match_ids):
    rows = []
    for name, mode, label in rows_to_report():
        cells = [name, label]
        for index in range(len(match_ids)):
            values = []
            for variant in ("raw", "merged3s"):
                one_match = [[by_match[index]] for by_match in scored[(name, mode, variant)]]
                values.append(score.mean_summary(one_match)["spell_f1"])
            cells.append("{:.3f} / {:.3f}".format(*values))
        rows.append(cells)
    return "## Per match: spellF1, raw truth / merged 3 s\n\n{}\n".format(
        table(["pipeline", "tags"] + list(match_ids), rows))


def paired_table(scored, variants, n_seeds, n_draws):
    pairs = [(adapter.CURRENT, adapter.SMOOTH50)] + [(name, adapter.CURRENT)
                                                    for name, _ in adapter.PIPELINES[1:]]
    rows = []
    for mode, _ in TAG_MODES:
        for name, reference in pairs:
            cells = ["{} - {}".format(name, reference), mode]
            for variant in variants:
                point, low, high, above = score.paired(scored[(name, mode, variant)],
                                                       scored[(reference, mode, variant)], n_draws)
                cells.append("{:+.3f} [{:+.3f}, {:+.3f}] {:.0%}".format(point, low, high, above))
            rows.append(cells)
    return ("## Paired differences in spellF1, mean of {} seeds: [90% CI], share of draws above "
            "zero\n\n{}\n".format(n_seeds, table(["difference", "tags"] + variants, rows)))


def seeds_table(scored, n_seeds):
    rows = []
    for name, mode, label in rows_to_report():
        if name not in (adapter.CURRENT, adapter.SMOOTH50):
            continue
        values = [score.summary(by_match)["spell_f1"] for by_match in scored[(name, mode, "raw")]]
        rows.append([name, label] + ["{:.3f}".format(v) for v in values]
                    + ["{:.3f}".format(np.std(values, ddof=1))])
    return "## Degradation seeds: pooled spellF1, raw truth\n\n{}\n".format(
        table(["pipeline", "tags"] + ["seed {}".format(s) for s in range(n_seeds)] + ["sd"], rows))


def write_csv(scored, match_ids, out_dir):
    """Every summary number per seed, pipeline, tag mode, truth and match (and pooled)."""
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    path = os.path.join(out_dir, "devset_results.csv")
    keys = None
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        for (name, mode, variant), by_seed in sorted(scored.items()):
            for seed, by_match in enumerate(by_seed):
                for match_id, one in list(zip(match_ids, by_match)) + [("pooled", None)]:
                    values = score.summary(by_match if one is None else [one])
                    if keys is None:
                        keys = sorted(values)
                        writer.writerow(["seed", "pipeline", "tags", "truth", "match"] + keys)
                    writer.writerow([seed, name, mode, variant, match_id]
                                    + [values[key] for key in keys])
    return path


if __name__ == '__main__':
    main()
