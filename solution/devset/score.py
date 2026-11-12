"""Score dev-set predictions with the competition's own metric, per match and over all of them.

Everything goes through evaluation/possession_metrics.py; nothing here re-defines a metric. The only
additions are how matches are combined and how uncertain the result is:

  pooled     possessions of every match matched one-to-one within their own match (they cannot
             overlap across matches), then one F1 per IoU threshold from the summed counts -- the
             same number pm.evaluate would give on the matches laid end to end with X between.
  bootstrap  two levels, because two kinds of luck move the score: which matches we happen to have
             (resampled with replacement), and which possessions inside them (resampled within each
             drawn match, as pm.bootstrap_spell_f1 does on one match). With two matches the first
             level has three distinct outcomes, so the interval mostly reflects possessions; it
             becomes meaningful as matches are added.
  paired     the difference between two pipelines under the same draws, which is what decides
             whether a change is real -- far narrower than comparing two intervals.
  seeds      the degradation is random, and its seed alone moves pooled spellF1 by about 0.008
             (sd over five seeds), as much as a small change to the pipeline. So every number
             here can be averaged over seeds: point estimates are the mean over seeds, and each
             bootstrap draw resamples the same matches and possessions for every seed (the truth
             does not depend on the seed) and averages over them.

Arguments named `by_seed` are lists over seeds of lists over matches of Scored.
"""

import numpy as np

from possession.repo import pm

PERCENTILES = (5, 95)


class Scored(object):
    """One prediction against one truth: the per-possession match flags behind spellF1, the
    confusion matrix behind macroF1, and the counts worth printing."""

    def __init__(self, gt, pred):
        runs = pm.valid_runs(gt)
        filled = pm.fill_abstentions(pred, runs)
        gt_spells = pm.possession_spells(gt, runs)
        pred_spells = pm.possession_spells(filled, runs)
        overlaps = pm.spell_overlaps(gt_spells, pred_spells)
        self.flags = np.vstack([pm.matched_gt_flags(overlaps, len(gt_spells), threshold)
                                for threshold in pm.SPELL_IOUS])  # (n_ious, n_gt)
        self.n_gt, self.n_pred = len(gt_spells), len(pred_spells)
        self.cm3 = pm.scored_confusion(pm.confusion(gt, pred))
        self.minutes = sum(stop - start for start, stop in runs) / float(pm.FPS) / 60.0
        segments = pm.segments(filled, runs)
        self.pred_stoppages = sum(code == pm.D for _, _, code in segments)
        self.gt_stoppages = sum(code == pm.D for _, _, code in pm.segments(gt, runs))


def spell_f1(true_positive, n_gt, n_pred):
    """Mean over IoU thresholds of F1 from counts; true_positive is one count per threshold."""
    f1s = [2.0 * tp / (2 * tp + max(0, n_pred - tp) + (n_gt - tp)) if tp else 0.0
           for tp in true_positive]
    return float(np.mean(f1s))


def summary(scored):
    """Pooled metrics over a list of Scored (one per match)."""
    true_positive = np.sum([s.flags.sum(axis=1) for s in scored], axis=0)
    n_gt = sum(s.n_gt for s in scored)
    n_pred = sum(s.n_pred for s in scored)
    cm3 = np.sum([s.cm3 for s in scored], axis=0)
    minutes = sum(s.minutes for s in scored)
    precision, recall, _ = pm.per_class_prf(cm3, pm.D)
    return {
        "spell_f1": spell_f1(true_positive, n_gt, n_pred),
        "macro_f1": pm.macro_f1(cm3),
        "dead_precision": precision,
        "dead_recall": recall,
        "pred_per_min": n_pred / minutes,
        "gt_per_min": n_gt / minutes,
        "n_pred": n_pred,
        "n_gt": n_gt,
        "pred_stoppages": sum(s.pred_stoppages for s in scored),
        "gt_stoppages": sum(s.gt_stoppages for s in scored),
    }


def mean_summary(by_seed):
    """summary() averaged over seeds, key by key."""
    summaries = [summary(scored) for scored in by_seed]
    return {key: float(np.mean([one[key] for one in summaries])) for key in summaries[0]}


def _draws(scored_lists, n_draws, seed):
    """spellF1 per draw for each list of per-match Scored, all under the same resampling."""
    rng = np.random.RandomState(seed)
    n_matches = len(scored_lists[0])
    out = np.empty((len(scored_lists), n_draws))
    for draw in range(n_draws):
        picks = rng.randint(0, n_matches, n_matches)
        inner = [rng.randint(0, scored_lists[0][m].n_gt, scored_lists[0][m].n_gt) for m in picks]
        n_gt = sum(len(rows) for rows in inner)
        for index, scored in enumerate(scored_lists):
            true_positive = sum(scored[m].flags[:, rows].sum(axis=1)
                                for m, rows in zip(picks, inner))
            n_pred = sum(scored[m].n_pred for m in picks)
            out[index, draw] = spell_f1(true_positive, n_gt, n_pred)
    return out


def bootstrap(by_seed, n_draws=1000, seed=0):
    """(low, high) of seed-averaged pooled spellF1 under the two-level bootstrap."""
    draws = _draws(by_seed, n_draws, seed).mean(axis=0)
    low, high = np.percentile(draws, PERCENTILES)
    return float(low), float(high)


def paired(a_by_seed, b_by_seed, n_draws=1000, seed=0):
    """(difference, low, high, share of draws where a beats b) for seed-averaged spellF1 of a minus
    b; both must hold the same matches against the same truth, in the same order."""
    n_seeds = len(a_by_seed)
    draws = _draws(list(a_by_seed) + list(b_by_seed), n_draws, seed)
    difference = draws[:n_seeds].mean(axis=0) - draws[n_seeds:].mean(axis=0)
    low, high = np.percentile(difference, PERCENTILES)
    point = mean_summary(a_by_seed)["spell_f1"] - mean_summary(b_by_seed)["spell_f1"]
    return point, float(low), float(high), float(np.mean(difference > 0))
