"""A development set from the open DFL / Sportec matches: the pipeline scored on far more than the
17 public possessions. See README.md for results and caveats.

    dfl         parse one DFL match (positions streamed, events, match information), cache arrays
    truth       H / A / D ground truth from BallStatus and BallPossession; merged variants
    degrade     make the clean DFL tracking look like the competition's, as an atd.Tracking
    fit         fit degrade's parameters from the competition tracking into params.json
    tags        a Smart Tagging stand-in from DFL set-piece events, degraded like the real tags
    adapter     run possession.pipeline.label_frames unchanged on a dev match
    baselines   baseline_smooth50 on a Tracking
    score       the competition's metric pooled over matches, with a two-level bootstrap
    run_devset  the one command
"""
