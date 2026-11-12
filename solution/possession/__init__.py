"""Frame-level ball possession for More than a Hack 2027. See ../DESIGN.md for the plan.

The modules follow the pipeline stages:

    atd       stage 0  parse the raw tracking once, cache it as arrays
    ball      stage 1  clean the ball track, interpolate short gaps
    keepers   stage 1  each goalkeeper slot read as the team whose half it stands in
    features  stage 2  per-frame geometry: nearest player per team, team shape
    control   stage 2  controlled touches, and the possession they imply
    contest   stage 2  a touch the other team takes straight back in a duel does not count
    learned   stage 2b a learned P(home has the ball) on every frame, from all of the above (M4)
    deadball  stage 3  dead-ball signals and the dead-ball mask
    segment   stage 4  possessions between the stoppages, decoded from the learned evidence (M4)
    boundary  stage 5  each change of possession moved back to the touch that won the ball
    labels             graded frames, codes to labels, writing the submission
    pipeline           the stages composed into one call

    config             every tunable number
    model.json         M4's trained model (devset/train_model.py), from the DFL data alone
    runs               run-length helpers
    repo               access to games.py, possession_metrics and the baseline
"""
