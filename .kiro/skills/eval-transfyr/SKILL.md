---
name: eval-transfyr
description: Run and report evaluation on the Transfyr pipette well dataset. Use after any change to localizer/, tracking/, or dispense/, or when asked how well the model performs.
---

# Evaluating on the Transfyr dataset

1. Run: `python scripts/eval_transfyr.py --split val --view topview`
   (use `--view fpv` or `--view both` only when asked).
2. Never run on `--split test` unless the user explicitly asks.
3. Read `runs/<latest>/metrics.json` and report a short table:
   - exact-set accuracy (predicted well set == ground-truth set)
   - per-well precision / recall
   - off-by-one rate (row or column off by 1) — main expected failure mode
   - multichannel clip accuracy, single-channel clip accuracy (separately)
   - mean latency per frame
4. Compare against the previous run in `runs/` and state clearly what improved or got worse.
5. List the 5 worst clips by clip_id with predicted vs. ground-truth wells, so failures can be
   inspected. Do not paste or save frames from the dataset anywhere outside `runs/`.
