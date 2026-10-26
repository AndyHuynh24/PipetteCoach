# Pipette Coach

Hackathon project (Hackachusetts, Biotech & Healthcare track). Goal: a fixed top-view webcam
watches someone pipetting by hand into a 96-well plate, and the app figures out which well(s)
they dispensed into, so it can catch wrong, skipped, or double-filled wells and log the run.

## Setup facts
- Live input: one fixed, mounted top-view webcam. No custom hardware.
- Data: Transfyr pipette_well_challenge dataset in `data/raw/` (Topview + FPV clips, `labels.json`
  with dispensed wells, including multichannel).

## You choose the approach
- Architecture, models, methods, and libraries are open. Explore the data first, then propose
  2–3 options with tradeoffs (accuracy, speed, effort, demo risk) and a recommendation.
- Prefer quick experiments that answer a question over big builds. Start with the simplest
  thing that could work, measure it, then iterate.
- Wait for my approval before committing to an approach or making changes across many files.
- Report results honestly, including when something got worse or didn't work.

## Data rules
- `data/raw/` is READ ONLY. Never edit, move, rename, or delete raw clips or `labels.json`.
- Derived data (frames, crops, annotations, features) goes in `data/processed/` and must be
  regenerable by a script.
- Split train/val/test BY PLATE (`Plate_<n>_clip_<m>`), never by clip — clips from the same plate
  share camera pose and lighting, so clip-level splits leak. Save the split to
  `configs/splits.json`; don't change it once created without asking.
- Tune and compare on val only. Run on test only when I ask.
- The dataset has no license: never commit it, redistribute it, upload it anywhere, or put its
  frames/clips in the README, slides, or demo. My own recordings go in `data/own/`.

## Files & outputs
- Experiment outputs, metrics, and plots go in `runs/<timestamp>_<short-name>/`, each with a
  `metrics.json` and a short `notes.md` (what was tried, result, takeaway).
- Model weights and checkpoints go in `models/` (gitignored). Note in `notes.md` how to regenerate them.
- Don't delete files you didn't create. Don't create new top-level folders without saying why.
- Seed randomness in scripts so results are reproducible.
- Keep a short `docs/DECISIONS.md` log: date, decision, why, alternatives considered.

## Git rules
- Never commit: `data/raw/`, `data/processed/`, `runs/`, `models/`, `.env`, API keys,
  or any file > 10 MB.
- Small, focused commits with clear messages (e.g. `feat: ...`, `fix: ...`, `exp: ...`).
- Use a branch per experiment or feature (`exp/<name>`, `feat/<name>`); keep `main` runnable.
- Never force-push, rewrite history, or push to a remote without asking.
- Check `git status` before committing; don't stage unrelated changes.

## Secrets & dependencies
- API keys only from environment variables (e.g. `ANTHROPIC_API_KEY`). Never print, log,
  or commit them.
- Pin dependency versions in `requirements.txt`. Say why before adding a new dependency.

## Working style
- Ask instead of guessing when something about the data, labels, or lab procedure is unclear.
- Prefer small diffs; don't refactor unrelated code.
- Any reusable code gets at least a basic test (`python -m pytest -q`).