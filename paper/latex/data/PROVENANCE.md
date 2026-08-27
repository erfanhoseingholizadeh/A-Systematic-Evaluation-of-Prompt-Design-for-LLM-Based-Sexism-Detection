# Data provenance for `paper/latex/data/`

These CSVs are derived, paper-ready summaries computed from the raw
per-model result files synced from the compute cluster (`results/` at the
repo root, itself pulled from the cluster's `sexism-study/results/`
directory on 2026-08-24). Every
number in `sections/results.tex` traces back to one of these files or to
a raw `results/*.csv` file directly. Regenerate by re-running the
extraction script used to build them (see CHANGELOG.md's entry for this
session if it needs to be reconstructed).

## QLoRA row selection (`baseline_comparison.csv`)

`results/finetune_qlora_phi3.csv` and `results/finetune_qlora_qwen2.5.csv`
each contain **two rows**, not one — an artifact of the QLoRA epoch
re-selection work (see HANDOFF.md / CHANGELOG.md, "epoch-bump
significance" investigation). The first row in each file is a stale
evaluation from an earlier epoch configuration; the second (last) row is
the final dev-selected-epoch result that the project's own significance
testing was run against. `baseline_comparison.csv` was built by taking
`.iloc[-1]` (the last row) for every model, which for phi3 and qwen2.5
recovers 0.734 and 0.786 F1 respectively — matching the numbers already
documented as final in HANDOFF.md. Do not use the first row for either
model; it is not the reported baseline.

## Confidence-based AUC (`tab:auc` in `sections/results.tex`, Sec.~V-I)

Not built from a file in this directory: the source is
`results/confidence_capture_by_model.csv` (per-model F1-best
non-chain-of-thought variant, produced by
`scripts/aggregate_confidence_capture.py` from the real 42-unit cluster
run) joined against `results/finetune_classical.csv`'s `auc` column
(0.848, from `classical.evaluate_classical_baseline`'s `predict_proba`).
Both are raw `results/*.csv` files, not curated copies here, per this
file's opening paragraph. Qwen3's row uses V7, not its true best-F1
variant V8, because V8 is chain-of-thought and this method cannot score
it; `results/confidence_capture_by_model.csv`'s own
`covers_true_best_variant` column flags this explicitly. Regenerate with
`python scripts/aggregate_confidence_capture.py` after re-syncing
`checkpoints/confidence/` from the cluster.

## Best/worst variant selection (error-by-subtype and sensitivity files)

"Best variant" per model is the argmax of `f1` in that model's main-grid
`results/<model>.csv`. "Worst variant" (`error_by_subtype_worst_variant.csv`
only) is the argmax of mean `fn_rate` across the four sexism subtypes
(`none` excluded — false-negative rate is undefined for the not-sexist
category). llama3.1's V14 is excluded from all of this by construction:
it never appears in `results/llama3.1.csv` (fail-ratio gate), so it can't
be selected as best or worst and doesn't appear in any curated file here.
It is reported separately in the paper's Sec. V-C / Table III and the
project's supplementary material.
