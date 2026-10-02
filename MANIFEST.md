# Release provenance

The original code and aggregate result snapshot came from `Pig-uncle/Sign-type-Evidence-Audited-Transport`, commit `263c8bd5f2d7a23df340158db62f52713ade9832`, and was copied to `NEU-TCM-Lab/Sign-type-Evidence-Audited-Transport` with its history intact.

The repaired release uses repository-relative/configurable paths. The old machine-specific locations are not runtime prerequisites. `src/` is the canonical runnable source; `seat/` is synchronized by `scripts/sync_review_copy.py` for review by role.

`results/v14/` retains the original per-class, cross-seed, paired-stage and deletion aggregate artifacts. `results/baselines/` retains the Qwen LoRA and ShizhenGPT aggregate metric snapshots. These files have not been regenerated and do not establish that the original experimental protocol was free of leakage.

No raw images, masks, patient identity tables, sample-level prediction archives, checkpoints, feature caches or training logs were available in the source snapshot. Dataset-level and patient-level conclusions require the original data and metadata.
