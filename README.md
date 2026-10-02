# SEAT: Sign-Type Evidence-Audited Transport

Research code for frozen vision features, sign-specific readout heads, validation-calibrated fusion, and evidence audits. Run the modules in `src/`. The `seat/` tree is an identical copy grouped for code review, not an importable Python package.

## Install

Use Python 3.11 or newer. Install a PyTorch/torchvision build appropriate for your hardware, then:

```bash
python -m pip install -r requirements.txt
# Regression tests
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests use synthetic inputs and CPU models. Full dataset training and large-backbone extraction require your local data, model access and suitable hardware; the committed historical scores have not been recomputed after these fixes.

## Local paths

No original server directory is required. Defaults are relative to this repository, independent of the working directory. Set overrides before starting a command:

| Environment variable | Default |
| --- | --- |
| `SEAT_DATASET_ROOT` | `data/` |
| `SEAT_LABEL_ROOT` | `<dataset root>/list/` |
| `SEAT_MODEL_PATH` | `models/Qwen3-VL-4B-Instruct/` |
| `SEAT_ARTIFACTS_DIR` | `artifacts/` |
| `SEAT_MANIFEST_DIR` | `<artifacts>/manifests/` |
| `SEAT_FEATURES_DIR` | `<artifacts>/features/` |
| `SEAT_RUNS_DIR` | `<artifacts>/runs/` |
| `SEAT_OUTPUTS_DIR` | `outputs/` |
| `SEAT_FIGURES_DIR` | `<outputs>/figures/` |
| `SEAT_RESNET_CHECKPOINT` | `<runs>/resnet34_pseudo_seed42/best.pt` |
| `SEAT_PYTHON` | `python`, used by Bash experiment scripts |

Relative overrides resolve against the repository root. Data-manifest and cache commands also accept `--dataset-root`, `--manifest-dir`, and `--features-dir`. Manifest construction accepts `--label-root`; Qwen extraction accepts `--model-path`.

## Dataset interface and split audit

Raw data, masks, annotation manifests, model weights and sample-level predictions are not distributed in this repository. Supply data you are authorized to use. This release consumes existing pseudo masks and boxes; it does not reproduce the external Qwen/SAM2 pseudo-annotation generation pipeline.

Required label CSVs are `train_fold1.csv`, `val_fold1.csv` and `test.csv`. Each row contains a relative `image_path` and eight binary columns in this order: TonguePale, TipSideRed, Spot, Ecchymosis, Crack, Toothmark, FurThick, FurYellow. The original protocol has 3371/843/895 rows; manifest construction checks those counts.

The dataset root must contain `annotations/{train,val,test}_pseudo.json`. Each file is a JSON list with `id`, `original_relative_path` (the CSV key), `image_path`, `pseudo_mask_path`, `sam2_bbox_px` and `qwen_bbox_px`. Image and mask paths must be relative to the dataset root and images/masks must have identical dimensions. If a patient/subject field is available in annotations or the label CSV, copy it into the generated manifest as `group_id`:

```bash
python src/build_manifest.py --hash-images --group-field patient_id
python src/audit_dataset.py --hash-images
```

Use the actual group-field name from your data. If group metadata is unavailable, omit `--group-field`; the audit explicitly reports that patient-level separation remains unverified. Never infer patient identity from filenames without a validated mapping. Hashing checks exact file duplicates; it does not detect all re-encoded or near-duplicate images.

Training checks cross-split IDs and available image paths, hashes and group IDs before optimization. Cached labels must agree after ID alignment. The image-hash and patient checks must also be run on the actual dataset before reporting independent held-out performance.

## Feature extraction and training

```bash
python src/cache_patchgrid_features.py --backbone dinov2 --tag dinov2grid_seed42
python src/cache_color_features.py --tag color_seed42
python src/train_readout.py --tag dinov2grid_seed42 --readout softmax --num-heads 8 --selection-metric macro_f1 --run-name baseline
```

`--geometry square` preserves the original resize protocol. For aspect-preserving padding, use `--geometry arpad` for BOTH image and mask caches and choose a distinct tag. Cache metadata records geometry. Merely naming a cache `arpad` does not change its preprocessing. `scripts/run_arpad_experiment.sh` now builds actual padded caches.

Bash experiment scripts locate `src/` in the current checkout, stop on errors, and do not delete existing runs. Runs use `exist_ok=False`; choose new run names or archive previous outputs deliberately. Partial caches are not silently overwritten; complete/remove them deliberately or use the cache command's explicit `--overwrite` flag.

## Checkpoints and fusion

New readout checkpoints include complete constructor settings, including head count, Sinkhorn iterations, peak detection, text queries and feature inputs. Evaluation loads weights strictly and reconstructs concatenated modalities in sample-ID order. A legacy checkpoint missing non-tensor settings is rejected. Retrain it, or supply a verified version-2 `model_config.json` made with `model_config_from_args` using the original training arguments. For concatenated modalities also supply `input_config.json`; unknown settings must not be guessed from weight shapes.

Use `fuse_npz.py`, `all8_fuse.py` or `late_fuse.py` to fit fusion coefficients and thresholds using validation labels. Test labels are used only to score the fixed selection. Prediction NPZ files require label order, logits, labels and IDs for validation/test; training fields are checked when present. Old archives with no identities or Python object arrays must be regenerated.

The structured corrector requires `Ptr/Ytr`, `Pv/Yv`, `Pt/Yt`, `train_ids`, `val_ids`, `test_ids` and `label_order`. It fits on training predictions, computes co-occurrence from training labels, chooses its variant/thresholds on validation, and evaluates the chosen model on test. Prefer out-of-fold training predictions or an independent calibration-fit set for stacking. The origin of external prediction archives and pretrained weights must be audited separately.

## Local output and public-release checks

`evaluate_head.py` writes aggregate metrics by default. Use `--export-predictions` only when you need local sample-level CSVs. Heatmap and teacher dumps also contain sample information; keep them local. Generated `artifacts/`, `outputs/`, datasets, model files, manifests and environment files are ignored by Git.

```bash
python src/audit_release.py --history
python scripts/sync_review_copy.py --check
```

The release scanner checks common credential patterns and sample-level JSON/CSV/binary artifacts in the working release and locally available Git history. It is a heuristic check, not a guarantee that all sensitive information is absent. Review manually before publishing additional files.

`results/` contains archived aggregate CSV/JSON results from the original snapshot. They are retained for provenance and are not new validation of this repaired version. See `DATA_LEAKAGE_AUDIT.md` for the audit boundary.
