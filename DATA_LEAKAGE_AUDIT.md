# Data leakage and release audit

Audit date: 2026-10-02. Original snapshot: `263c8bd5f2d7a23df340158db62f52713ade9832`.

## Confirmed code-level defects and repairs

- `late_fuse.py` formerly chose the best fusion coefficient by test F1. It now selects coefficients and thresholds from validation data and reports only the fixed selected test result.
- `structured_corrector.py` formerly fitted on validation data and chose the winning variant by test F1. It now requires training predictions, fits on training labels, computes its co-occurrence initialization from training labels, selects the variant and thresholds on validation, and scores the selected model on test.
- The aggregate-alpha selection in `ot_review.py` formerly used test macro-F1. It now uses validation macro-F1. Its heatmap and deletion analyses remain descriptive held-out audits, not sources for fitting the classifier.
- Fusion formerly relied on cache row order. It now aligns by unique sample IDs and rejects identity-set or ground-truth mismatches. Such mismatches are an evaluation correctness issue; they are not proof of actual dataset leakage.
- Reconstructed models formerly lost settings such as multi-head count and color peak detection. Complete settings are now saved and loaded strictly. This fixes reproducibility; it cannot retroactively validate scores produced by an older evaluation path.

## Training boundary checked

The readout loop uses training labels for loss gradients, positive-class weights, logit-adjustment priors, co-occurrence cost and empirical frequency targets. Validation selects epochs and thresholds. Test scoring occurs after loading the selected checkpoint. The pooled-head loop also fits on training labels and selects checkpoints on validation.

A regression test runs training twice with identical training/validation data but inverted test labels. The resulting checkpoint tensors, model configuration and selected thresholds must be identical. Further tests check that test-label changes do not change the late-fusion coefficient selection.

Frozen backbone caching processes each split separately, with fixed pretrained preprocessing and per-image computations. Saving a held-out label for later evaluation is not itself training leakage. External prediction archives, model pretraining data, pseudo-annotation generation and the historical human experiment-selection process are outside what this code snapshot can verify.

## Dataset contamination protection

- Training rejects cross-split sample IDs and available image paths, original paths, image hashes and subject/group IDs.
- Manifest auditing checks duplicate identities and paths, exact image-byte duplicates, and group overlap when validated patient/subject metadata is supplied.
- Generated caches preserve available identity/hash/group metadata; legacy caches must be regenerated to acquire missing metadata.
- A patient may have several images within one split, but a known group cannot occur in two splits.
- Missing patient metadata is reported as unverified. Exact-byte hashes do not rule out visually similar, re-encoded or near-duplicate images; a separate image/patient audit is still required on the actual data.

Run on the real dataset:

```bash
python src/build_manifest.py --hash-images --group-field patient_id
python src/audit_dataset.py --hash-images
```

Replace `patient_id` with the verified field in the source annotations or CSV. The generated manifest uses `group_id`. If this metadata does not exist, do not claim patient-independent splits.

## Privacy and public release

The working release and locally available original Git history were scanned for common credential patterns, sample-level JSON/CSV artifacts, raw-image/binary archives and model/feature files. The original snapshot contains aggregate results; the scan did not flag credentials or sample-level data files. The scanner is heuristic and is not a proof that all possible PII is absent.

Sample-level prediction export is now opt-in. Teacher/heatmap archives remain local outputs that may contain sample IDs and labels. Git ignores generated artifacts, outputs, datasets, models, manifests and environment files. Intentional force-add or publishing files through another channel still requires review.

```bash
python src/audit_release.py --history
```

No code path in the inspected repository explicitly sends sample images or labels to an external API. Downloading pretrained model files is separate from local feature inference; third-party dependency and model behavior require their own audit.

## Verification and limits

Verification covers real CPU readout checkpoint round-trips, synthetic training, modality/ID alignment, split-overlap detection, patient-group and byte-hash checks, color patch geometry, release-scanner behavior, command help entry points and Bash syntax. Tests use synthetic data only.

The repaired version passed 37 regression tests, all 33 command help entry points and syntax checks for seven Bash scripts. All 94 Python files parsed and compiled. The test environment used Python 3.12 and CPU PyTorch; full backbone extraction and the paper experiments were not run.

Raw TongueDx images, patient identity mappings, original feature caches, checkpoints, full experiment logs and external baseline prediction archives were not provided. Therefore this audit cannot determine whether the historical dataset actually has patient overlap or whether all published scores were obtained without test-driven manual tuning. Archived `results/` files were not changed or recomputed. A new independent evaluation on audited splits is required before making a stronger dataset-level or paper-level claim.
