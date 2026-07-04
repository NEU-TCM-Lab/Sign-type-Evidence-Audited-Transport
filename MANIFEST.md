# Release Candidate Manifest

## Source Snapshot

Primary code source:

```text
/root/autodl-tmp/TongueDx2_Dinov2_V14/scripts
```

Primary result source:

```text
/root/autodl-tmp/TongueDx2_Dinov2_V14/outputs
```

Baseline metric sources:

```text
/root/autodl-tmp/TongueDx2_Qwen3VL4B_stage4A_V10_OT_macrof1/artifacts/runs/stage4A_v10_noot_macrof1_fulltrain_seed42/metrics_test.json
/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls/artifacts/runs/mt768_fulltrain_seed42_global_mask/metrics_test.json
```

## Core Code

```text
src/common.py
src/metrics.py
src/losses.py
src/model.py
src/train_readout.py
src/train_head.py
src/evaluate_head.py
src/ensemble_eval.py
src/late_fuse.py
src/all8_fuse.py
src/cross_seed_fusion.py
src/paired_stage_eval.py
```

## Feature Cache and Manifest Code

```text
src/build_manifest.py
src/cache_backbone_features.py
src/cache_patchgrid_features.py
src/cache_color_features.py
src/cache_texture_features.py
src/cache_pooled_features.py
src/merge_feature_shards.py
```

## OT, C1, C2, and Audit Code

```text
src/ot_readout_head.py
src/ot_sign_head.py
src/ot_gen_head.py
src/ot_explain.py
src/ot_review.py
src/structured_corrector.py
src/logit_adjust.py
src/color_counterfactual.py
src/color_counterfactual_multi.py
src/color_faithfulness.py
src/gradcam_deletion.py
src/resnet_gradcam.py
src/render_interp_figure.py
src/render_base_patches.py
src/render_c1_materials.py
src/render_c3_coverage.py
```

## Pulled Result Files

```text
results/v14/per_class_v14.csv
results/v14/per_class_v14_paired_stages.csv
results/v14/paired_stage_seed42_47.csv
results/v14/paired_stage_seed42_47.json
results/v14/cross_seed_old6_logitmean_color_g15.json
results/v14/cross_seed_old6_logitmean_color_g15_per_class.csv
results/v14/cross_seed_fusion_summary.csv
results/v14/gradcam_deletion_Toothmark.json
results/baselines/qwen3vl4b_lora/metrics_test.json
results/baselines/shizhengpt_7b_vl/metrics_test.json
```
