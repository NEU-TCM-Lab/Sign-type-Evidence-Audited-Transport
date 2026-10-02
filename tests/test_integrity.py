import copy
import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image

import checkpoints
import train_readout
from settings import LABELS
from data_validation import alignment_order, validate_split_payloads, audit_manifest_splits, relative_dataset_path
from fusion import select_fusion, evaluate_selected, validate_prediction_dump
from geometry import resize_image


def payload(split, ids):
    n = len(ids)
    return {"split": split, "ids": ids, "label_order": LABELS,
            "labels": torch.tensor(np.tile([0, 1], (n, 4)), dtype=torch.float32)}


@pytest.mark.parametrize("options", [
    ["--readout", "softmax", "--num-heads", "8"],
    ["--readout", "ot", "--num-heads", "4"],
    ["--readout", "uot", "--uot-rho", "0.4"],
    ["--readout", "mpsa"], ["--readout", "mlp"],
    ["--readout", "otgen", "--otgen-combine", "concat", "--peak-topk", "3"],
    ["--readout", "dualot", "--dino-dim", "8", "--peak-topk", "3"],
    ["--readout", "signot", "--evidence", "aot", "--ot-relax", "partial", "--partial-m", "0.6"],
])
def test_checkpoint_restores_same_logits(tmp_path, monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["train_readout.py", "--tag", "synthetic", "--proj-dim", "16", "--sinkhorn-iters", "7"] + options)
    args = train_readout.parse_args()
    config = checkpoints.model_config_from_args(args, 16)
    model = checkpoints.build_readout_model(config).eval()
    x, mask = torch.randn(2, 9, 16), torch.rand(2, 9)
    with torch.no_grad(): expected = model(x, mask)
    run = tmp_path / "run"; run.mkdir()
    torch.save({"label_order": LABELS, "model_config": config, "model_state": model.state_dict()}, run / "best.pt")
    monkeypatch.setattr(checkpoints, "RUNS_DIR", tmp_path)
    restored, _ = checkpoints.load_trained_model("run", "cpu", 16)
    with torch.no_grad(): actual = restored(x, mask)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_text_queries_restore_without_source_embedding_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_readout.py", "--tag", "x", "--readout", "softmax", "--proj-dim", "16"])
    args = train_readout.parse_args(); query = torch.randn(8, 12)
    cfg = checkpoints.model_config_from_args(args, 16, query)
    model = checkpoints.build_readout_model(cfg, query).eval()
    run = tmp_path / "run"; run.mkdir()
    torch.save({"label_order": LABELS, "model_config": cfg, "model_state": model.state_dict()}, run / "best.pt")
    monkeypatch.setattr(checkpoints, "RUNS_DIR", tmp_path)
    restored, _ = checkpoints.load_trained_model("run", "cpu")
    torch.testing.assert_close(model.query_emb, restored.query_emb)
    x, mask = torch.randn(2, 9, 16), torch.ones(2, 9)
    with torch.no_grad():
        torch.testing.assert_close(model(x, mask), restored(x, mask), rtol=0, atol=0)


def test_ambiguous_legacy_checkpoint_rejected(tmp_path, monkeypatch):
    run = tmp_path / "run"; run.mkdir()
    torch.save({"label_order": LABELS, "model_state": {}}, run / "best.pt")
    monkeypatch.setattr(checkpoints, "RUNS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Legacy checkpoint"):
        checkpoints.load_trained_model("run", "cpu")


def test_id_alignment_is_order_invariant_and_validates_labels():
    ref = payload("val", ["a", "b", "c"])
    other = copy.deepcopy(ref); other["ids"] = ["c", "a", "b"]
    other["labels"] = ref["labels"][[2, 0, 1]]
    assert alignment_order(ref, other) == [1, 2, 0]
    other["labels"][0, 0] = 1
    with pytest.raises(ValueError, match="ground-truth"):
        alignment_order(ref, other)


@pytest.mark.parametrize("field", ["ids", "image_paths", "original_relative_paths", "image_sha256", "group_ids"])
def test_training_refuses_cross_split_overlap(field):
    train, val = payload("train", ["t"]), payload("val", ["v"])
    train[field] = val[field] = ["same"]
    with pytest.raises(ValueError, match="Cross-split"):
        validate_split_payloads({"train": train, "val": val})


def test_fusion_selection_does_not_depend_on_test_labels():
    y = np.tile(np.array([0, 1, 0, 1])[:, None], (1, 8))
    base = np.tile(np.array([.2, -.2, .2, -.2])[:, None], (1, 8))
    extra = np.tile(np.array([-2., 2., -2., 2.])[:, None], (1, 8))
    selected = select_fusion(base, extra, y)
    before = copy.deepcopy(selected)
    good = evaluate_selected(base, extra, y, selected)
    bad = evaluate_selected(base, extra, 1 - y, selected)
    assert selected == before and good != bad
    assert any(cfg["gamma"] > 0 for cfg in selected)


@pytest.mark.parametrize("value", ["/server/data.jpg", "C:\\server\\data.jpg", "../data.jpg", "images/../../data.jpg"])
def test_nonportable_dataset_paths_rejected(value):
    with pytest.raises(ValueError): relative_dataset_path(value)


def manifest_row(i, group=None):
    row = {"id": str(i), "image_path": f"{i}.png", "mask_path": f"mask_{i}.png", "label_order": LABELS, "labels": [0, 1] * 4}
    if group: row["group_id"] = group
    return row


def test_patient_overlap_and_image_duplicates_detected(tmp_path):
    rows = {"train": [manifest_row(1, "patient")], "test": [manifest_row(2, "patient")]}
    with pytest.raises(ValueError, match="group_id"): audit_manifest_splits(rows)
    rows["test"][0]["group_id"] = "other"
    for i in (1, 2): Image.new("RGB", (4, 4), "red").save(tmp_path / f"{i}.png")
    with pytest.raises(ValueError, match="image_sha256"): audit_manifest_splits(rows, tmp_path, True)


def test_missing_group_metadata_is_reported_unverified():
    result = audit_manifest_splits({"train": [manifest_row(1)], "test": [manifest_row(2)]})
    assert result["group_metadata_complete"] is False


def test_aspect_padding_matches_image_and_mask_geometry():
    image = Image.new("RGB", (40, 20), "white")
    mask = Image.new("L", (40, 20), 255)
    rgb = np.asarray(resize_image(image, 20, "arpad"))
    weights = np.asarray(resize_image(mask, 20, "arpad"))
    assert np.array_equal(rgb[:, :, 0] > 0, weights > 0)
    assert (weights[:5] == 0).all() and (weights[5:15] == 255).all()


def test_color_extremes_respects_patch_locations(tmp_path):
    from cache_color_features import color_grid, GRID, PATCH
    colors = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    colors[:, :, 0] = np.arange(GRID)[:, None] * 6
    colors[:, :, 1] = np.arange(GRID)[None, :] * 6
    rgb = np.repeat(np.repeat(colors, PATCH, axis=0), PATCH, axis=1)
    path = tmp_path / "patches.png"; Image.fromarray(rgb).save(path)
    features = color_grid(path, extremes=True)
    np.testing.assert_allclose(features[:, 8], features[:, 0], atol=1e-3)
    np.testing.assert_allclose(features[:, 9], features[:, 2], atol=1e-3)
    np.testing.assert_allclose(features[:, 10], features[:, 1], atol=1e-3)
    np.testing.assert_allclose(features[:, 11], features[:, 4], atol=1e-3)


def test_release_scanner_detects_credentials_and_private_artifacts(tmp_path):
    from audit_release import scan_files
    (tmp_path / "secret.txt").write_text("gh" + "p_" + "x" * 36)
    (tmp_path / "private.json").write_text(json.dumps({"patient_id": "synthetic"}))
    (tmp_path / "metrics.json").write_text(json.dumps({"macro": {"f1": 0.5}}))
    findings = scan_files(tmp_path, ["secret.txt", "private.json", "metrics.json"])
    assert {finding["file"] for finding in findings} == {"secret.txt", "private.json"}
