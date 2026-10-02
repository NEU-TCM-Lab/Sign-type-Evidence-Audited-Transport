import copy
import json
import sys
import numpy as np
import torch
import pytest
import checkpoints
import train_readout
import all8_fuse
import late_fuse
from settings import LABELS


def test_training_is_independent_of_test_labels(tmp_path, monkeypatch):
    features = tmp_path / "features"; features.mkdir()
    runs = tmp_path / "runs"; runs.mkdir()
    rng = np.random.default_rng(3)
    for split, n in (("train", 8), ("val", 4), ("test", 4)):
        payload = {"split": split, "label_order": LABELS, "ids": [f"{split}_{i}" for i in range(n)],
                   "labels": torch.tensor(np.tile(np.arange(n)[:, None] % 2, (1, 8)), dtype=torch.float32),
                   "patch_features": torch.tensor(rng.normal(size=(n, 4, 16)), dtype=torch.float32),
                   "mask_weights": torch.ones(n, 4), "feature_dim": 16, "geometry": "square"}
        torch.save(payload, features / f"{split}_patchgrid_synthetic.pt")
    monkeypatch.setattr(train_readout, "RUNS_DIR", runs)
    for name in ("first", "changed_test_labels"):
        if name != "first":
            path = features / "test_patchgrid_synthetic.pt"
            payload = torch.load(path, weights_only=False); payload["labels"] = 1 - payload["labels"]
            torch.save(payload, path)
        monkeypatch.setattr(sys, "argv", ["train_readout.py", "--tag", "synthetic", "--readout", "softmax",
                           "--num-heads", "8", "--proj-dim", "16", "--features-dir", str(features),
                           "--run-name", name, "--epochs", "2", "--batch-size", "4", "--num-workers", "0"])
        train_readout.main()
    first = torch.load(runs / "first/best.pt", weights_only=False)
    second = torch.load(runs / "changed_test_labels/best.pt", weights_only=False)
    assert first["thresholds"] == second["thresholds"]
    assert first["model_config"] == second["model_config"]
    for name in first["model_state"]:
        torch.testing.assert_close(first["model_state"][name], second["model_state"][name], rtol=0, atol=0)


def fusion_data(shuffle=False, invert_test=False):
    y = np.tile(np.array([0, 1, 0, 1])[:, None], (1, 8))
    base = np.tile(np.array([.2, -.2, .2, -.2])[:, None], (1, 8))
    add = np.tile(np.array([-2., 2., -2., 2.])[:, None], (1, 8))
    order = [1, 0, 3, 2] if shuffle else [0, 1, 2, 3]
    test_y = 1 - y if invert_test else y
    def get(run, tag, device):
        extra = run == "add"
        ix = order if extra else [0, 1, 2, 3]
        val = {"ids": [f"v_{i}" for i in ix], "labels": y[ix], "label_order": LABELS}
        test = {"ids": [f"t_{i}" for i in ix], "labels": test_y[ix], "label_order": LABELS}
        scores = add if extra else base
        return scores[ix], y[ix], scores[ix], test_y[ix], val, test
    return get


def test_actual_all8_cli_is_invariant_to_cache_order(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["all8_fuse.py", "--base-run", "base", "--add-run", "add", "--add-tag", "x"])
    results = []
    for shuffle in (False, True):
        monkeypatch.setattr(all8_fuse, "get_logits", fusion_data(shuffle))
        all8_fuse.main(); results.append(capsys.readouterr().out)
    assert results[0] == results[1]


def test_actual_late_fusion_selection_is_independent_of_test_labels(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["late_fuse.py", "--base-run", "base", "--add-run", "add", "--add-tag", "x"])
    selected = []
    for invert in (False, True):
        monkeypatch.setattr(late_fuse, "logits", fusion_data(shuffle=True, invert_test=invert))
        late_fuse.main()
        selected.append(next(line for line in capsys.readouterr().out.splitlines() if line.startswith("Validation-selected")))
    assert selected[0] == selected[1]


def test_extra_feature_cache_is_restored_in_id_order(tmp_path, monkeypatch):
    features = tmp_path / "features"; features.mkdir()
    run = tmp_path / "runs/run"; run.mkdir(parents=True)
    (run / "input_config.json").write_text(json.dumps({"tag": "base", "extra_tag": "color", "geometry": "square"}))
    monkeypatch.setattr(checkpoints, "RUNS_DIR", tmp_path / "runs")
    labels = torch.zeros(2, 8)
    base = {"ids": ["a", "b"], "labels": labels, "label_order": LABELS, "feature_dim": 3,
            "patch_features": torch.zeros(2, 4, 3), "mask_weights": torch.ones(2, 4), "geometry": "square"}
    color = {**base, "ids": ["b", "a"], "feature_dim": 1,
             "patch_features": torch.tensor([2., 1.]).reshape(2, 1, 1).expand(2, 4, 1)}
    torch.save(base, features / "val_patchgrid_base.pt")
    torch.save(color, features / "val_patchgrid_color.pt")
    loaded = checkpoints.load_run_payload("run", "base", "val", features)
    assert loaded["feature_dim"] == 4
    torch.testing.assert_close(loaded["patch_features"][:, :, 3], torch.tensor([[1.] * 4, [2.] * 4]))


@pytest.mark.parametrize("module", ["render_base_patches", "render_c1_materials", "render_c3_coverage", "render_interp_figure", "gradcam_deletion"])
def test_figure_help_requires_no_dataset(module, monkeypatch):
    import importlib
    module = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", [module.__name__, "--help"])
    with pytest.raises(SystemExit) as result: module.main()
    assert result.value.code == 0


def test_checkpoint_input_config_survives_missing_sidecar(tmp_path, monkeypatch):
    run = tmp_path / "run"; run.mkdir()
    input_config = {"tag": "base", "extra_tag": None, "geometry": "square"}
    torch.save({"input_config": input_config}, run / "best.pt")
    monkeypatch.setattr(checkpoints, "RUNS_DIR", tmp_path)
    cache = {"ids": ["a"], "labels": torch.zeros(1, 8), "label_order": LABELS,
             "patch_features": torch.zeros(1, 4, 3), "mask_weights": torch.ones(1, 4),
             "feature_dim": 3, "geometry": "square"}
    torch.save(cache, tmp_path / "val_patchgrid_base.pt")
    assert checkpoints.load_run_payload("run", "base", "val", tmp_path)["feature_dim"] == 3
    with pytest.raises(ValueError, match="tag"):
        checkpoints.load_run_payload("run", "wrong", "val", tmp_path)


def test_corrector_selection_never_uses_test_labels(tmp_path, monkeypatch, capsys):
    import structured_corrector
    rng = np.random.default_rng(7)
    data = {"label_order": np.array(LABELS)}
    for split, probs, labels, n in (("train", "Ptr", "Ytr", 8), ("val", "Pv", "Yv", 4), ("test", "Pt", "Yt", 4)):
        data[probs] = rng.uniform(.2, .8, size=(n, 8))
        data[labels] = np.tile(np.arange(n)[:, None] % 2, (1, 8))
        data[f"{split}_ids"] = np.array([f"{split}_{i}" for i in range(n)])
    path = tmp_path / "corrector.npz"
    choices = []
    for flip in (False, True):
        current = dict(data)
        if flip: current["Yt"] = 1 - current["Yt"]
        np.savez(path, **current)
        monkeypatch.setattr(sys, "argv", ["structured_corrector.py", "--npz", str(path), "--epochs", "2"])
        structured_corrector.main()
        choices.append(next(line for line in capsys.readouterr().out.splitlines() if line.startswith("Validation-selected")))
    assert choices[0] == choices[1]
