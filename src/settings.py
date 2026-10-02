"""Local paths. Environment overrides must be set before starting a command."""
from pathlib import Path
import os

_file = Path(__file__).resolve()
PROJECT_ROOT = _file.parents[1] if _file.parent.name == "src" else _file.parents[2]


def configured_path(name: str, default: Path) -> Path:
    path = Path(os.environ.get(name, str(default))).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


DATASET_ROOT = configured_path("SEAT_DATASET_ROOT", PROJECT_ROOT / "data")
LABEL_ROOT = configured_path("SEAT_LABEL_ROOT", DATASET_ROOT / "list")
MODEL_PATH = configured_path("SEAT_MODEL_PATH", PROJECT_ROOT / "models" / "Qwen3-VL-4B-Instruct")
ARTIFACTS_DIR = configured_path("SEAT_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
MANIFEST_DIR = configured_path("SEAT_MANIFEST_DIR", ARTIFACTS_DIR / "manifests")
FEATURES_DIR = configured_path("SEAT_FEATURES_DIR", ARTIFACTS_DIR / "features")
RUNS_DIR = configured_path("SEAT_RUNS_DIR", ARTIFACTS_DIR / "runs")
OUTPUTS_DIR = configured_path("SEAT_OUTPUTS_DIR", PROJECT_ROOT / "outputs")
FIGURES_DIR = configured_path("SEAT_FIGURES_DIR", OUTPUTS_DIR / "figures")
RESNET_CHECKPOINT = configured_path("SEAT_RESNET_CHECKPOINT", RUNS_DIR / "resnet34_pseudo_seed42" / "best.pt")
LABELS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis", "Crack", "Toothmark", "FurThick", "FurYellow"]
