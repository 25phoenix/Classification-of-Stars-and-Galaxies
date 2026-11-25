from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DEFAULT_SEED = 42
NUM_CLASSES = 10
ALL_CLASS_IDS = list(range(NUM_CLASSES))

GALAXY10_DECALS_CLASS_NAMES: dict[int, str] = {
    0: "Disturbed Galaxies",
    1: "Merging Galaxies",
    2: "Round Smooth Galaxies",
    3: "In-between Round Smooth Galaxies",
    4: "Cigar Shaped Smooth Galaxies",
    5: "Barred Spiral Galaxies",
    6: "Unbarred Tight Spiral Galaxies",
    7: "Unbarred Loose Spiral Galaxies",
    8: "Edge-on Galaxies without Bulge",
    9: "Edge-on Galaxies with Bulge",
}

PREPROCESSING_SPEC = (
    "rgb_uint8_0_255 -> tf.image.resize(bilinear) -> "
    "tf.keras.applications.mobilenet_v2.preprocess_input(-1_to_1)"
)


@dataclass(frozen=True)
class HeadHyperparams:
    hidden_units: int
    dropout: float
    learning_rate: float
    l2_reg: float
    use_class_weight: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def label(self) -> str:
        cw = "cw" if self.use_class_weight else "nocw"
        return (
            f"h{self.hidden_units}_do{self.dropout}_lr{self.learning_rate}_"
            f"l2{self.l2_reg}_{cw}"
        )


# Small, explicit hyperparameter search grid for CPU-friendly nested CV
FULL_SEARCH_GRID: list[HeadHyperparams] = [
    HeadHyperparams(hidden_units=0, dropout=0.2, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=False),
    HeadHyperparams(hidden_units=0, dropout=0.2, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=True),
    HeadHyperparams(hidden_units=128, dropout=0.3, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=False),
    HeadHyperparams(hidden_units=128, dropout=0.3, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=True),
    HeadHyperparams(hidden_units=128, dropout=0.4, learning_rate=5e-4, l2_reg=1e-3, use_class_weight=True),
]

SMOKE_SEARCH_GRID: list[HeadHyperparams] = [
    HeadHyperparams(hidden_units=0, dropout=0.2, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=False),
    HeadHyperparams(hidden_units=64, dropout=0.2, learning_rate=1e-3, l2_reg=1e-4, use_class_weight=True),
]


def default_repo_paths(project_root: Path) -> dict[str, Path]:
    return {
        "h5_path": project_root / "Galaxy10_DECals.h5",
        "dataset_json": project_root / "Kishore" / "results" / "final" / "dataset.json",
        "splits_json": project_root / "Kishore" / "results" / "final" / "splits.json",
        "assignments_csv": project_root / "Kishore" / "results" / "final" / "split_assignments.csv",
        "sagar_root": project_root / "Sagar",
    }