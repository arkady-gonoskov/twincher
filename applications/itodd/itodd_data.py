"""Location of the ITODD data of BOP.

The scripts read the archives of ITODD from the BOP benchmark (base archive, models,
validation and test images), unpacked into one directory with the subdirectories
itodd_base/, itodd_models/, itodd_val/ and itodd_test_all/ (README Section 8). Its path is
taken from the environment variable ITODD_DIR; without it, the directory ITODD next to
this file is used. In the configurations of LidarFit (lidarfit_*.json), paths that start
with {ITODD}/ refer to this directory.
"""

import os
from pathlib import Path

ROOT = Path(
    os.environ.get("ITODD_DIR", Path(__file__).resolve().parent / "ITODD")
).expanduser()
MODELS = ROOT / "itodd_models/models"
MODELS_EVAL = ROOT / "itodd_models/models_eval"
VAL = ROOT / "itodd_val/val/000001"
TEST = ROOT / "itodd_test_all/test/000001"


def model(obj):
    """The CAD model of object obj (PLY)."""
    return MODELS / f"obj_{obj:06d}.ply"


def resolve(base, path):
    """A path of a configuration: {ITODD}/... in the ITODD directory, others relative to
    the directory base."""
    path = str(path)
    if path.startswith("{ITODD}/"):
        return ROOT / path[len("{ITODD}/") :]
    return Path(base) / path
