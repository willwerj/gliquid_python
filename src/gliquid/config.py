import os
from pathlib import Path
from typing import Optional

_DIR_STRUCT_OPTS = ["flat", "nested"]
_DATA_DIR_ENV_VAR = "GLIQUID_DATA_DIR"
_MODEL_BUNDLE_REQUIRED_FILES = [
    "prediction_dataset_symmetric.xlsx",
    "prediction_dataset_antisymmetric.xlsx",
    "model/L0_a_model.joblib",
    "model/L0_b_model.joblib",
    "model/L1_a_model.joblib",
    "model/feature_names_symm.joblib",
    "model/feature_names_anti.joblib",
]

project_root = None
data_dir = None
dir_structure = None
fusion_enthalpies_file = None
fusion_temps_file = None
vaporization_temps_file = None


def set_project_root(path: Path | str):
    global project_root
    project_root = Path(path).expanduser().resolve()


def set_data_dir(path: Path | str):
    global data_dir
    global fusion_enthalpies_file
    global fusion_temps_file
    global vaporization_temps_file

    data_dir = Path(path).expanduser().resolve()
    fusion_enthalpies_file = Path(data_dir / "fusion_enthalpies.json")
    fusion_temps_file = Path(data_dir / "fusion_temperatures.json")
    vaporization_temps_file = Path(data_dir / "vaporization_temperatures.json")


def set_dir_structure(structure: str):
    global dir_structure
    if structure not in _DIR_STRUCT_OPTS:
        raise ValueError(f"dir_structure must be one of {_DIR_STRUCT_OPTS}")
    dir_structure = structure


def find_project_root(dirname: str = "gliquid_python") -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if candidate.name == dirname:
            return candidate
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "gliquid").exists():
            return candidate
    return current


def find_data_dir(project: Optional[Path] = None) -> Path:
    env_path = os.getenv(_DATA_DIR_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    project = project or find_project_root()
    candidate = project / "data"
    if candidate.exists():
        return candidate.resolve()
    return candidate.resolve()


def _is_model_bundle_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return all((path / rel).exists() for rel in _MODEL_BUNDLE_REQUIRED_FILES)


def find_model_bundle_dir(
    base_dir: Path | str | None = None,
    bundle_name: Optional[str] = None,
) -> Path:
    base = Path(base_dir if base_dir is not None else data_dir).expanduser().resolve()
    if bundle_name:
        selected = base / bundle_name
        if _is_model_bundle_dir(selected):
            return selected
        raise FileNotFoundError(
            f"Bundle '{bundle_name}' does not contain required model artifacts in '{base}'."
        )

    bundles = [p for p in base.iterdir() if _is_model_bundle_dir(p)]
    if not bundles:
        raise FileNotFoundError(f"No model bundle directories found in '{base}'.")
    bundles.sort(key=lambda p: (p.name, p.stat().st_mtime), reverse=True)
    return bundles[0]


set_project_root(find_project_root())
set_data_dir(find_data_dir(project_root))
set_dir_structure(_DIR_STRUCT_OPTS[0])
