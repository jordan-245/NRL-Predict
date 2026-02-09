"""
Model registry for saving, loading, and versioning trained models.

All models are serialised to ``outputs/models/<name>/v<version>/`` with
accompanying JSON metadata (training date, feature set, hyperparameters,
backtest metrics).

Supports two serialisation backends:

* **joblib** -- for scikit-learn models and pipelines.
* **torch.save** -- for PyTorch ``nn.Module`` models and ``NeuralTrainer``
  wrappers.

Typical usage
-------------
>>> from modelling.model_registry import save_model, load_model, list_models
>>> save_model(my_xgb, name="xgboost_v2_form", version="1",
...     metadata={"feature_set": "v2", "log_loss": 0.581})
>>> model = load_model("xgboost_v2_form")          # loads latest version
>>> model = load_model("xgboost_v2_form", version="1")
>>> for info in list_models():
...     print(info["name"], info["version"], info["metadata"])
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import OUTPUTS_DIR

logger = logging.getLogger(__name__)

_MODELS_DIR: Path = OUTPUTS_DIR / "models"
_MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Filenames used inside each version directory.
_SKLEARN_FILE = "model.joblib"
_TORCH_MODEL_FILE = "model.pt"
_TORCH_TRAINER_FILE = "trainer.pt"
_METADATA_FILE = "metadata.json"


# ============================================================================
# Internal helpers
# ============================================================================

def _version_dir(name: str, version: str) -> Path:
    """Return the directory for a specific model version."""
    return _MODELS_DIR / name / f"v{version}"


def _is_pytorch_model(model: Any) -> bool:
    """Check whether *model* is a PyTorch module or NeuralTrainer."""
    try:
        import torch.nn as nn
        from modelling.neural_models import NeuralTrainer

        if isinstance(model, nn.Module):
            return True
        if isinstance(model, NeuralTrainer):
            return True
    except ImportError:
        pass
    return False


def _latest_version(name: str) -> str | None:
    """Return the highest numeric version string for *name*, or None."""
    model_dir = _MODELS_DIR / name
    if not model_dir.is_dir():
        return None

    versions: list[int] = []
    for child in model_dir.iterdir():
        if child.is_dir() and child.name.startswith("v"):
            try:
                versions.append(int(child.name[1:]))
            except ValueError:
                continue

    if not versions:
        return None
    return str(max(versions))


def _next_version(name: str) -> str:
    """Return the next version number as a string."""
    latest = _latest_version(name)
    if latest is None:
        return "1"
    return str(int(latest) + 1)


# ============================================================================
# Public API
# ============================================================================

def save_model(
    model: Any,
    name: str,
    version: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Serialise a trained model to the registry.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn estimator, Pipeline, PyTorch ``nn.Module``,
        or ``NeuralTrainer`` instance.
    name : str
        Logical model name (e.g. ``"xgboost_v2_form"``).  Used as the
        directory name under ``outputs/models/``.
    version : str, optional
        Version label.  If ``None``, the next sequential integer is used.
    metadata : dict, optional
        Arbitrary metadata to persist alongside the model.  Typical keys:
        ``feature_set``, ``hyperparameters``, ``log_loss``, ``accuracy``,
        ``brier_score``, ``notes``.

    Returns
    -------
    Path
        The directory where the model was saved.

    Examples
    --------
    >>> path = save_model(my_model, "xgboost_v2_form", metadata={
    ...     "feature_set": "v2",
    ...     "log_loss": 0.581,
    ... })
    """
    if version is None:
        version = _next_version(name)

    vdir = _version_dir(name, version)
    vdir.mkdir(parents=True, exist_ok=True)

    # ---- Serialise the model ----
    if _is_pytorch_model(model):
        import torch
        from modelling.neural_models import NeuralTrainer

        if isinstance(model, NeuralTrainer):
            # Save the underlying nn.Module state dict and trainer config.
            torch.save(
                model.model.state_dict(),
                vdir / _TORCH_MODEL_FILE,
            )
            trainer_state = {
                "model_class": type(model.model).__name__,
                "model_config": _extract_nn_config(model.model),
                "trainer_params": model.get_params(),
                "scaler": model.scaler_,
                "is_lstm": model.is_lstm_,
                "feature_cols": model.feature_cols_,
            }
            import joblib
            joblib.dump(trainer_state, vdir / _TORCH_TRAINER_FILE)
        else:
            torch.save(model.state_dict(), vdir / _TORCH_MODEL_FILE)
    else:
        import joblib
        joblib.dump(model, vdir / _SKLEARN_FILE)

    # ---- Persist metadata ----
    meta = {
        "name": name,
        "version": version,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "model_type": type(model).__name__,
        "is_pytorch": _is_pytorch_model(model),
    }
    if metadata:
        meta["user_metadata"] = metadata

    with open(vdir / _METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    logger.info("Model '%s' v%s saved to %s", name, version, vdir)
    return vdir


def load_model(
    name: str,
    version: str | None = None,
) -> Any:
    """Load a model from the registry.

    Parameters
    ----------
    name : str
        Logical model name.
    version : str, optional
        Version to load.  If ``None``, the latest (highest-numbered)
        version is loaded.

    Returns
    -------
    estimator
        The deserialised model.

    Raises
    ------
    FileNotFoundError
        If the model or version does not exist.
    """
    if version is None:
        version = _latest_version(name)
        if version is None:
            raise FileNotFoundError(
                f"No saved versions found for model '{name}' "
                f"in {_MODELS_DIR / name}."
            )

    vdir = _version_dir(name, version)
    if not vdir.is_dir():
        raise FileNotFoundError(
            f"Version directory does not exist: {vdir}"
        )

    # ---- Read metadata to determine model type ----
    meta_path = vdir / _METADATA_FILE
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

    is_pytorch = meta.get("is_pytorch", False)

    if is_pytorch:
        return _load_pytorch_model(vdir, meta)

    # sklearn / joblib
    sklearn_path = vdir / _SKLEARN_FILE
    if not sklearn_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {sklearn_path}"
        )

    import joblib
    model = joblib.load(sklearn_path)
    logger.info("Loaded model '%s' v%s from %s", name, version, vdir)
    return model


def _load_pytorch_model(vdir: Path, meta: dict[str, Any]) -> Any:
    """Load a PyTorch model (nn.Module or NeuralTrainer)."""
    import torch

    trainer_path = vdir / _TORCH_TRAINER_FILE
    model_path = vdir / _TORCH_MODEL_FILE

    if trainer_path.is_file():
        # Full NeuralTrainer reconstruction.
        import joblib
        from modelling.neural_models import (
            MLPClassifier,
            LSTMModel,
            NeuralTrainer,
        )

        trainer_state = joblib.load(trainer_path)
        model_class_name = trainer_state.get("model_class", "MLPClassifier")
        model_config = trainer_state.get("model_config", {})

        if model_class_name == "LSTMModel":
            net = LSTMModel(**model_config)
        else:
            net = MLPClassifier(**model_config)

        state_dict = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        net.load_state_dict(state_dict)

        trainer_params = trainer_state.get("trainer_params", {})
        trainer = NeuralTrainer(model=net, **trainer_params)
        trainer.scaler_ = trainer_state.get("scaler")
        trainer.is_lstm_ = trainer_state.get("is_lstm", False)
        trainer.feature_cols_ = trainer_state.get("feature_cols")
        trainer.is_fitted_ = True
        trainer.model.eval()

        logger.info("Loaded NeuralTrainer from %s", vdir)
        return trainer

    if model_path.is_file():
        # Raw state dict -- caller must reconstruct the architecture.
        state_dict = torch.load(
            model_path, map_location="cpu", weights_only=True
        )
        logger.info(
            "Loaded raw PyTorch state dict from %s. "
            "Caller must instantiate the architecture and call load_state_dict.",
            vdir,
        )
        return state_dict

    raise FileNotFoundError(
        f"No PyTorch model file found in {vdir}"
    )


def _extract_nn_config(module: Any) -> dict[str, Any]:
    """Extract constructor kwargs from a known nn.Module subclass.

    Falls back to an empty dict for unknown architectures.
    """
    from modelling.neural_models import MLPClassifier, LSTMModel

    if isinstance(module, MLPClassifier):
        return {
            "input_dim": module.input_dim,
            "hidden_dims": module.hidden_dims,
            "dropout": module.dropout,
            "use_batch_norm": module.use_batch_norm,
        }
    if isinstance(module, LSTMModel):
        return {
            "input_dim": module.input_dim,
            "hidden_dim": module.hidden_dim,
            "num_layers": module.num_layers,
            "dropout": module.lstm.dropout,
            "bidirectional": module.bidirectional,
        }
    return {}


def list_models() -> list[dict[str, Any]]:
    """List all saved models with their metadata.

    Returns
    -------
    list[dict]
        Each dict contains keys ``name``, ``version``, ``saved_at``,
        ``model_type``, and optionally ``user_metadata``.

    Examples
    --------
    >>> for info in list_models():
    ...     print(info["name"], f"v{info['version']}", info.get("saved_at"))
    """
    results: list[dict[str, Any]] = []

    if not _MODELS_DIR.is_dir():
        return results

    for model_dir in sorted(_MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        name = model_dir.name

        for version_dir in sorted(model_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.startswith("v"):
                continue

            meta_path = version_dir / _METADATA_FILE
            if meta_path.is_file():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            else:
                meta = {
                    "name": name,
                    "version": version_dir.name[1:],
                    "saved_at": None,
                    "model_type": "unknown",
                }

            results.append(meta)

    return results


def delete_model(name: str, version: str | None = None) -> None:
    """Delete a model (specific version, or all versions if not specified).

    Parameters
    ----------
    name : str
        Logical model name.
    version : str, optional
        Specific version to delete.  If ``None``, all versions are removed.
    """
    import shutil

    if version is not None:
        vdir = _version_dir(name, version)
        if vdir.is_dir():
            shutil.rmtree(vdir)
            logger.info("Deleted model '%s' v%s.", name, version)
        else:
            logger.warning("Version directory not found: %s", vdir)
        return

    model_dir = _MODELS_DIR / name
    if model_dir.is_dir():
        shutil.rmtree(model_dir)
        logger.info("Deleted all versions of model '%s'.", name)
    else:
        logger.warning("Model directory not found: %s", model_dir)


def get_model_metadata(
    name: str,
    version: str | None = None,
) -> dict[str, Any]:
    """Retrieve metadata for a specific model version.

    Parameters
    ----------
    name : str
        Logical model name.
    version : str, optional
        Version.  If ``None``, returns metadata for the latest version.

    Returns
    -------
    dict
        The metadata dictionary.

    Raises
    ------
    FileNotFoundError
        If the model or metadata file does not exist.
    """
    if version is None:
        version = _latest_version(name)
        if version is None:
            raise FileNotFoundError(
                f"No saved versions found for model '{name}'."
            )

    vdir = _version_dir(name, version)
    meta_path = vdir / _METADATA_FILE
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}"
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
