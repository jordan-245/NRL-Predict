"""
Neural-network models for NRL match winner prediction.

Provides two PyTorch ``nn.Module`` architectures:

* **MLPClassifier** -- feedforward network with configurable hidden layers,
  dropout, and batch normalisation.
* **LSTMModel** -- LSTM for sequential match data (last *N* matches per
  team) followed by a classification head.

The **NeuralTrainer** class wraps either architecture with a scikit-learn-
compatible training loop (``fit``, ``predict``, ``predict_proba``), including
early stopping, learning-rate scheduling, and automatic data preparation
(scaling, sequence creation for LSTM).

Typical usage
-------------
>>> from modelling.neural_models import MLPClassifier, NeuralTrainer
>>> net = MLPClassifier(input_dim=40, hidden_dims=[128, 64])
>>> trainer = NeuralTrainer(net, lr=1e-3, epochs=100)
>>> trainer.fit(X_train, y_train, X_val=X_val, y_val=y_val)
>>> proba = trainer.predict_proba(X_test)
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from config.settings import RANDOM_SEED

logger = logging.getLogger(__name__)

# Ensure reproducibility.
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================================
# Utility helpers
# ============================================================================

def _get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _to_numpy(X: pd.DataFrame | np.ndarray | ArrayLike) -> np.ndarray:
    """Convert input to a 2-D float32 numpy array."""
    if isinstance(X, pd.DataFrame):
        return X.values.astype(np.float32)
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


# ============================================================================
# 1. MLP Classifier
# ============================================================================

class MLPClassifier(nn.Module):
    """Feedforward neural network with configurable architecture.

    Parameters
    ----------
    input_dim : int
        Number of input features.
    hidden_dims : list[int]
        Sizes of each hidden layer (e.g. ``[128, 64, 32]``).
    dropout : float
        Dropout probability applied after each hidden layer.
    use_batch_norm : bool
        Whether to include batch normalisation before activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hdim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hdim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, 2)  # binary: 2 logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits of shape ``(batch, 2)``."""
        h = self.feature_extractor(x)
        return self.classifier(h)


# ============================================================================
# 2. LSTM Model
# ============================================================================

class LSTMModel(nn.Module):
    """LSTM for sequential match data (last *N* matches per team).

    The model expects input of shape ``(batch, seq_len, input_dim)`` where
    each time step represents the feature vector for a historical match.
    The final hidden state is passed through a classification head.

    Parameters
    ----------
    input_dim : int
        Feature dimension per time step.
    hidden_dim : int
        LSTM hidden-state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    dropout : float
        Dropout between LSTM layers (applied only if ``num_layers > 1``).
    bidirectional : bool
        Use a bidirectional LSTM.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, seq_len, input_dim)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(batch, 2)``.
        """
        # lstm_out: (batch, seq_len, hidden_dim * num_directions)
        # We take the output at the last time step.
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.classifier(last_hidden)


# ============================================================================
# 3. Sequence Builder (for LSTM)
# ============================================================================

class SequenceBuilder:
    """Create fixed-length match sequences for each team.

    Given a chronological DataFrame with a ``team`` identifier column, this
    helper produces ``(X_seq, y)`` tensors where ``X_seq`` has shape
    ``(n_samples, seq_len, n_features)`` by looking back at each team's
    last ``seq_len`` matches.

    Parameters
    ----------
    seq_len : int
        Number of historical matches per sequence.
    team_col : str
        Column name identifying the team in each row.
    """

    def __init__(
        self,
        seq_len: int = 5,
        team_col: str = "home_team",
    ) -> None:
        self.seq_len = seq_len
        self.team_col = team_col

    def build_sequences(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None = None,
        feature_cols: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build sequential input arrays.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``team_col`` and the feature columns.  Rows should
            be in chronological order.
        y : np.ndarray, optional
            Target values aligned with *X*.  Rows without enough history
            are dropped from both *X_seq* and *y_seq*.
        feature_cols : list[str], optional
            Feature columns to extract.  If ``None``, all numeric columns
            except ``team_col`` are used.

        Returns
        -------
        X_seq : np.ndarray of shape (n_valid, seq_len, n_features)
        y_seq : np.ndarray of shape (n_valid,) or None
        """
        if feature_cols is None:
            feature_cols = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c != self.team_col
            ]

        # Group history by team.
        team_histories: dict[str, list[np.ndarray]] = {}
        sequences: list[np.ndarray] = []
        targets: list[int] = []

        for idx in range(len(X)):
            team = str(X.iloc[idx][self.team_col])
            feats = X.iloc[idx][feature_cols].values.astype(np.float32)

            history = team_histories.setdefault(team, [])

            if len(history) >= self.seq_len:
                seq = np.stack(history[-self.seq_len :])
                sequences.append(seq)
                if y is not None:
                    targets.append(int(y[idx]))

            history.append(feats)

        X_seq = np.stack(sequences) if sequences else np.empty(
            (0, self.seq_len, len(feature_cols)), dtype=np.float32
        )
        y_seq = np.array(targets, dtype=np.int64) if y is not None else None

        logger.debug(
            "SequenceBuilder: %d sequences of length %d from %d rows.",
            len(sequences),
            self.seq_len,
            len(X),
        )
        return X_seq, y_seq


# ============================================================================
# 4. Neural Trainer (sklearn-compatible wrapper)
# ============================================================================

class NeuralTrainer:
    """PyTorch training wrapper with a scikit-learn-compatible interface.

    Supports both ``MLPClassifier`` (2-D input) and ``LSTMModel`` (3-D
    sequential input).  Provides ``fit``, ``predict``, and
    ``predict_proba``.

    Parameters
    ----------
    model : nn.Module
        The PyTorch module to train (``MLPClassifier`` or ``LSTMModel``).
    lr : float
        Initial learning rate for the Adam optimiser.
    epochs : int
        Maximum number of training epochs.
    batch_size : int
        Mini-batch size.
    patience : int
        Number of epochs with no validation improvement before early
        stopping.  Set to ``0`` to disable early stopping.
    weight_decay : float
        L2 regularisation coefficient.
    seq_len : int
        Sequence length for LSTM models.  Ignored for MLP.
    seq_team_col : str
        Column name used by ``SequenceBuilder`` when preparing LSTM data.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 10,
        weight_decay: float = 1e-4,
        seq_len: int = 5,
        seq_team_col: str = "home_team",
    ) -> None:
        self.model = model
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.weight_decay = weight_decay
        self.seq_len = seq_len
        self.seq_team_col = seq_team_col

        self.device = _get_device()
        self.scaler_: StandardScaler | None = None
        self.is_lstm_: bool = isinstance(model, LSTMModel)
        self.seq_builder_: SequenceBuilder | None = None
        self.feature_cols_: list[str] | None = None
        self.best_state_dict_: dict | None = None
        self.train_losses_: list[float] = []
        self.val_losses_: list[float] = []
        self.is_fitted_: bool = False

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_data_mlp(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        fit_scaler: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Scale features and convert to tensors for MLP."""
        X_np = _to_numpy(X)

        if fit_scaler:
            self.scaler_ = StandardScaler()
            X_np = self.scaler_.fit_transform(X_np)
        elif self.scaler_ is not None:
            X_np = self.scaler_.transform(X_np)

        X_t = torch.from_numpy(X_np).float()
        y_t = torch.from_numpy(np.asarray(y, dtype=np.int64)) if y is not None else None
        return X_t, y_t

    def _prepare_data_lstm(
        self,
        X: pd.DataFrame,
        y: np.ndarray | None = None,
        fit_scaler: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Build sequences, scale, and convert to tensors for LSTM."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "LSTM models require a pandas DataFrame with a team column "
                f"('{self.seq_team_col}') for sequence construction."
            )

        if fit_scaler:
            self.seq_builder_ = SequenceBuilder(
                seq_len=self.seq_len,
                team_col=self.seq_team_col,
            )
            # Identify numeric feature columns.
            self.feature_cols_ = [
                c for c in X.select_dtypes(include=[np.number]).columns
                if c != self.seq_team_col
            ]

        X_seq, y_seq = self.seq_builder_.build_sequences(
            X, y, self.feature_cols_
        )

        # Scale across the feature dimension.
        n, s, f = X_seq.shape
        X_flat = X_seq.reshape(-1, f)

        if fit_scaler:
            self.scaler_ = StandardScaler()
            X_flat = self.scaler_.fit_transform(X_flat)
        elif self.scaler_ is not None:
            X_flat = self.scaler_.transform(X_flat)

        X_seq = X_flat.reshape(n, s, f).astype(np.float32)

        X_t = torch.from_numpy(X_seq).float()
        y_t = torch.from_numpy(y_seq) if y_seq is not None else None
        return X_t, y_t

    def _prepare(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        fit_scaler: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Dispatch to MLP or LSTM data preparation."""
        if self.is_lstm_:
            return self._prepare_data_lstm(X, y, fit_scaler=fit_scaler)
        return self._prepare_data_mlp(X, y, fit_scaler=fit_scaler)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: ArrayLike,
        X_val: pd.DataFrame | np.ndarray | None = None,
        y_val: ArrayLike | None = None,
    ) -> "NeuralTrainer":
        """Train the neural network.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Training features.
        y : array-like of shape (n_samples,)
            Binary target (1 = home win).
        X_val, y_val : optional
            Validation set for early stopping.  If not provided, 15 % of the
            training data is held out automatically.

        Returns
        -------
        self
        """
        y_np = np.asarray(y, dtype=np.int64)

        # If no explicit validation set, split off the last 15 %.
        if X_val is None or y_val is None:
            n = len(y_np)
            split = int(n * 0.85)
            if isinstance(X, pd.DataFrame):
                X_train_raw, X_val_raw = X.iloc[:split], X.iloc[split:]
            else:
                X_train_raw, X_val_raw = X[:split], X[split:]
            y_train_raw, y_val_raw = y_np[:split], y_np[split:]
        else:
            X_train_raw, X_val_raw = X, X_val
            y_train_raw = y_np
            y_val_raw = np.asarray(y_val, dtype=np.int64)

        # Prepare tensors.
        X_train_t, y_train_t = self._prepare(
            X_train_raw, y_train_raw, fit_scaler=True
        )
        X_val_t, y_val_t = self._prepare(X_val_raw, y_val_raw)

        # Handle edge case: sequence builder may yield zero valid samples.
        if X_train_t.shape[0] == 0:
            logger.warning(
                "No valid training samples after sequence construction. "
                "Consider reducing seq_len or providing more data."
            )
            self.is_fitted_ = True
            return self

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True
        )

        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, verbose=False
        )

        best_val_loss = float("inf")
        epochs_no_improve = 0
        self.train_losses_ = []
        self.val_losses_ = []

        for epoch in range(1, self.epochs + 1):
            # --- Training ---
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimiser.zero_grad()
                logits = self.model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            self.train_losses_.append(avg_train_loss)

            # --- Validation ---
            val_loss = self._evaluate_loss(X_val_t, y_val_t, criterion)
            self.val_losses_.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.best_state_dict_ = copy.deepcopy(
                    self.model.state_dict()
                )
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
                    "best_val=%.4f  lr=%.2e",
                    epoch,
                    self.epochs,
                    avg_train_loss,
                    val_loss,
                    best_val_loss,
                    optimiser.param_groups[0]["lr"],
                )

            if self.patience > 0 and epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (no improvement for %d epochs).",
                    epoch,
                    self.patience,
                )
                break

        # Restore best model weights.
        if self.best_state_dict_ is not None:
            self.model.load_state_dict(self.best_state_dict_)

        self.model.eval()
        self.is_fitted_ = True
        logger.info(
            "Training complete. Best validation loss: %.4f", best_val_loss
        )
        return self

    # ------------------------------------------------------------------
    def _evaluate_loss(
        self,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        """Compute loss on a dataset without gradient tracking."""
        self.model.eval()
        with torch.no_grad():
            X_t = X_t.to(self.device)
            y_t = y_t.to(self.device)
            logits = self.model(X_t)
            loss = criterion(logits, y_t)
        return loss.item()

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return class probabilities ``[P(away_win), P(home_win)]``.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features (same format as training data).

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
        """
        if not self.is_fitted_:
            raise RuntimeError("NeuralTrainer has not been fitted yet.")

        X_t, _ = self._prepare(X)

        if X_t.shape[0] == 0:
            return np.empty((0, 2), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            X_t = X_t.to(self.device)
            logits = self.model(X_t)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Return predicted class labels (0 = away win, 1 = home win).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        if proba.shape[0] == 0:
            return np.empty(0, dtype=int)
        return np.argmax(proba, axis=1)

    # ------------------------------------------------------------------
    # sklearn compatibility helpers
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "seq_len": self.seq_len,
            "seq_team_col": self.seq_team_col,
        }

    def set_params(self, **params: Any) -> "NeuralTrainer":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter '{key}' for NeuralTrainer."
                )
        return self

    def __repr__(self) -> str:
        model_name = type(self.model).__name__
        return (
            f"NeuralTrainer(model={model_name}, lr={self.lr}, "
            f"epochs={self.epochs}, batch_size={self.batch_size}, "
            f"patience={self.patience})"
        )
