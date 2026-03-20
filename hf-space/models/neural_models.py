#!/usr/bin/env python3
"""
NBA Quant AI — Neural Network Models (2025-2026 SOTA)
======================================================
Real, production-grade neural architectures for NBA game prediction.

Models implemented:
  1. LSTMSequenceModel      — Bidirectional LSTM over last N games
  2. TransformerAttentionModel — Self-attention over game history
  3. TabNetModel             — Attention-based tabular learning (Arik & Pfister 2021)
  4. FTTransformerModel      — Feature Tokenizer + Transformer (Gorishniy et al. 2021)
  5. DeepEnsemble            — N independent nets, averaged predictions
  6. ConformalPredictionWrapper — Calibrated prediction intervals (any base model)
  7. AutoGluonEnsemble       — Auto-stacking over hundreds of configs

All models:
  - Handle NaN gracefully (median imputation)
  - Work with 6000+ features
  - Use early stopping
  - CPU-only PyTorch (no CUDA needed)
  - Fit in 16 GB RAM (HF Spaces free tier)

THIS RUNS ON HF SPACES ONLY — NOT ON VM.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pickle
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Lazy imports — heavy libraries loaded only when a model is instantiated
# ---------------------------------------------------------------------------

def _import_torch():
    """Import torch lazily to avoid startup cost."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, optim, DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Base class — common interface for all models
# ---------------------------------------------------------------------------

class BaseNBAModel(ABC):
    """Abstract base for all NBA prediction models."""

    def __init__(self, **params):
        self.params = params
        self._scaler: Optional[StandardScaler] = None
        self._feature_medians: Optional[np.ndarray] = None
        self._is_fitted = False

    # --- public interface ---------------------------------------------------

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseNBAModel":
        """Train the model. Returns self."""
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(home_win) for each row — shape (n,)."""
        ...

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameter dict (JSON-serialisable)."""
        return {k: v for k, v in self.params.items() if _is_jsonable(v)}

    def save(self, path: Union[str, Path]) -> None:
        """Persist to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseNBAModel":
        """Load from disk."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj

    # --- NaN handling & scaling --------------------------------------------

    def _impute(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Replace NaN/Inf with column medians. If *fit*, compute medians first."""
        X = np.array(X, dtype=np.float32)
        X = np.where(np.isfinite(X), X, np.nan)
        if fit:
            self._feature_medians = np.nanmedian(X, axis=0)
            self._feature_medians = np.where(
                np.isfinite(self._feature_medians), self._feature_medians, 0.0
            )
        medians = self._feature_medians if self._feature_medians is not None else np.zeros(X.shape[1])
        inds = np.where(np.isnan(X))
        X[inds] = np.take(medians, inds[1])
        return X

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Standard-scale features."""
        if fit:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X).astype(np.float32)
        if self._scaler is not None:
            return self._scaler.transform(X).astype(np.float32)
        return X.astype(np.float32)

    def _prepare(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Impute + scale."""
        X = self._impute(X, fit=fit)
        X = self._scale(X, fit=fit)
        return X

    def _auto_val_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        val_frac: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """If no validation set provided, carve one from the tail (time-ordered)."""
        if X_val is not None and y_val is not None:
            return X, y, X_val, y_val
        split = int(len(X) * (1 - val_frac))
        return X[:split], y[:split], X[split:], y[split:]


# ===========================================================================
# 1. LSTM Game Sequence Model
# ===========================================================================

class LSTMSequenceModel(BaseNBAModel):
    """
    Bidirectional LSTM over the last *seq_len* games of features per team.

    Input shape: (batch, seq_len, n_features)
    Architecture: BiLSTM(128) -> BiLSTM(64) -> Dense(32) -> Sigmoid

    For flat input (n_samples, n_features), the model internally reshapes
    using a sliding window of *seq_len* rows, treating consecutive games as
    the sequence dimension.  For true per-team sequences, pass 3-D arrays
    directly.
    """

    def __init__(
        self,
        seq_len: int = 10,
        hidden1: int = 128,
        hidden2: int = 64,
        dense_dim: int = 32,
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        epochs: int = 120,
        patience: int = 15,
        **kw,
    ):
        super().__init__(
            seq_len=seq_len, hidden1=hidden1, hidden2=hidden2,
            dense_dim=dense_dim, dropout=dropout, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size,
            epochs=epochs, patience=patience, **kw,
        )
        self.seq_len = seq_len
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.dense_dim = dense_dim
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self._net = None

    # --- PyTorch module (defined inside method to keep torch lazy) ----------

    @staticmethod
    def _build_net(n_features: int, cfg: dict):
        torch, nn, _, _, _ = _import_torch()

        class BiLSTMNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(
                    input_size=n_features,
                    hidden_size=cfg["hidden1"],
                    batch_first=True,
                    bidirectional=True,
                    dropout=cfg["dropout"] if cfg["hidden2"] else 0,
                )
                self.lstm2 = nn.LSTM(
                    input_size=cfg["hidden1"] * 2,  # bidirectional doubles
                    hidden_size=cfg["hidden2"],
                    batch_first=True,
                    bidirectional=True,
                )
                self.dropout = nn.Dropout(cfg["dropout"])
                self.fc1 = nn.Linear(cfg["hidden2"] * 2, cfg["dense_dim"])
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(cfg["dense_dim"], 1)

            def forward(self, x):
                # x: (batch, seq_len, features)
                out, _ = self.lstm1(x)
                out = self.dropout(out)
                out, _ = self.lstm2(out)
                # Take last hidden state
                out = out[:, -1, :]
                out = self.dropout(out)
                out = self.relu(self.fc1(out))
                out = self.dropout(out)
                return torch.sigmoid(self.fc2(out)).squeeze(-1)

        return BiLSTMNet()

    # --- Sequence construction from flat arrays ----------------------------

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert flat (n_games, n_features) into (n_sequences, seq_len, n_features).
        Uses a sliding window — game i maps to window [i-seq_len+1 .. i].
        The first seq_len-1 games are dropped (not enough history).
        """
        if X.ndim == 3:
            return X, y  # already sequential
        seqs, labels = [], []
        for i in range(self.seq_len - 1, len(X)):
            seqs.append(X[i - self.seq_len + 1 : i + 1])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    # --- fit / predict -----------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LSTMSequenceModel":
        torch, nn, optim, DataLoader, TensorDataset = _import_torch()

        # Prepare
        X_train = self._prepare(X_train, fit=True)
        X_train, y_train, X_val, y_val = self._auto_val_split(X_train, y_train, X_val, y_val)
        if X_val is not None:
            X_val = self._prepare(X_val)

        # Build sequences
        X_tr_seq, y_tr_seq = self._make_sequences(X_train, y_train)
        X_va_seq, y_va_seq = self._make_sequences(X_val, y_val)

        n_features = X_tr_seq.shape[2]
        self._net = self._build_net(n_features, {
            "hidden1": self.hidden1, "hidden2": self.hidden2,
            "dense_dim": self.dense_dim, "dropout": self.dropout,
        })

        optimizer = optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.BCELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_X_t = torch.from_numpy(X_va_seq)
        val_y_t = torch.from_numpy(y_va_seq)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                preds = self._net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)

            # Validation
            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(val_X_t)
                val_loss = criterion(val_preds, val_y_t).item()
            self._net.train()

            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, _, _, _, _ = _import_torch()
        assert self._is_fitted, "Model not fitted yet"

        X = self._prepare(X)
        # If flat, create sequences with padding for early games
        if X.ndim == 2:
            seqs = []
            for i in range(len(X)):
                start = max(0, i - self.seq_len + 1)
                seq = X[start : i + 1]
                if len(seq) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(seq), X.shape[1]), dtype=np.float32)
                    seq = np.concatenate([pad, seq], axis=0)
                seqs.append(seq)
            X_seq = np.array(seqs, dtype=np.float32)
        else:
            X_seq = X.astype(np.float32)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_seq))
        return preds.numpy()


# ===========================================================================
# 2. Transformer Attention Model
# ===========================================================================

class TransformerAttentionModel(BaseNBAModel):
    """
    Self-attention over team performance history.

    Architecture:
        Linear projection -> Positional encoding ->
        TransformerEncoder (2 layers, 4 heads) ->
        Global average pool -> Dense -> Sigmoid

    For flat input the model treats each game as one token in a
    sequence of *seq_len* tokens (same sliding-window as LSTM model).
    """

    def __init__(
        self,
        seq_len: int = 10,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.2,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs: int = 120,
        patience: int = 15,
        **kw,
    ):
        super().__init__(
            seq_len=seq_len, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, dim_ff=dim_ff, dropout=dropout,
            lr=lr, weight_decay=weight_decay, batch_size=batch_size,
            epochs=epochs, patience=patience, **kw,
        )
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self._net = None

    @staticmethod
    def _build_net(n_features: int, cfg: dict):
        torch, nn, _, _, _ = _import_torch()

        class PositionalEncoding(nn.Module):
            """Sinusoidal positional encoding for game order."""
            def __init__(self, d_model: int, max_len: int = 200):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])  # handle odd d_model
                pe = pe.unsqueeze(0)  # (1, max_len, d_model)
                self.register_buffer("pe", pe)

            def forward(self, x):
                return x + self.pe[:, : x.size(1), :]

        class TransformerNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_proj = nn.Linear(n_features, cfg["d_model"])
                self.pos_enc = PositionalEncoding(cfg["d_model"], max_len=cfg["seq_len"] + 10)
                self.layer_norm_in = nn.LayerNorm(cfg["d_model"])
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cfg["d_model"],
                    nhead=cfg["n_heads"],
                    dim_feedforward=cfg["dim_ff"],
                    dropout=cfg["dropout"],
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=cfg["n_layers"]
                )
                self.dropout = nn.Dropout(cfg["dropout"])
                self.fc1 = nn.Linear(cfg["d_model"], cfg["d_model"] // 2)
                self.gelu = nn.GELU()
                self.fc2 = nn.Linear(cfg["d_model"] // 2, 1)

            def forward(self, x):
                # x: (batch, seq_len, n_features)
                x = self.input_proj(x)
                x = self.pos_enc(x)
                x = self.layer_norm_in(x)
                x = self.encoder(x)
                # Global average pooling across sequence dim
                x = x.mean(dim=1)
                x = self.dropout(x)
                x = self.gelu(self.fc1(x))
                x = self.dropout(x)
                return torch.sigmoid(self.fc2(x)).squeeze(-1)

        return TransformerNet()

    def _make_sequences(self, X: np.ndarray, y: np.ndarray):
        if X.ndim == 3:
            return X, y
        seqs, labels = [], []
        for i in range(self.seq_len - 1, len(X)):
            seqs.append(X[i - self.seq_len + 1 : i + 1])
            labels.append(y[i])
        return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TransformerAttentionModel":
        torch, nn, optim, DataLoader, TensorDataset = _import_torch()

        X_train = self._prepare(X_train, fit=True)
        X_train, y_train, X_val, y_val = self._auto_val_split(X_train, y_train, X_val, y_val)
        if X_val is not None:
            X_val = self._prepare(X_val)

        X_tr_seq, y_tr_seq = self._make_sequences(X_train, y_train)
        X_va_seq, y_va_seq = self._make_sequences(X_val, y_val)

        n_features = X_tr_seq.shape[2]
        self._net = self._build_net(n_features, {
            "d_model": self.d_model, "n_heads": self.n_heads,
            "n_layers": self.n_layers, "dim_ff": self.dim_ff,
            "dropout": self.dropout, "seq_len": self.seq_len,
        })

        optimizer = optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        criterion = nn.BCELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_X_t = torch.from_numpy(X_va_seq)
        val_y_t = torch.from_numpy(y_va_seq)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                preds = self._net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            scheduler.step(epoch + epoch_loss)  # warm restart input

            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(val_X_t)
                val_loss = criterion(val_preds, val_y_t).item()
            self._net.train()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, _, _, _, _ = _import_torch()
        assert self._is_fitted, "Model not fitted yet"

        X = self._prepare(X)
        if X.ndim == 2:
            seqs = []
            for i in range(len(X)):
                start = max(0, i - self.seq_len + 1)
                seq = X[start : i + 1]
                if len(seq) < self.seq_len:
                    pad = np.zeros((self.seq_len - len(seq), X.shape[1]), dtype=np.float32)
                    seq = np.concatenate([pad, seq], axis=0)
                seqs.append(seq)
            X_seq = np.array(seqs, dtype=np.float32)
        else:
            X_seq = X.astype(np.float32)

        self._net.eval()
        with torch.no_grad():
            preds = self._net(torch.from_numpy(X_seq))
        return preds.numpy()


# ===========================================================================
# 3. TabNet — Attention-based Tabular Model
# ===========================================================================

class TabNetModel(BaseNBAModel):
    """
    TabNet (Arik & Pfister 2021) — SOTA attention-based tabular learning.

    Uses sequential attention to select features at each decision step,
    providing built-in interpretability via attention masks.

    Wraps pytorch_tabnet.TabNetClassifier with NaN handling and
    early stopping.
    """

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 1e-4,
        n_independent: int = 2,
        n_shared: int = 2,
        lr: float = 2e-2,
        batch_size: int = 1024,
        virtual_batch_size: int = 256,
        epochs: int = 200,
        patience: int = 20,
        mask_type: str = "entmax",
        **kw,
    ):
        super().__init__(
            n_d=n_d, n_a=n_a, n_steps=n_steps, gamma=gamma,
            lambda_sparse=lambda_sparse, n_independent=n_independent,
            n_shared=n_shared, lr=lr, batch_size=batch_size,
            virtual_batch_size=virtual_batch_size, epochs=epochs,
            patience=patience, mask_type=mask_type, **kw,
        )
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.lr = lr
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.epochs = epochs
        self.patience = patience
        self.mask_type = mask_type
        self._clf = None
        self._feature_importances: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "TabNetModel":
        from pytorch_tabnet.tab_model import TabNetClassifier

        X_train = self._impute(X_train, fit=True)
        X_train, y_train, X_val, y_val = self._auto_val_split(X_train, y_train, X_val, y_val)
        if X_val is not None:
            X_val = self._impute(X_val)

        y_train = y_train.astype(np.int64)
        y_val = y_val.astype(np.int64)

        self._clf = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            n_independent=self.n_independent,
            n_shared=self.n_shared,
            optimizer_fn=None,  # default Adam
            optimizer_params={"lr": self.lr},
            mask_type=self.mask_type,
            scheduler_fn=None,
            scheduler_params=None,
            verbose=0,
            device_name="cpu",
        )

        self._clf.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["logloss"],
            max_epochs=self.epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(self.virtual_batch_size, self.batch_size),
            drop_last=False,
        )

        self._feature_importances = self._clf.feature_importances_
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self._is_fitted, "Model not fitted yet"
        X = self._impute(X)
        proba = self._clf.predict_proba(X)  # shape (n, 2)
        return proba[:, 1]

    def get_feature_importances(self) -> Optional[np.ndarray]:
        """Return TabNet attention-based feature importances."""
        return self._feature_importances

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Return per-sample feature attention masks."""
        assert self._is_fitted, "Model not fitted yet"
        X = self._impute(X)
        masks, _ = self._clf.explain(X)
        return masks


# ===========================================================================
# 4. FT-Transformer (Feature Tokenizer + Transformer)
# ===========================================================================

class FTTransformerModel(BaseNBAModel):
    """
    FT-Transformer (Gorishniy et al. 2021) — confirmed SOTA for tabular
    data in 2025-2026 benchmarks.

    Each numerical feature is projected into a *d_token*-dimensional embedding.
    A [CLS] token is prepended. Self-attention across all feature tokens
    captures cross-feature interactions.  The [CLS] representation feeds a
    classification head.

    Because the full 6000+ features would create 6000+ tokens (too large for
    self-attention on CPU), we first apply a learned linear bottleneck to
    reduce to *n_tokens* feature groups.
    """

    def __init__(
        self,
        n_tokens: int = 128,
        d_token: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_ff: int = 256,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        batch_size: int = 512,
        epochs: int = 120,
        patience: int = 15,
        **kw,
    ):
        super().__init__(
            n_tokens=n_tokens, d_token=d_token, n_heads=n_heads,
            n_layers=n_layers, dim_ff=dim_ff, dropout=dropout,
            attention_dropout=attention_dropout, lr=lr,
            weight_decay=weight_decay, batch_size=batch_size,
            epochs=epochs, patience=patience, **kw,
        )
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self._net = None

    @staticmethod
    def _build_net(n_features: int, cfg: dict):
        torch, nn, _, _, _ = _import_torch()

        class FTTransformerNet(nn.Module):
            """
            Feature Tokenizer + Transformer.

            1) Bottleneck: Linear(n_features -> n_tokens)  — group features
            2) Token embed: each of *n_tokens* scalars -> d_token vector
            3) Prepend [CLS] token
            4) TransformerEncoder
            5) [CLS] output -> classification head
            """

            def __init__(self):
                super().__init__()
                n_tok = cfg["n_tokens"]
                d_tok = cfg["d_token"]

                # Bottleneck projection: reduce 6000 features to n_tokens groups
                self.bottleneck = nn.Linear(n_features, n_tok)
                self.bn_norm = nn.LayerNorm(n_tok)

                # Per-token embedding: each scalar -> d_token vector
                # Implemented as a shared Linear(1 -> d_token) + per-token bias
                self.token_weight = nn.Parameter(torch.randn(n_tok, d_tok) * 0.02)
                self.token_bias = nn.Parameter(torch.zeros(n_tok, d_tok))

                # [CLS] token
                self.cls_token = nn.Parameter(torch.randn(1, 1, d_tok) * 0.02)

                # Transformer
                self.layer_norm = nn.LayerNorm(d_tok)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_tok,
                    nhead=cfg["n_heads"],
                    dim_feedforward=cfg["dim_ff"],
                    dropout=cfg["dropout"],
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer, num_layers=cfg["n_layers"]
                )

                # Head
                self.head = nn.Sequential(
                    nn.LayerNorm(d_tok),
                    nn.Linear(d_tok, d_tok // 2),
                    nn.GELU(),
                    nn.Dropout(cfg["dropout"]),
                    nn.Linear(d_tok // 2, 1),
                )

            def forward(self, x):
                # x: (batch, n_features)
                batch_size = x.size(0)

                # Bottleneck: (batch, n_features) -> (batch, n_tokens)
                x = self.bn_norm(self.bottleneck(x))

                # Token embedding: (batch, n_tokens) -> (batch, n_tokens, d_token)
                # x_i * weight_i + bias_i for each token
                x = x.unsqueeze(-1) * self.token_weight.unsqueeze(0) + self.token_bias.unsqueeze(0)

                # Prepend [CLS]
                cls = self.cls_token.expand(batch_size, -1, -1)
                x = torch.cat([cls, x], dim=1)  # (batch, 1 + n_tokens, d_token)

                x = self.layer_norm(x)
                x = self.encoder(x)

                # Extract [CLS] output
                cls_out = x[:, 0, :]
                return torch.sigmoid(self.head(cls_out)).squeeze(-1)

        return FTTransformerNet()

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "FTTransformerModel":
        torch, nn, optim, DataLoader, TensorDataset = _import_torch()

        X_train = self._prepare(X_train, fit=True)
        X_train, y_train, X_val, y_val = self._auto_val_split(X_train, y_train, X_val, y_val)
        if X_val is not None:
            X_val = self._prepare(X_val)

        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)

        n_features = X_train.shape[1]
        self._net = self._build_net(n_features, {
            "n_tokens": min(self.n_tokens, n_features),
            "d_token": self.d_token,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "dim_ff": self.dim_ff,
            "dropout": self.dropout,
        })

        optimizer = optim.AdamW(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr * 10, total_steps=self.epochs,
            pct_start=0.1, anneal_strategy="cos",
        )
        criterion = nn.BCELoss()

        train_ds = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_X_t = torch.from_numpy(X_val)
        val_y_t = torch.from_numpy(y_val)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        self._net.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for xb, yb in train_dl:
                optimizer.zero_grad()
                preds = self._net(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            epoch_loss /= len(train_ds)
            scheduler.step()

            self._net.eval()
            with torch.no_grad():
                val_preds = self._net(val_X_t)
                val_loss = criterion(val_preds, val_y_t).item()
            self._net.train()

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = copy.deepcopy(self._net.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        torch, _, _, _, _ = _import_torch()
        assert self._is_fitted, "Model not fitted yet"

        X = self._prepare(X)
        X_t = torch.from_numpy(X)

        self._net.eval()
        # Batch to avoid OOM on large inputs
        preds_list = []
        bs = self.batch_size
        for i in range(0, len(X_t), bs):
            with torch.no_grad():
                p = self._net(X_t[i : i + bs])
            preds_list.append(p.numpy())
        return np.concatenate(preds_list)


# ===========================================================================
# 5. Deep Ensemble
# ===========================================================================

class DeepEnsemble(BaseNBAModel):
    """
    Train N independent neural networks with different random seeds.

    Average their predictions for:
      - Better calibration (ensemble smoothing)
      - Uncertainty estimation (prediction variance)

    Each member is a simple but effective MLP with skip connections (ResNet-style),
    which is the 2025 consensus best architecture for tabular deep learning
    when ensembled (Kadra et al. 2021 "Well-Tuned Simple Nets").
    """

    def __init__(
        self,
        n_members: int = 10,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.3,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 512,
        epochs: int = 100,
        patience: int = 12,
        **kw,
    ):
        super().__init__(
            n_members=n_members, hidden_dims=list(hidden_dims),
            dropout=dropout, lr=lr, weight_decay=weight_decay,
            batch_size=batch_size, epochs=epochs, patience=patience, **kw,
        )
        self.n_members = n_members
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self._members: List = []

    @staticmethod
    def _build_mlp(n_features: int, hidden_dims: Tuple[int, ...], dropout: float, seed: int):
        """Build one ResNet-style MLP member."""
        torch, nn, _, _, _ = _import_torch()
        torch.manual_seed(seed)

        class ResBlock(nn.Module):
            """Pre-activation residual block."""
            def __init__(self, dim: int, drop: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                    nn.Dropout(drop),
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Linear(dim, dim),
                    nn.Dropout(drop),
                )

            def forward(self, x):
                return x + self.net(x)

        layers = []
        in_dim = n_features
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            # Add residual block at each hidden layer
            layers.append(ResBlock(h_dim, dropout))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))

        class EnsembleMLP(nn.Module):
            def __init__(self, layer_list):
                super().__init__()
                self.net = nn.Sequential(*layer_list)

            def forward(self, x):
                return torch.sigmoid(self.net(x)).squeeze(-1)

        return EnsembleMLP(layers)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DeepEnsemble":
        torch, nn, optim, DataLoader, TensorDataset = _import_torch()

        X_train = self._prepare(X_train, fit=True)
        X_train, y_train, X_val, y_val = self._auto_val_split(X_train, y_train, X_val, y_val)
        if X_val is not None:
            X_val = self._prepare(X_val)

        y_train = y_train.astype(np.float32)
        y_val = y_val.astype(np.float32)
        n_features = X_train.shape[1]

        val_X_t = torch.from_numpy(X_val)
        val_y_t = torch.from_numpy(y_val)
        criterion = nn.BCELoss()

        self._members = []
        for member_idx in range(self.n_members):
            seed = 42 + member_idx * 1337
            net = self._build_mlp(n_features, self.hidden_dims, self.dropout, seed)

            # Each member gets a different random seed for data shuffling too
            torch.manual_seed(seed)
            np.random.seed(seed)

            optimizer = optim.AdamW(
                net.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
            )

            train_ds = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

            best_val_loss = float("inf")
            best_state = None
            wait = 0

            net.train()
            for epoch in range(self.epochs):
                for xb, yb in train_dl:
                    optimizer.zero_grad()
                    preds = net(xb)
                    loss = criterion(preds, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    optimizer.step()

                net.eval()
                with torch.no_grad():
                    vp = net(val_X_t)
                    vl = criterion(vp, val_y_t).item()
                net.train()
                scheduler.step(vl)

                if vl < best_val_loss - 1e-6:
                    best_val_loss = vl
                    best_state = copy.deepcopy(net.state_dict())
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.patience:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self._members.append(net)

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return mean prediction across ensemble members."""
        torch, _, _, _, _ = _import_torch()
        assert self._is_fitted and self._members, "Model not fitted yet"

        X = self._prepare(X)
        X_t = torch.from_numpy(X)

        all_preds = []
        for net in self._members:
            net.eval()
            with torch.no_grad():
                p = net(X_t).numpy()
            all_preds.append(p)

        return np.mean(all_preds, axis=0)

    def predict_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (mean_prediction, std_prediction) across ensemble members.
        High std = high model uncertainty = less confident prediction.
        """
        torch, _, _, _, _ = _import_torch()
        assert self._is_fitted and self._members, "Model not fitted yet"

        X = self._prepare(X)
        X_t = torch.from_numpy(X)

        all_preds = []
        for net in self._members:
            net.eval()
            with torch.no_grad():
                p = net(X_t).numpy()
            all_preds.append(p)

        stacked = np.array(all_preds)  # (n_members, n_samples)
        return stacked.mean(axis=0), stacked.std(axis=0)


# ===========================================================================
# 6. Conformal Prediction Wrapper
# ===========================================================================

class ConformalPredictionWrapper(BaseNBAModel):
    """
    Wraps ANY model to provide calibrated prediction intervals with
    guaranteed coverage.

    Uses split conformal prediction:
      1. Train base model on training set
      2. Compute non-conformity scores on calibration holdout
      3. At inference, use quantile of scores to produce prediction sets

    For binary classification:
      - Returns P(home_win) from base model (point prediction)
      - Also provides prediction_set() that returns {0}, {1}, or {0,1}
        with guaranteed marginal coverage >= (1 - alpha)
    """

    def __init__(
        self,
        base_model: BaseNBAModel,
        alpha: float = 0.10,
        cal_fraction: float = 0.20,
        **kw,
    ):
        super().__init__(alpha=alpha, cal_fraction=cal_fraction, **kw)
        self.base_model = base_model
        self.alpha = alpha
        self.cal_fraction = cal_fraction
        self._qhat: Optional[float] = None
        self._cal_scores: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "ConformalPredictionWrapper":
        """
        Split data into proper-training and calibration sets.
        Train base model on proper-training, compute conformal scores on calibration.
        """
        n = len(X_train)
        cal_size = int(n * self.cal_fraction)
        # Use the LAST cal_size samples for calibration (time-ordered)
        X_proper = X_train[: n - cal_size]
        y_proper = y_train[: n - cal_size]
        X_cal = X_train[n - cal_size :]
        y_cal = y_train[n - cal_size :]

        # Train base model
        self.base_model.fit(X_proper, y_proper, X_val, y_val)

        # Compute non-conformity scores on calibration set
        cal_probs = self.base_model.predict_proba(X_cal)
        # Score = 1 - P(true_class)
        scores = np.where(y_cal == 1, 1.0 - cal_probs, cal_probs)
        self._cal_scores = np.sort(scores)

        # Quantile for desired coverage
        n_cal = len(self._cal_scores)
        level = np.ceil((1.0 - self.alpha) * (n_cal + 1)) / n_cal
        level = min(level, 1.0)
        self._qhat = np.quantile(self._cal_scores, level, method="higher")

        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions from base model."""
        assert self._is_fitted, "Model not fitted yet"
        return self.base_model.predict_proba(X)

    def predict_sets(self, X: np.ndarray) -> List[set]:
        """
        Return prediction sets with guaranteed (1-alpha) coverage.

        Each set is one of:
          - {1}    — confident home win
          - {0}    — confident away win
          - {0, 1} — uncertain (both plausible)
        """
        assert self._is_fitted, "Model not fitted yet"
        probs = self.base_model.predict_proba(X)
        sets = []
        for p in probs:
            s = set()
            # Include class 1 if score would be <= qhat
            if 1.0 - p <= self._qhat:
                s.add(1)
            # Include class 0 if score would be <= qhat
            if p <= self._qhat:
                s.add(0)
            if not s:
                # Shouldn't happen, but include most likely
                s.add(1 if p >= 0.5 else 0)
            sets.append(s)
        return sets

    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (lower_bound, upper_bound) calibrated probability intervals.

        Width of interval reflects model uncertainty after conformal calibration.
        """
        assert self._is_fitted, "Model not fitted yet"
        probs = self.base_model.predict_proba(X)
        lower = np.clip(probs - self._qhat, 0.0, 1.0)
        upper = np.clip(probs + self._qhat, 0.0, 1.0)
        return lower, upper

    def get_params(self) -> Dict[str, Any]:
        base_params = self.base_model.get_params()
        return {
            "wrapper": "conformal",
            "alpha": self.alpha,
            "cal_fraction": self.cal_fraction,
            "qhat": float(self._qhat) if self._qhat is not None else None,
            "base_model": base_params,
        }


# ===========================================================================
# 7. AutoGluon Ensemble
# ===========================================================================

class AutoGluonEnsemble(BaseNBAModel):
    """
    AutoGluon Tabular — auto-search and stack hundreds of model configurations.

    Time-budgeted: runs for *max_time* seconds, tries GBMs, neural nets,
    linear models, k-NN, then stacks the best ones.

    Presets: "best_quality" = maximum stacking/bagging (slow but best),
             "good_quality" = reasonable speed/quality trade-off,
             "medium_quality" = fastest.
    """

    def __init__(
        self,
        max_time: int = 3600,
        preset: str = "best_quality",
        eval_metric: str = "log_loss",
        num_bag_folds: int = 5,
        num_stack_levels: int = 1,
        verbosity: int = 1,
        **kw,
    ):
        super().__init__(
            max_time=max_time, preset=preset, eval_metric=eval_metric,
            num_bag_folds=num_bag_folds, num_stack_levels=num_stack_levels,
            verbosity=verbosity, **kw,
        )
        self.max_time = max_time
        self.preset = preset
        self.eval_metric = eval_metric
        self.num_bag_folds = num_bag_folds
        self.num_stack_levels = num_stack_levels
        self.verbosity = verbosity
        self._predictor = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "AutoGluonEnsemble":
        try:
            from autogluon.tabular import TabularPredictor
            import pandas as pd
        except ImportError:
            raise ImportError(
                "autogluon.tabular not installed. Install with: "
                "pip install autogluon.tabular"
            )

        X_train = self._impute(X_train, fit=True)

        # Build DataFrame with feature columns + label
        n_features = X_train.shape[1]
        col_names = [f"f_{i}" for i in range(n_features)]
        df_train = pd.DataFrame(X_train, columns=col_names)
        df_train["label"] = y_train.astype(int)

        # Validation data (optional tuning set)
        df_val = None
        if X_val is not None and y_val is not None:
            X_val = self._impute(X_val)
            df_val = pd.DataFrame(X_val, columns=col_names)
            df_val["label"] = y_val.astype(int)

        self._col_names = col_names

        self._predictor = TabularPredictor(
            label="label",
            eval_metric=self.eval_metric,
            problem_type="binary",
            verbosity=self.verbosity,
        )

        fit_kwargs = {
            "train_data": df_train,
            "time_limit": self.max_time,
            "presets": self.preset,
            "num_bag_folds": self.num_bag_folds,
            "num_stack_levels": self.num_stack_levels,
        }
        if df_val is not None:
            fit_kwargs["tuning_data"] = df_val

        self._predictor.fit(**fit_kwargs)
        self._is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import pandas as pd

        assert self._is_fitted, "Model not fitted yet"
        X = self._impute(X)
        df = pd.DataFrame(X, columns=self._col_names)
        proba = self._predictor.predict_proba(df)
        # Returns DataFrame with columns 0, 1 — we want P(class=1)
        if isinstance(proba, pd.DataFrame):
            return proba[1].values
        return proba

    def leaderboard(self):
        """Return AutoGluon model leaderboard."""
        assert self._is_fitted, "Model not fitted yet"
        return self._predictor.leaderboard(silent=True)

    def feature_importance(self, X: np.ndarray, y: np.ndarray) -> "pd.DataFrame":
        """Return permutation feature importance."""
        import pandas as pd

        X = self._impute(X)
        df = pd.DataFrame(X, columns=self._col_names)
        df["label"] = y.astype(int)
        return self._predictor.feature_importance(df)

    def save(self, path: Union[str, Path]) -> None:
        """AutoGluon has its own save mechanism."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self._predictor is not None:
            self._predictor.save(str(path / "autogluon_predictor"))
        # Save wrapper state
        state = {
            "params": self.params,
            "_col_names": getattr(self, "_col_names", None),
            "_feature_medians": self._feature_medians.tolist() if self._feature_medians is not None else None,
            "_is_fitted": self._is_fitted,
        }
        with open(path / "wrapper_state.json", "w") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AutoGluonEnsemble":
        from autogluon.tabular import TabularPredictor

        path = Path(path)
        with open(path / "wrapper_state.json") as f:
            state = json.load(f)

        obj = cls(**state["params"])
        obj._col_names = state["_col_names"]
        if state["_feature_medians"] is not None:
            obj._feature_medians = np.array(state["_feature_medians"], dtype=np.float32)
        obj._predictor = TabularPredictor.load(str(path / "autogluon_predictor"))
        obj._is_fitted = state["_is_fitted"]
        return obj


# ===========================================================================
# Utilities
# ===========================================================================

def _is_jsonable(v: Any) -> bool:
    """Check if a value is JSON serialisable."""
    try:
        json.dumps(v)
        return True
    except (TypeError, OverflowError, ValueError):
        return False


# ===========================================================================
# Model Registry — maps names to classes for the genetic algorithm
# ===========================================================================

NEURAL_MODEL_REGISTRY: Dict[str, type] = {
    "lstm": LSTMSequenceModel,
    "transformer": TransformerAttentionModel,
    "tabnet": TabNetModel,
    "ft_transformer": FTTransformerModel,
    "deep_ensemble": DeepEnsemble,
    "conformal": ConformalPredictionWrapper,
    "autogluon": AutoGluonEnsemble,
}


def build_neural_model(model_type: str, **params) -> BaseNBAModel:
    """
    Factory function to build a neural model by name.

    Usage:
        model = build_neural_model("ft_transformer", n_tokens=128, d_token=64)
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)

    For conformal wrapper, pass base_model_type and base_model_params:
        model = build_neural_model(
            "conformal",
            base_model_type="deep_ensemble",
            base_model_params={"n_members": 5},
            alpha=0.1,
        )
    """
    if model_type == "conformal":
        base_type = params.pop("base_model_type", "deep_ensemble")
        base_params = params.pop("base_model_params", {})
        base_model = build_neural_model(base_type, **base_params)
        return ConformalPredictionWrapper(base_model=base_model, **params)

    cls = NEURAL_MODEL_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: {list(NEURAL_MODEL_REGISTRY.keys())}"
        )
    return cls(**params)


# ===========================================================================
# Quick smoke test (runs if executed directly)
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Quant AI — Neural Models Smoke Test")
    print("=" * 60)

    np.random.seed(42)
    N_TRAIN, N_TEST, N_FEAT = 500, 100, 200

    X_train = np.random.randn(N_TRAIN, N_FEAT).astype(np.float32)
    # Inject some NaNs to test imputation
    mask = np.random.random(X_train.shape) < 0.05
    X_train[mask] = np.nan
    y_train = (np.random.random(N_TRAIN) > 0.5).astype(np.float32)

    X_test = np.random.randn(N_TEST, N_FEAT).astype(np.float32)
    y_test = (np.random.random(N_TEST) > 0.5).astype(np.float32)

    # Test each model (with small configs for speed)
    tests = [
        ("FT-Transformer", FTTransformerModel(
            n_tokens=32, d_token=16, n_heads=2, n_layers=1,
            epochs=5, patience=3, batch_size=128,
        )),
        ("Deep Ensemble (3 members)", DeepEnsemble(
            n_members=3, hidden_dims=(64, 32),
            epochs=5, patience=3, batch_size=128,
        )),
        ("LSTM Sequence", LSTMSequenceModel(
            seq_len=5, hidden1=32, hidden2=16, dense_dim=16,
            epochs=5, patience=3, batch_size=128,
        )),
        ("Transformer Attention", TransformerAttentionModel(
            seq_len=5, d_model=32, n_heads=2, n_layers=1,
            dim_ff=64, epochs=5, patience=3, batch_size=128,
        )),
    ]

    for name, model in tests:
        print(f"\n--- {name} ---")
        try:
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)
            print(f"  Predictions shape: {probs.shape}")
            print(f"  Mean pred: {probs.mean():.4f}, Std: {probs.std():.4f}")
            print(f"  Min: {probs.min():.4f}, Max: {probs.max():.4f}")
            print(f"  Params: {list(model.get_params().keys())}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test conformal wrapper
    print("\n--- Conformal Prediction Wrapper ---")
    try:
        base = DeepEnsemble(
            n_members=2, hidden_dims=(64, 32),
            epochs=5, patience=3, batch_size=128,
        )
        conformal = ConformalPredictionWrapper(base_model=base, alpha=0.1)
        conformal.fit(X_train, y_train)
        probs = conformal.predict_proba(X_test)
        sets = conformal.predict_sets(X_test)
        lower, upper = conformal.predict_intervals(X_test)
        print(f"  Point preds shape: {probs.shape}")
        print(f"  Prediction sets (first 5): {sets[:5]}")
        print(f"  Intervals: [{lower[:3]}] - [{upper[:3]}]")
        print(f"  Avg interval width: {(upper - lower).mean():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test TabNet (may fail if pytorch_tabnet not installed)
    print("\n--- TabNet ---")
    try:
        tab = TabNetModel(
            n_d=8, n_a=8, n_steps=3, epochs=5, patience=3, batch_size=128,
        )
        tab.fit(X_train, y_train)
        probs = tab.predict_proba(X_test)
        print(f"  Predictions shape: {probs.shape}")
        print(f"  Mean pred: {probs.mean():.4f}")
        fi = tab.get_feature_importances()
        if fi is not None:
            print(f"  Feature importances shape: {fi.shape}")
    except ImportError:
        print("  SKIPPED (pytorch_tabnet not installed)")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Test factory
    print("\n--- Factory: build_neural_model ---")
    try:
        m = build_neural_model("ft_transformer", n_tokens=32, d_token=16,
                               n_heads=2, n_layers=1, epochs=3, batch_size=128)
        m.fit(X_train, y_train)
        print(f"  Factory FT-Transformer OK, preds mean: {m.predict_proba(X_test).mean():.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("Smoke test complete.")
    print("=" * 60)
