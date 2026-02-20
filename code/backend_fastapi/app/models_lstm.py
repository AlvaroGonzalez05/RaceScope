from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .config import DEFAULT_CONTEXT_LAPS


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMPaceNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


@dataclass
class ModelBundle:
    model_state: Dict
    encoders: Dict[str, Dict]
    stats: Dict[str, float]


class LSTMPaceModel:
    def __init__(self, context_len: int = DEFAULT_CONTEXT_LAPS):
        self.context_len = context_len
        self.model: LSTMPaceNet | None = None
        self.encoders: Dict[str, Dict] = {}
        self.stats: Dict[str, float] = {}

    def _encode(self, series: pd.Series, encoder: Dict) -> np.ndarray:
        return series.map(lambda x: encoder.get(x, 0)).fillna(0).astype(int).values

    def _build_encoders(self, df: pd.DataFrame) -> None:
        self.encoders["compound"] = {v: i + 1 for i, v in enumerate(sorted(df["compound"].dropna().unique()))}
        self.encoders["session_type"] = {v: i + 1 for i, v in enumerate(sorted(df["session_type"].dropna().unique()))}
        self.encoders["circuit_id"] = {v: i + 1 for i, v in enumerate(sorted(df["circuit_id"].dropna().unique()))}

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        df = df.sort_values(["session_key", "driver_id", "lap_number"])
        self.stats["lap_mean"] = df["lap_time"].mean()
        self.stats["lap_std"] = df["lap_time"].std() or 1.0
        df["lap_norm"] = (df["lap_time"] - self.stats["lap_mean"]) / self.stats["lap_std"]

        self._build_encoders(df)

        compounds = self._encode(df["compound"], self.encoders["compound"])
        session_types = self._encode(df["session_type"], self.encoders["session_type"])
        circuits = self._encode(df["circuit_id"], self.encoders["circuit_id"])

        features = np.column_stack([
            df["lap_number"].values,
            df["stint_age"].values,
            compounds,
            session_types,
            circuits,
            df["track_temp"].fillna(df["track_temp"].mean()).values,
            df["air_temp"].fillna(df["air_temp"].mean()).values,
            df["lap_norm"].values,
        ])

        X, y = [], []
        for i in range(self.context_len, len(features)):
            if df.iloc[i]["session_key"] != df.iloc[i - 1]["session_key"]:
                continue
            X.append(features[i - self.context_len : i])
            y.append(df.iloc[i]["lap_norm"])

        if not X:
            return np.empty((0, self.context_len, features.shape[1])), np.empty((0, 1))

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

    def train(self, df: pd.DataFrame, epochs: int = 8, batch_size: int = 128) -> ModelBundle:
        X, y = self._prepare_sequences(df)
        if len(X) == 0:
            raise ValueError("No sequences to train on.")

        input_dim = X.shape[-1]
        self.model = LSTMPaceNet(input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        dataset = SequenceDataset(torch.tensor(X), torch.tensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for _ in range(epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                optimizer.step()

        return ModelBundle(self.model.state_dict(), self.encoders, self.stats)

    def load(self, bundle: ModelBundle, input_dim: int) -> None:
        self.model = LSTMPaceNet(input_dim)
        self.model.load_state_dict(bundle.model_state)
        self.encoders = bundle.encoders
        self.stats = bundle.stats

    def predict_stint(self, stint_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded.")

        df = stint_df.copy()
        df["lap_norm"] = (df["lap_time"] - self.stats["lap_mean"]) / self.stats["lap_std"]

        compounds = self._encode(df["compound"], self.encoders["compound"])
        session_types = self._encode(df["session_type"], self.encoders["session_type"])
        circuits = self._encode(df["circuit_id"], self.encoders["circuit_id"])

        features = np.column_stack([
            df["lap_number"].values,
            df["stint_age"].values,
            compounds,
            session_types,
            circuits,
            df["track_temp"].fillna(df["track_temp"].mean()).values,
            df["air_temp"].fillna(df["air_temp"].mean()).values,
            df["lap_norm"].values,
        ])

        X = []
        for i in range(self.context_len, len(features)):
            X.append(features[i - self.context_len : i])

        if not X:
            return df["lap_time"].values

        self.model.eval()
        with torch.no_grad():
            preds = self.model(torch.tensor(np.array(X, dtype=np.float32)))
        preds = preds.squeeze().numpy()
        preds = np.atleast_1d(preds) * self.stats["lap_std"] + self.stats["lap_mean"]

        warmup = df["lap_time"].values[: self.context_len]
        return np.concatenate([warmup, preds])
