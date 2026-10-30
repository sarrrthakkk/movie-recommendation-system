"""Biased matrix factorization trained with PyTorch."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn

from movie_recommender.evaluation import ndcg_at_k, rank_of
from movie_recommender.models.hybrid import rank_order
from movie_recommender.utils import get_logger, load_json, resolve_device, save_json, set_seed

logger = get_logger(__name__)


class BiasedMF(nn.Module):
    """``r_hat = mu + b_u + b_i + p_u · q_i``."""

    def __init__(self, n_users: int, n_items: int, n_factors: int, mu: float) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(n_users, n_factors)
        self.item_factors = nn.Embedding(n_items, n_factors)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.register_buffer("mu", torch.tensor(float(mu), dtype=torch.float32))
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_vectors = self.user_factors(users)
        item_vectors = self.item_factors(items)
        prediction = (
            self.mu
            + self.user_bias(users).squeeze(-1)
            + self.item_bias(items).squeeze(-1)
            + (user_vectors * item_vectors).sum(dim=-1)
        )
        return prediction


class MatrixFactorizationRecommender:
    """Mini-batch biased MF with validation NDCG early stopping.

    The validation split is used only to choose the epoch. It is never appended
    to the training interactions. Early stopping watches validation NDCG@10
    because every validation label is a relevant rating, so RMSE on that slice
    is not comparable to training error. RMSE and MAE are still recorded and
    reported as secondary rating metrics.
    """

    def __init__(self) -> None:
        self.model: BiasedMF | None = None
        self.device = torch.device("cpu")
        self.user_ids: list[int] = []
        self.movie_ids: list[int] = []
        self.user_to_index: dict[int, int] = {}
        self.movie_to_index: dict[int, int] = {}
        self.history: dict[str, list[float]] = {"train_loss": [], "val_rmse": [], "val_mae": []}
        self.metadata: dict = {}

    def fit(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        movie_ids: np.ndarray,
        *,
        n_factors: int,
        learning_rate: float,
        regularization: float,
        epochs: int,
        batch_size: int,
        patience: int,
        seed: int,
        device: str | None = None,
        checkpoint_dir: Path | None = None,
    ) -> dict:
        overlap = set(zip(train["user_id"].astype(int), train["movie_id"].astype(int))) & set(
            zip(validation["user_id"].astype(int), validation["movie_id"].astype(int))
        )
        if overlap:
            raise RuntimeError("Refusing to train: validation pairs are present in training")
        set_seed(seed)
        self.device = resolve_device(device)
        self.user_ids = sorted(int(user_id) for user_id in train["user_id"].unique())
        self.movie_ids = sorted(int(movie_id) for movie_id in movie_ids)
        self.user_to_index = {user_id: index for index, user_id in enumerate(self.user_ids)}
        self.movie_to_index = {movie_id: index for index, movie_id in enumerate(self.movie_ids)}
        mu = float(train["rating"].mean()) if not train.empty else 0.0
        self.model = BiasedMF(len(self.user_ids), len(self.movie_ids), n_factors, mu).to(self.device)
        # SGD matches the usual scale of the configured learning rate. Adam at
        # 0.005 takes steps that overfit this dataset within a single epoch.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        user_index = train["user_id"].map(self.user_to_index).to_numpy(dtype=np.int64)
        item_index = train["movie_id"].map(self.movie_to_index).to_numpy(dtype=np.int64)
        if np.isnan(user_index.astype(float)).any() or np.isnan(item_index.astype(float)).any():
            raise RuntimeError("Training rows contain ids that were not mapped")
        ratings = train["rating"].to_numpy(dtype=np.float32)
        val_users, val_items, val_ratings = self._validation_arrays(validation)
        seen_by_user = {
            int(user_id): set(frame["movie_id"].astype(int))
            for user_id, frame in train.groupby("user_id")
        }
        val_targets = (
            {
                int(row.user_id): int(row.movie_id)
                for row in validation.itertuples(index=False)
            }
            if not validation.empty
            else {}
        )
        catalog = np.asarray(self.movie_ids, dtype=int)
        rng = np.random.default_rng(seed)
        best_ndcg = -1.0
        best_state = None
        best_epoch = 0
        waits = 0
        self.history = {"train_loss": [], "val_rmse": [], "val_mae": [], "val_ndcg": []}
        logger.info(
            "Training biased MF on %s: users=%d movies=%d factors=%d epochs=%d",
            self.device,
            len(self.user_ids),
            len(self.movie_ids),
            n_factors,
            epochs,
        )
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(
                user_index, item_index, ratings, optimizer, regularization, batch_size, rng
            )
            val_rmse, val_mae = self._score_pairs(val_users, val_items, val_ratings)
            val_ndcg = self._mean_validation_ndcg(catalog, seen_by_user, val_targets)
            self.history["train_loss"].append(float(train_loss))
            self.history["val_rmse"].append(float(val_rmse))
            self.history["val_mae"].append(float(val_mae))
            self.history["val_ndcg"].append(float(val_ndcg))
            logger.info(
                "epoch %d train_loss=%.4f val_rmse=%.4f val_mae=%.4f val_ndcg@10=%.4f",
                epoch,
                train_loss,
                val_rmse,
                val_mae,
                val_ndcg,
            )
            if val_ndcg > best_ndcg + 1e-4:
                best_ndcg = val_ndcg
                best_epoch = epoch
                waits = 0
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_dir is not None:
                    self._write_checkpoint(checkpoint_dir, seed, n_factors, best_epoch, epoch)
            else:
                waits += 1
                if waits >= patience:
                    logger.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        self.metadata = {
            "seed": seed,
            "mu": mu,
            "latent_dimensions": n_factors,
            "best_epoch": best_epoch,
            "epochs_ran": len(self.history["train_loss"]),
            "best_val_ndcg@10": None if best_ndcg < 0 else best_ndcg,
            "early_stopping_metric": "validation_ndcg@10",
            "device": str(self.device),
            "learning_rate": learning_rate,
            "regularization": regularization,
            "batch_size": batch_size,
            "patience": patience,
        }
        if checkpoint_dir is not None:
            self._write_checkpoint(checkpoint_dir, seed, n_factors, best_epoch, len(self.history["train_loss"]))
        return self.metadata

    def _train_epoch(
        self,
        user_index: np.ndarray,
        item_index: np.ndarray,
        ratings: np.ndarray,
        optimizer: torch.optim.Optimizer,
        regularization: float,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        assert self.model is not None
        self.model.train()
        permutation = rng.permutation(len(ratings))
        total = 0.0
        seen = 0
        for start in range(0, len(ratings), batch_size):
            batch = permutation[start : start + batch_size]
            users = torch.tensor(user_index[batch].copy(), dtype=torch.long, device=self.device)
            items = torch.tensor(item_index[batch].copy(), dtype=torch.long, device=self.device)
            target = torch.tensor(ratings[batch].copy(), dtype=torch.float32, device=self.device)
            prediction = self.model(users, items)
            mse = torch.mean((prediction - target) ** 2)
            l2 = (
                self.model.user_factors(users).pow(2).sum()
                + self.model.item_factors(items).pow(2).sum()
                + self.model.user_bias(users).pow(2).sum()
                + self.model.item_bias(items).pow(2).sum()
            ) / users.shape[0]
            loss = mse + regularization * l2
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        return total / max(seen, 1)

    def _validation_arrays(
        self, validation: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if validation.empty:
            return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32)
        users = validation["user_id"].map(self.user_to_index).to_numpy(dtype=np.int64)
        items = validation["movie_id"].map(self.movie_to_index).to_numpy(dtype=np.int64)
        ratings = validation["rating"].to_numpy(dtype=np.float32)
        return users, items, ratings

    def _score_pairs(
        self, users: np.ndarray, items: np.ndarray, ratings: np.ndarray
    ) -> tuple[float, float]:
        if len(ratings) == 0:
            return float("nan"), float("nan")
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(
                torch.tensor(np.array(users, dtype=np.int64, copy=True), dtype=torch.long, device=self.device),
                torch.tensor(np.array(items, dtype=np.int64, copy=True), dtype=torch.long, device=self.device),
            )
        predicted = prediction.detach().cpu().numpy()
        rmse = float(np.sqrt(mean_squared_error(ratings, predicted)))
        mae = float(mean_absolute_error(ratings, predicted))
        return rmse, mae

    def _mean_validation_ndcg(
        self,
        catalog: np.ndarray,
        seen_by_user: dict[int, set[int]],
        targets: dict[int, int],
        k: int = 10,
    ) -> float:
        """Mean leave-one-out NDCG@K on the validation targets. Higher is better."""
        if not targets:
            return float("nan")
        assert self.model is not None
        self.model.eval()
        scores: list[float] = []
        for user_id, target in targets.items():
            seen = seen_by_user.get(user_id, set())
            unseen = catalog[~np.isin(catalog, list(seen))]
            ranked = unseen[rank_order(unseen, self.score_movies(user_id, unseen))]
            scores.append(ndcg_at_k(rank_of(ranked, target), k))
        return float(np.mean(scores)) if scores else float("nan")

    def predict(self, user_id: int, movie_id: int) -> float:
        scores = self.score_movies(user_id, np.asarray([movie_id], dtype=int))
        return float(scores[0])

    def score_movies(self, user_id: int, movie_ids: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Matrix factorization model is not loaded")
        if int(user_id) not in self.user_to_index:
            raise KeyError(f"No matrix-factorization embedding for user {user_id}")
        ids = np.asarray(movie_ids, dtype=int)
        missing = [int(movie_id) for movie_id in ids if int(movie_id) not in self.movie_to_index]
        if missing:
            raise KeyError(f"Unknown movie ids: {missing[:5]}")
        if ids.size == 0:
            return np.asarray([], dtype=float)
        user_index = self.user_to_index[int(user_id)]
        item_index = [self.movie_to_index[int(movie_id)] for movie_id in ids]
        self.model.eval()
        with torch.no_grad():
            user_vector = self.model.user_factors.weight[user_index]
            user_bias = self.model.user_bias.weight[user_index, 0]
            items = torch.as_tensor(item_index, dtype=torch.long, device=self.device)
            item_vectors = self.model.item_factors.weight[items]
            item_bias = self.model.item_bias.weight[items, 0]
            scores = self.model.mu + user_bias + item_bias + (item_vectors * user_vector).sum(dim=-1)
        return scores.detach().cpu().numpy().astype(float)

    def recommend(
        self, user_id: int, seen_movie_ids: set[int] | list[int], k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        seen = {int(movie_id) for movie_id in seen_movie_ids}
        candidates = np.asarray([movie_id for movie_id in self.movie_ids if movie_id not in seen], dtype=int)
        if len(candidates) == 0 or k < 1:
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        scores = self.score_movies(user_id, candidates)
        order = rank_order(candidates, scores)[:k]
        return candidates[order], scores[order]

    def rating_metrics(self, frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {"rmse": float("nan"), "mae": float("nan"), "n": 0}
        users = frame["user_id"].map(self.user_to_index).to_numpy(dtype=np.int64)
        items = frame["movie_id"].map(self.movie_to_index).to_numpy(dtype=np.int64)
        ratings = frame["rating"].to_numpy(dtype=np.float32)
        rmse, mae = self._score_pairs(users, items, ratings)
        return {"rmse": rmse, "mae": mae, "n": int(len(frame))}

    def _write_checkpoint(
        self, checkpoint_dir: Path, seed: int, n_factors: int, best_epoch: int, epochs_ran: int
    ) -> None:
        assert self.model is not None
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), checkpoint_dir / "mf.pt")
        save_json(
            checkpoint_dir / "mf_metadata.json",
            {
                **self.metadata,
                "seed": seed,
                "latent_dimensions": n_factors,
                "best_epoch": best_epoch,
                "epochs_ran": epochs_ran,
                "mu": float(self.model.mu.detach().cpu()),
                "user_ids": self.user_ids,
                "movie_ids": self.movie_ids,
                "history": self.history,
                "device_trained": str(self.device),
            },
        )

    def save(self, checkpoint_dir: str | Path) -> None:
        self._write_checkpoint(
            Path(checkpoint_dir),
            int(self.metadata.get("seed", 0)),
            int(self.metadata.get("latent_dimensions", 0)),
            int(self.metadata.get("best_epoch", 0)),
            int(self.metadata.get("epochs_ran", 0)),
        )

    @classmethod
    def load(cls, checkpoint_dir: str | Path, device: str | None = None) -> "MatrixFactorizationRecommender":
        directory = Path(checkpoint_dir)
        metadata_path = directory / "mf_metadata.json"
        weights_path = directory / "mf.pt"
        if not metadata_path.exists() or not weights_path.exists():
            raise FileNotFoundError(
                f"Matrix factorization checkpoint not found in {directory}. Run python scripts/train_models.py."
            )
        metadata = load_json(metadata_path)
        recommender = cls()
        recommender.metadata = metadata
        recommender.history = metadata.get("history", {"train_loss": [], "val_rmse": [], "val_mae": []})
        recommender.user_ids = [int(user_id) for user_id in metadata["user_ids"]]
        recommender.movie_ids = [int(movie_id) for movie_id in metadata["movie_ids"]]
        recommender.user_to_index = {user_id: index for index, user_id in enumerate(recommender.user_ids)}
        recommender.movie_to_index = {movie_id: index for index, movie_id in enumerate(recommender.movie_ids)}
        recommender.device = resolve_device(device)
        model = BiasedMF(
            len(recommender.user_ids),
            len(recommender.movie_ids),
            int(metadata["latent_dimensions"]),
            float(metadata["mu"]),
        )
        state = torch.load(weights_path, map_location=recommender.device, weights_only=True)
        model.load_state_dict(state)
        model.to(recommender.device)
        model.eval()
        recommender.model = model
        return recommender
