import argparse
import copy
import typing as tp
import warnings
from random import Random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DatasetDict, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

DEFAULT_RUNS_PER_SESSION = 5
N_FOLDS = 5
FIRST_TEST_SESSION = 19


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TemporalDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True,
        )  # baseline temporal model, nothing clever yet

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        h = self.dropout(h[-1])
        return self.head(h)


def get_y(batch: dict[str, torch.Tensor], target: str) -> torch.Tensor:
    """Helper to fetch or concatenate the chosen target."""
    if target == "both":
        return torch.cat([batch["Y_text"], batch["Y_audio"]], dim=-1)
    return batch[f"Y_{target}"]


def clip_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    logits = pred @ target.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def compute_retrieval_ranks(
    Y_pred: torch.Tensor,  # (B_1, D)
    pos_sim: torch.Tensor,  # (B_1,)
    loader: DataLoader,
    target: str,
    device: str,
    pbar: tqdm | None = None,
) -> torch.Tensor:
    Y_pred_n = F.normalize(Y_pred, dim=-1)  # (B_1, D)

    ranks = torch.zeros(Y_pred.shape[0], device=Y_pred.device)
    for batch in loader:
        Y_batch = get_y(batch, target=target).to(device)  # (B_2, D)
        neg_sim = Y_pred_n @ F.normalize(Y_batch, dim=-1).T  # (B_1, B_2)
        ranks += (neg_sim > pos_sim[:, None]).sum(dim=-1).float()

        if pbar:
            pbar.update(len(Y_batch))

    return ranks


def eval_metrics(model, loader, device, target, split_name=""):
    cosine_sims, ranks = [], []
    pbar = tqdm(total=len(loader.dataset) * len(loader), desc=f"Evaluating on {split_name}", leave=False)
    with pbar:
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Evaluating {split_name}", leave=False):
                X = batch["X"].to(device).float()
                Y_true = get_y(batch, target).to(device).float()
                Y_pred = model(X)
                pos_sim = F.cosine_similarity(Y_pred, Y_true, dim=-1)
                cosine_sims.append(pos_sim.cpu())
                ranks.append(compute_retrieval_ranks(Y_pred, pos_sim, loader, target, device, pbar))

    cosine_sim = torch.cat(cosine_sims).mean().item()
    ranks = torch.cat(ranks)

    return {
        "Cosine Similarity (↑)": cosine_sim,
        "Median Relative Rank (↓)": ((ranks + 1) / len(loader.dataset)).median().item(),
        "Top-10 Retrieval Accuracy % (↑)": (ranks < 10).float().mean().item() * 100.0,
    }


def select_training_sessions(dataset, n_sessions, fold):
    """Select nested training subsets with a deterministic shuffle per fold."""
    metadata = dataset.select_columns(["ses", "run"]).to_pandas()
    runs_per_session = metadata.groupby("ses")["run"].nunique()
    n_available_sessions = len(runs_per_session)

    if n_sessions < 1 or n_sessions > n_available_sessions:
        raise ValueError(f"Requested {n_sessions} sessions, but only {n_available_sessions} are available")

    # Full training includes all sessions, even those with fewer or more than five runs.
    if n_sessions == n_available_sessions:
        return dataset

    # Shuffle once per fold so smaller subsets are contained in larger ones.
    complete_sessions = runs_per_session.index[runs_per_session == DEFAULT_RUNS_PER_SESSION].tolist()
    Random(fold).shuffle(complete_sessions)
    if n_sessions > len(complete_sessions):
        raise ValueError(f"Only {len(complete_sessions)} complete five-run sessions are available")

    selected_sessions = complete_sessions[:n_sessions]
    indices = np.flatnonzero(metadata["ses"].isin(selected_sessions))
    return dataset.select(indices)


def train(
    sub: tp.Literal["0", "1", "2"] = "0",
    space: tp.Literal["fsaverage5", "T1w", "MNI"] = "fsaverage5",
    target: tp.Literal["text", "audio", "both"] = "text",
    n_train_sessions: int | None = None,
    fold: int | None = None,
    batch_size: int = 2048,
    max_epochs: int = 100,
    patience: int = 5,
    lr: float = 3e-4,
    weight_decay: float = 3e-4,
    temperature: float = 2e-2,
    bottleneck_dim: int = 4096,
    dropout: float = 0.5,
    seed: int = 0,
    device: str | None = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset = load_dataset("karavela/cephalonauts_one", name=f"sub-{sub}_{space}")

    # If fold is provided, replace the Hugging Face splits with session-based train/val/test splits.
    if fold is not None:
        dataset = concatenate_datasets([dataset[split] for split in ("train", "val", "test")])
        ses = dataset.with_format("pandas", columns=["ses"])["ses"]

        if fold not in range(N_FOLDS):
            raise ValueError(f"fold must be between 0 and {N_FOLDS - 1}")

        test_session = FIRST_TEST_SESSION + fold
        val_session = FIRST_TEST_SESSION + (fold + 1) % N_FOLDS
        print(f"Fold {fold}: test session = {test_session}, val session = {val_session}")

        dataset = DatasetDict({
            "train": dataset.select(ses[~ses.isin([test_session, val_session])].index),
            "val": dataset.select(ses[ses == val_session].index),
            "test": dataset.select(ses[ses == test_session].index),
        })

    # Select training sessions after holding out validation and test sessions.
    if n_train_sessions is not None:
        selection_seed = fold if fold is not None else seed
        dataset["train"] = select_training_sessions(dataset["train"], n_train_sessions, selection_seed)

    # Reduce the batch size when the training subset is too small.
    dataset = dataset.with_format("torch")
    if len(dataset["train"]) < batch_size:
        warnings.warn(
            f"Requested batch size {batch_size} is larger than train split size {len(dataset['train'])}, reducing batch size to {len(dataset['train'])}"
        )
        batch_size = len(dataset["train"])

    train_dl = DataLoader(dataset["train"], batch_size=batch_size, shuffle=len(dataset["train"]) > 0, drop_last=True)
    val_dl = DataLoader(dataset["val"], batch_size=batch_size)
    test_dl = DataLoader(dataset["test"], batch_size=batch_size)

    print(
        f"Train samples: {len(train_dl.dataset)} | Val samples: {len(val_dl.dataset)} | Test samples: {len(test_dl.dataset)}"
    )
    print(f"Training on device: {device}, targeting: {target}\n")

    # Infer model dimensions from one batch.
    sample = next(iter(test_dl))
    y_dim = get_y(sample, target).shape[1]
    model = Decoder(sample["X"].shape[1], y_dim, bottleneck_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rank = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    best_epoch = 0

    # Train for one epoch, then check validation performance.
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_dl, desc=f"Epoch {epoch}", leave=False):
            X = batch["X"].to(device).float()
            Y_true = get_y(batch, target).to(device).float()
            optimizer.zero_grad(set_to_none=True)
            loss = clip_loss(model(X), Y_true, temperature)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_metrics = eval_metrics(model, val_dl, device, target, split_name="val")
        val_rank = val_metrics["Median Relative Rank (↓)"]
        print(
            f"Epoch {epoch:3d} / {max_epochs} | Train Loss (↓): {train_loss / len(train_dl):8.4f} | Val Median Relative Rank (↓): {val_rank:8.4f}"
        )

        # Keep the best checkpoint; stop after too many epochs without improvement.
        if val_rank < best_val_rank:
            best_val_rank, best_epoch, epochs_without_improvement = val_rank, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    # Evaluate the best checkpoint on the held-out test split.
    model.load_state_dict(best_state)
    model.eval()
    test_results = eval_metrics(model, test_dl, device, target, split_name="test")

    # Count each source run once when reporting training hours.
    train_runs = dataset["train"].select_columns(["ses", "run", "run_fmri_duration"]).to_pandas()
    train_hours = train_runs.drop_duplicates(["ses", "run"])["run_fmri_duration"].sum() / 3600
    test_results["Training Data (hours)"] = train_hours

    print(f"\nBest Epoch: {best_epoch}")
    print("\nTest Metrics:")
    for key, value in test_results.items():
        print(f"  {key:.<40} {value:.3f}")

    return test_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    p.add_argument("--sub", choices=["0", "1", "2"], default="0")
    p.add_argument("--space", choices=["fsaverage5", "T1w", "MNI"], default="fsaverage5")
    p.add_argument("--target", choices=["text", "audio", "both"])
    p.add_argument(
        "--n-train-sessions", type=int,
        help="Number of training sessions to use (default: all available training sessions).",
    )
    p.add_argument("--fold", type=int, help="Evaluation fold, 0–4 (default: use the published Hugging Face splits).")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--bottleneck-dim", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--device")

    train(**vars(p.parse_args()))
