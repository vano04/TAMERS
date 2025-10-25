from pathlib import Path
from typing import cast
import numpy as np
import torch
from torch.amp.autocast_mode import autocast
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm

from data.dataloader import get_dataloader, MELD_LABELS
from model import (
    CLAPFusionHead,
    audio_embed,
    clap_infonce,
    collate_paired,
    set_model_device,
    text_embed,
)

checkpoint_path = Path("checkpoints/final_MELD_norm_ft.pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_model_device(device)


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """Return a copy of ``state_dict`` with ``prefix`` removed when all keys match."""
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Handle common wrappers (DDP, module submodules) found in checkpoints."""
    prefixes = ("module.", "head.", "fusion_head.")
    for prefix in prefixes:
        state_dict = _strip_prefix(state_dict, prefix)
    return state_dict


def load_fusion_head(checkpoint_file: Path, device: torch.device) -> tuple[CLAPFusionHead, dict]:
    # Reload the fusion head with hyperparameters saved alongside the checkpoint.
    package = torch.load(checkpoint_file, map_location=device)
    if isinstance(package, dict):
        state_dict = package.get("state_dict", package)
        saved_args = package.get("args", {})
    else:
        state_dict = package
        saved_args = {}

    state_dict = _normalize_state_dict(state_dict)

    if saved_args:
        print("Checkpoint hyperparameters:")
        for key, value in sorted(saved_args.items()):
            print(f"  {key}: {value}")

    head = CLAPFusionHead(
        n_classes=len(MELD_LABELS),
        d_shared=saved_args.get("fusion_shared_dim", 1024),
        dropout=saved_args.get("fusion_dropout", 0.3),
        hidden_mult=saved_args.get("fusion_hidden_mult", 2.0),
        classifier_depth=saved_args.get("fusion_classifier_depth", 3),
    )
    try:
        head.load_state_dict(state_dict)
    except RuntimeError as exc:  # pragma: no cover - provides clearer error context for users
        raise RuntimeError(f"Failed to load fusion head checkpoint '{checkpoint_file}': {exc}") from exc
    head.to(device)
    head.eval()
    return head, saved_args


def evaluate(checkpoint_file: Path, batch_size: int = 8, num_workers: int = 0) -> tuple[dict, np.ndarray, np.ndarray, str]:
    head, _ = load_fusion_head(checkpoint_file, device)
    prefetch_factor = 2 if num_workers > 0 else None
    test_loader = get_dataloader(
        ".",
        split="test",
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_paired,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=prefetch_factor,
    )
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    totals = {"loss": 0.0, "ctr": 0.0, "ce": 0.0, "correct": 0, "samples": 0, "steps": 0}
    amp_enabled = torch.cuda.is_available()
    with torch.no_grad():
        progress = tqdm(test_loader, desc="Evaluating", unit="batch", leave=False)
        for a_inputs, t_inputs, labels in progress:
            z_a = audio_embed(a_inputs)
            z_t = text_embed(t_inputs)
            with autocast(device_type=device.type, enabled=amp_enabled):
                _, _, sims, logits = head(z_a, z_t)
            targets = labels.to(device, non_blocking=True)
            ctr = clap_infonce(sims)
            ce = torch.nn.functional.cross_entropy(logits, targets)
            loss = ctr + ce
            preds = logits.argmax(dim=1)
            correct = (preds == targets).sum().item()
            batch_size_curr = targets.size(0)

            totals["loss"] += loss.item()
            totals["ctr"] += ctr.item()
            totals["ce"] += ce.item()
            totals["correct"] += correct
            totals["samples"] += batch_size_curr
            totals["steps"] += 1

            running_acc = totals["correct"] / max(totals["samples"], 1)
            running_loss = totals["loss"] / max(totals["steps"], 1)
            progress.set_postfix({"loss": f"{running_loss:.3f}", "acc": f"{running_acc:.2%}"})

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        progress.close()
    if not all_labels:
        raise RuntimeError("Test loader yielded no samples; verify the test split is available.")
    steps = totals["steps"] or 1
    samples = totals["samples"] or 1
    metrics = {
        "loss": totals["loss"] / steps,
        "ctr": totals["ctr"] / steps,
        "ce": totals["ce"] / steps,
        "acc": totals["correct"] / samples,
        "steps": totals["steps"],
        "samples": totals["samples"],
    }
    stacked_preds = np.concatenate(all_preds)
    stacked_labels = np.concatenate(all_labels)
    ordered_labels = list(range(len(MELD_LABELS)))
    cm = confusion_matrix(stacked_labels, stacked_preds, labels=ordered_labels)
    report = cast(
        str,
        classification_report(
            stacked_labels,
            stacked_preds,
            labels=ordered_labels,
            target_names=MELD_LABELS,
            digits=4,
            zero_division=0,
        ),
    )
    return metrics, cm, stacked_labels, report


if __name__ == "__main__":
    metrics, cm, labels, report = evaluate(checkpoint_path)

    print(
        "\n"
        "Aggregate metrics | loss={loss:.4f} ctr={ctr:.4f} ce={ce:.4f} acc={acc:.2%} "
        "| batches={steps} samples={samples}".format(**metrics)
    )
    print("\nPer-label true-positive breakdown:")
    print(f"{'label':>10}  {'tp':>4}  {'fp':>4}  {'fn':>4}  {'precision':>9}  {'recall':>7}")
    for idx, label_name in enumerate(MELD_LABELS):
        tp = int(cm[idx, idx])
        fn = int(cm[idx, :].sum() - tp)
        fp = int(cm[:, idx].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        print(f"{label_name:>10}  {tp:4d}  {fp:4d}  {fn:4d}  {precision:9.4f}  {recall:7.4f}")
    print("\nClassification report:")
    print(report)