import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn.utils import clip_grad_norm_

from data.dataloader import AudioDataset, get_dataloader, MELD_LABELS, lab2id
from model import (
    CLAPFusionHead,
    audio_embed,
    clap_infonce,
    collate_paired,
    set_model_device,
    text_embed,
)

# CUDA device setup with error handling
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("Install PyTorch with CUDA: uv pip install torch --index-url https://download.pytorch.org/whl/cu124")
    sys.exit(1)

device = torch.device("cuda")
n_gpus = torch.cuda.device_count()
print(f"Found {n_gpus} GPUs:")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Test GPU before starting
print("\nTesting GPU access...")
try:
    test_tensor = torch.randn(10, 10).to(device)
    _ = test_tensor @ test_tensor.t()
    print("✓ GPU test passed\n")
except Exception as e:
    print(f"✗ GPU test failed: {e}")
    sys.exit(1)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward_batch(head, batch, class_weights=None):
    a_inputs, t_inputs, labels = batch
    z_a = audio_embed(a_inputs)
    z_t = text_embed(t_inputs)
    _, _, sims, logits = head(z_a, z_t)
    targets = labels.to(device)
    ctr = clap_infonce(sims)
    if class_weights is not None:
        ce = F.cross_entropy(logits, targets, weight=class_weights)
    else:
        ce = F.cross_entropy(logits, targets)
    loss = ctr + ce
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return loss, ctr, ce, correct, targets.size(0), preds.detach(), targets.detach()


def init_stats():
    return {"loss": 0.0, "ctr": 0.0, "ce": 0.0, "correct": 0, "samples": 0, "steps": 0}


def update_stats(stats, loss, ctr, ce, correct, batch_size):
    stats["loss"] += loss
    stats["ctr"] += ctr
    stats["ce"] += ce
    stats["correct"] += correct
    stats["samples"] += batch_size
    stats["steps"] += 1


def summarize_stats(stats):
    if stats["steps"] == 0:
        return {"loss": 0.0, "ctr": 0.0, "ce": 0.0, "acc": 0.0, "samples": 0, "steps": 0}

    steps = stats["steps"]
    samples = max(stats["samples"], 1)
    return {
        "loss": stats["loss"] / steps,
        "ctr": stats["ctr"] / steps,
        "ce": stats["ce"] / steps,
        "acc": stats["correct"] / samples,
        "samples": stats["samples"],
        "steps": steps,
    }


def evaluate(
    head,
    loader,
    max_batches=None,
    amp_enabled=True,
    class_weights=None,
    *,
    collect_confusion: bool = False,
    num_classes: int | None = None,
    label_names: list[str] | None = None,
):
    was_training = head.training
    head.eval()
    totals = init_stats()
    confusion = None
    if collect_confusion:
        if num_classes is None:
            raise ValueError("num_classes must be provided when collect_confusion=True")
        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            with autocast(device_type=device.type, enabled=amp_enabled):
                loss, ctr, ce, correct, batch_size, preds, targets = forward_batch(
                    head,
                    batch,
                    class_weights,
                )
            update_stats(
                totals,
                loss.item(),
                ctr.item(),
                ce.item(),
                correct,
                batch_size,
            )

            if collect_confusion and confusion is not None:
                preds_cpu = preds.long().cpu()
                targets_cpu = targets.long().cpu()
                for t, p in zip(targets_cpu, preds_cpu):
                    confusion[t, p] += 1

            if max_batches and batch_idx >= max_batches:
                break

    head.train(was_training)
    summary = summarize_stats(totals)

    if collect_confusion and confusion is not None:
        diag = torch.diag(confusion)
        per_class = []
        preds_per_class = confusion.sum(dim=0)
        targets_per_class = confusion.sum(dim=1)
        for idx in range(confusion.size(0)):
            tp = diag[idx].item()
            fp = preds_per_class[idx].item() - tp
            fn = targets_per_class[idx].item() - tp
            precision = tp / preds_per_class[idx].item() if preds_per_class[idx].item() > 0 else 0.0
            recall = tp / targets_per_class[idx].item() if targets_per_class[idx].item() > 0 else 0.0
            label = label_names[idx] if label_names and idx < len(label_names) else str(idx)
            per_class.append({
                "label": label,
                "tp": tp,
                "fp": max(fp, 0),
                "fn": max(fn, 0),
                "precision": precision,
                "recall": recall,
            })
        summary["per_class"] = per_class
        summary["confusion"] = confusion

    return summary


class HyperParamController:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        init_lr: float,
        init_weight_decay: float,
        min_lr: float,
        max_weight_decay: float,
        lr_decay: float,
        wd_growth: float,
        patience: int,
        tolerance: float,
    ) -> None:
        self.optimizer = optimizer
        self.lr = init_lr
        self.weight_decay = init_weight_decay
        self.min_lr = max(min_lr, 1e-8)
        self.max_weight_decay = max(max_weight_decay, 0.0)
        self.lr_decay = min(max(lr_decay, 1e-4), 1.0)
        self.wd_growth = max(wd_growth, 1.0)
        self.patience = max(patience, 1)
        self.tolerance = max(tolerance, 0.0)
        self.best_eval = float("inf")
        self.bad_count = 0
        self.apply()

    def apply(self) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
            group["weight_decay"] = self.weight_decay

    def state(self) -> dict:
        return {"lr": self.lr, "weight_decay": self.weight_decay}

    def step(self, eval_loss: float) -> bool:
        if eval_loss + self.tolerance < self.best_eval:
            self.best_eval = eval_loss
            self.bad_count = 0
            return False

        self.bad_count += 1
        if self.bad_count < self.patience:
            return False

        self.bad_count = 0
        updated = False

        new_lr = max(self.lr * self.lr_decay, self.min_lr)
        if new_lr < self.lr - 1e-12:
            self.lr = new_lr
            updated = True

        new_wd = min(self.weight_decay * self.wd_growth, self.max_weight_decay)
        if new_wd > self.weight_decay + 1e-12:
            self.weight_decay = new_wd
            updated = True

        if updated:
            self.apply()
            print(
                f"[HyperParamController] Adjusted lr={self.lr:.2e}, weight_decay={self.weight_decay:.2e} "
                f"after eval_loss={eval_loss:.4f}"
            )
        else:
            print(
                f"[HyperParamController] Eval plateau detected (loss={eval_loss:.4f}); bounds prevent updates."
            )
        return updated


class MetricsLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        fieldnames = [
            "epoch",
            "global_step",
            "phase",
            "loss",
            "ctr",
            "ce",
            "acc",
            "lr",
            "wd",
        ]
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()

    def log(self, *, epoch: int, global_step: int, phase: str, stats: dict, hp_state: dict) -> None:
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "phase": phase,
            "loss": stats.get("loss", float("nan")),
            "ctr": stats.get("ctr", float("nan")),
            "ce": stats.get("ce", float("nan")),
            "acc": stats.get("acc", float("nan")),
            "lr": hp_state.get("lr", float("nan")),
            "wd": hp_state.get("weight_decay", float("nan")),
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CLAP fusion head with adaptive hyperparameters")
    parser.add_argument("--root", type=str, default=".", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-GPU batch size")
    parser.add_argument("--num-workers", type=int, default=16, help="DataLoader workers")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Initial weight decay")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Lower bound on learning rate")
    parser.add_argument("--max-weight-decay", type=float, default=5e-2, help="Upper bound on weight decay")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="Factor to decay LR when eval stalls")
    parser.add_argument("--wd-growth-factor", type=float, default=1.1, help="Factor to grow weight decay when eval stalls")
    parser.add_argument("--hp-patience", type=int, default=2, help="Evaluations to wait before updating hyperparameters")
    parser.add_argument("--hp-tolerance", type=float, default=1e-3, help="Required eval loss improvement")
    parser.add_argument("--eval-frequency", choices=["epoch", "step"], default="epoch", help="How often to run eval")
    parser.add_argument("--eval-every", type=int, default=1, help="Eval every N epochs or steps (depending on frequency)")
    parser.add_argument("--eval-batches", type=int, default=0, help="Number of eval mini-batches per check (0 = full eval)")
    parser.add_argument("--eval-shuffle", action="store_true", help="Shuffle eval loader for diverse mini-batches")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--use-multi-gpu", action="store_true", help="Enable DataParallel if multiple GPUs available")
    parser.add_argument("--save-path", type=str, default=None, help="Destination file for the trained fusion head")
    parser.add_argument("--fusion-shared-dim", type=int, default=1024, help="Width of the shared embedding space in the fusion head")
    parser.add_argument("--fusion-dropout", type=float, default=0.3, help="Dropout probability applied inside the fusion head")
    parser.add_argument("--fusion-hidden-mult", type=float, default=2.0, help="Multiplier applied to shared dim for intermediate widths")
    parser.add_argument("--fusion-classifier-depth", type=int, default=3, help="Number of hidden layers inside the classifier head")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Number of micro-batches to accumulate before optimizer step")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Number of batches to prefetch per worker")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Force DataLoader pin_memory=True")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Force DataLoader pin_memory=False")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable automatic mixed precision training")
    parser.add_argument("--metrics-log", type=str, default="logs/train_metrics.csv", help="Path to CSV file for metric logging")
    parser.add_argument(
        "--class-weight-gamma",
        type=float,
        default=1.0,
        help="Exponent applied to inverse-frequency class weights (1.0 = full weighting, 0.0 = uniform).",
    )
    parser.add_argument(
        "--train-sampler",
        choices=["shuffle", "weighted", "interleaved"],
        default="shuffle",
        help="Training batch strategy: standard shuffle, fully weighted, or interleaved mix.",
    )
    parser.add_argument(
        "--interleave-period",
        type=int,
        default=4,
        help="When using interleaved sampling, insert one balanced batch after this many shuffled batches.",
    )
    parser.add_argument("--log-eval-per-class", action="store_true", help="Compute and print per-class metrics during eval runs")
    parser.add_argument("--train-metrics-every", type=int, default=0, help="Run an additional eval on the train split every N epochs (0 disables)")
    parser.add_argument("--train-metrics-batches", type=int, default=0, help="Limit batches when running train metrics (0 = full split)")
    parser.add_argument("--early-stop-patience", type=int, default=0, help="Stop early after this many consecutive degraded evals (0 disables)")
    parser.add_argument("--early-stop-threshold", type=float, default=0.2, help="Fractional increase over best eval loss that counts as degradation")
    parser.add_argument("--max-grad-norm", type=float, default=0.0, help="Clip gradients to this norm before optimizer step (0 disables)")
    parser.add_argument("--load-checkpoint", type=str, default=None, help="Optional path to an existing fusion-head checkpoint to resume from")
    parser.set_defaults(pin_memory=None, amp=True)
    return parser.parse_args()


def _normalize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "head.", "fusion_head.")
    for prefix in prefixes:
        if state_dict and all(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def compute_class_weights(dataset, num_classes: int, gamma: float = 1.0) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    if hasattr(dataset, "meta"):
        for item in dataset.meta:
            label = item.get("Emotion", "").lower()
            idx = lab2id.get(label)
            if idx is not None:
                counts[idx] += 1.0
    else:
        for _, _, _, idx in dataset:
            counts[int(idx)] += 1.0

    counts.clamp_min_(1.0)
    weights = counts.sum() / (counts * num_classes)
    weights /= weights.mean()
    if gamma != 1.0:
        weights = weights.pow(gamma)
        weights /= weights.mean()
    return weights


class InterleavedLoader:
    def __init__(self, primary_loader, secondary_loader, period: int) -> None:
        self.primary_loader = primary_loader
        self.secondary_loader = secondary_loader
        self.period = max(period, 1)

    def __len__(self) -> int:
        base = len(self.primary_loader)
        extras = base // self.period
        return base + extras

    def __iter__(self):
        primary_iter = iter(self.primary_loader)
        secondary_iter = iter(self.secondary_loader)
        batches_since_secondary = 0

        for batch in primary_iter:
            yield batch
            batches_since_secondary += 1
            if batches_since_secondary >= self.period:
                batches_since_secondary = 0
                try:
                    yield next(secondary_iter)
                except StopIteration:
                    secondary_iter = iter(self.secondary_loader)
                    yield next(secondary_iter)


def _build_dataloader(
    dataset: AudioDataset,
    *,
    batch_size: int,
    num_workers: int,
    collate_fn,
    pin_memory: bool,
    prefetch_factor: int | None,
    shuffle: bool = False,
    sampler = None,
):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = shuffle

    if num_workers > 0:
        kwargs["persistent_workers"] = True
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**kwargs)


def main() -> None:
    args = parse_args()

    set_seed(args.seed)

    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    metrics_logger = MetricsLogger(Path(args.metrics_log))
    print(f"Logging metrics to {metrics_logger.path}")

    head = CLAPFusionHead(
        n_classes=7,
        d_shared=args.fusion_shared_dim,
        dropout=args.fusion_dropout,
        hidden_mult=args.fusion_hidden_mult,
        classifier_depth=args.fusion_classifier_depth,
    ).to(device)
    set_model_device(device)

    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        package = torch.load(checkpoint_path, map_location=device)
        state_dict = package.get("state_dict") if isinstance(package, dict) else None
        if state_dict is None:
            state_dict = package if isinstance(package, dict) else package
        state_dict = _normalize_state_dict(state_dict)
        head.load_state_dict(state_dict, strict=True)
        print(f"Loaded fusion head weights from {checkpoint_path}")

    use_multi_gpu = args.use_multi_gpu and n_gpus > 1
    if use_multi_gpu:
        print(f"Using DataParallel across {n_gpus} GPUs")
        head = torch.nn.DataParallel(head)
    else:
        print("Using single GPU training")

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    hp_controller = HyperParamController(
        optimizer=optimizer,
        init_lr=args.lr,
        init_weight_decay=args.weight_decay,
        min_lr=args.min_lr,
        max_weight_decay=args.max_weight_decay,
        lr_decay=args.lr_decay_factor,
        wd_growth=args.wd_growth_factor,
        patience=args.hp_patience,
        tolerance=args.hp_tolerance,
    )

    pin_memory = args.pin_memory if args.pin_memory is not None else torch.cuda.is_available()

    train_dataset = AudioDataset(args.root, split="train")

    sampler_class_weights = compute_class_weights(
        train_dataset,
        len(MELD_LABELS),
        gamma=max(args.class_weight_gamma, 0.0),
    )
    if getattr(train_dataset, "meta", None) is not None:
        sample_weights = [
            float(sampler_class_weights[lab2id[item["Emotion"].lower()]])
            for item in train_dataset.meta
        ]
    else:
        sample_weights = [
            float(sampler_class_weights[int(train_dataset[i][-1])])
            for i in range(len(train_dataset))
        ]

    weighted_sampler = None
    if sample_weights:
        weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    if args.train_sampler == "weighted":
        train_loader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_paired,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            sampler=weighted_sampler,
        )
    elif args.train_sampler == "interleaved":
        base_loader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_paired,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
        )
        balanced_loader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_paired,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            sampler=weighted_sampler,
        )
        train_loader = InterleavedLoader(base_loader, balanced_loader, period=max(args.interleave_period, 1))
    else:
        train_loader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_paired,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=True,
        )

    train_eval_loader = None
    train_metrics_limit = _resolve_eval_batches(args.train_metrics_batches)
    if args.train_metrics_every > 0:
        train_eval_loader = _build_dataloader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=collate_paired,
            pin_memory=pin_memory,
            prefetch_factor=args.prefetch_factor,
            shuffle=False,
        )

    class_weights = compute_class_weights(train_dataset, len(MELD_LABELS), gamma=max(args.class_weight_gamma, 0.0)).to(device)
    print("Class weights (after gamma scaling):", class_weights.tolist())

    eval_loader = get_dataloader(
        args.root,
        split="eval",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_paired,
        shuffle=args.eval_shuffle,
        pin_memory=pin_memory,
        prefetch_factor=args.prefetch_factor,
    )
    global_step = 0
    last_eval_stats = None
    eval_batch_limit = _resolve_eval_batches(args.eval_batches)
    scaler = GradScaler(device.type, enabled=args.amp)
    steps_per_epoch = len(train_loader)

    best_eval_loss = float("inf")
    early_stop_count = 0
    early_stop_triggered = False
    early_stop_enabled = args.early_stop_patience > 0

    def _process_metrics(stats: dict, prefix: str, apply_early_stop: bool) -> None:
        nonlocal best_eval_loss, early_stop_count, early_stop_triggered
        if args.log_eval_per_class and "per_class" in stats:
            print(f"{prefix} per-class metrics:")
            for entry in stats["per_class"]:
                print(
                    "  {label:<8} tp={tp:>4} fp={fp:>4} fn={fn:>4} precision={precision:.3f} recall={recall:.3f}".format(
                        label=entry["label"],
                        tp=entry["tp"],
                        fp=entry["fp"],
                        fn=entry["fn"],
                        precision=entry["precision"],
                        recall=entry["recall"],
                    )
                )

        if apply_early_stop and early_stop_enabled:
            eval_loss = stats.get("loss", float("inf"))
            if not math.isfinite(eval_loss):
                return
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                early_stop_count = 0
            elif best_eval_loss < float("inf") and eval_loss > best_eval_loss * (1.0 + args.early_stop_threshold):
                early_stop_count += 1
                print(
                    f"[EarlyStop] Eval loss {eval_loss:.4f} exceeded baseline {best_eval_loss:.4f} by >= {args.early_stop_threshold * 100:.1f}% "
                    f"({early_stop_count}/{args.early_stop_patience})"
                )
                if early_stop_count >= args.early_stop_patience:
                    print("[EarlyStop] Patience exhausted; stopping training.")
                    early_stop_triggered = True
            else:
                early_stop_count = 0

    final_epoch = 0
    for epoch in range(1, args.epochs + 1):
        final_epoch = epoch
        head.train()
        train_totals = init_stats()
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)
        for batch_idx, batch in enumerate(progress_bar, start=1):
            with autocast(device_type=device.type, enabled=args.amp):
                loss, ctr, ce, correct, batch_size, _, _ = forward_batch(head, batch, class_weights)

            loss_for_backward = loss / args.grad_accum_steps
            scaler.scale(loss_for_backward).backward()

            should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == steps_per_epoch)
            if should_step:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(head.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            update_stats(
                train_totals,
                loss.item(),
                ctr.item(),
                ce.item(),
                correct,
                batch_size,
            )

            averages = summarize_stats(train_totals)

            postfix = {
                "loss": f"{averages['loss']:.4f}",
                "ctr": f"{averages['ctr']:.4f}",
                "ce": f"{averages['ce']:.4f}",
                "acc": f"{averages['acc']:.4f}",
                "lr": f"{hp_controller.lr:.2e}",
                "wd": f"{hp_controller.weight_decay:.2e}",
            }

            progress_bar.set_postfix(postfix)

            if args.eval_frequency == "step" and global_step % max(args.eval_every, 1) == 0:
                eval_stats = evaluate(
                    head,
                    eval_loader,
                    eval_batch_limit,
                    amp_enabled=args.amp,
                    class_weights=class_weights,
                    collect_confusion=args.log_eval_per_class,
                    num_classes=len(MELD_LABELS),
                    label_names=MELD_LABELS,
                )
                last_eval_stats = eval_stats
                hp_controller.step(eval_stats["loss"])
                metrics_logger.log(
                    epoch=epoch,
                    global_step=global_step,
                    phase="eval",
                    stats=eval_stats,
                    hp_state=hp_controller.state(),
                )
                _process_metrics(eval_stats, f"Eval@step {global_step}", apply_early_stop=True)
                tqdm.write(
                    "Eval step {} | loss={:.4f} ctr={:.4f} ce={:.4f} acc={:.2f}% | lr={} wd={}".format(
                        global_step,
                        eval_stats["loss"],
                        eval_stats["ctr"],
                        eval_stats["ce"],
                        eval_stats["acc"] * 100.0,
                        f"{hp_controller.lr:.2e}",
                        f"{hp_controller.weight_decay:.2e}",
                    )
                )
                if early_stop_triggered:
                    break

        train_summary = summarize_stats(train_totals)

        if not early_stop_triggered and args.eval_frequency == "epoch" and epoch % max(args.eval_every, 1) == 0:
            eval_stats = evaluate(
                head,
                eval_loader,
                eval_batch_limit,
                amp_enabled=args.amp,
                class_weights=class_weights,
                collect_confusion=args.log_eval_per_class,
                num_classes=len(MELD_LABELS),
                label_names=MELD_LABELS,
            )
            last_eval_stats = eval_stats
            hp_controller.step(eval_stats["loss"])
            metrics_logger.log(
                epoch=epoch,
                global_step=global_step,
                phase="eval",
                stats=eval_stats,
                hp_state=hp_controller.state(),
            )
            _process_metrics(eval_stats, f"Eval@epoch {epoch}", apply_early_stop=True)
        
        eval_display = last_eval_stats or {"loss": float("nan"), "ctr": float("nan"), "ce": float("nan"), "acc": float("nan")}

        print(
            "Epoch {} done | train_loss={:.4f} ctr={:.4f} ce={:.4f} acc={:.2f}% | eval_loss={:.4f} ctr={:.4f} ce={:.4f} acc={:.2f}% | lr={} wd={}".format(
                epoch,
                train_summary["loss"],
                train_summary["ctr"],
                train_summary["ce"],
                train_summary["acc"] * 100.0,
                eval_display["loss"],
                eval_display["ctr"],
                eval_display["ce"],
                eval_display["acc"] * 100.0,
                f"{hp_controller.lr:.2e}",
                f"{hp_controller.weight_decay:.2e}",
            )
        )

        metrics_logger.log(
            epoch=epoch,
            global_step=global_step,
            phase="train",
            stats=train_summary,
            hp_state=hp_controller.state(),
        )

        if not early_stop_triggered and args.train_metrics_every > 0 and train_eval_loader is not None and epoch % args.train_metrics_every == 0:
            train_metrics_stats = evaluate(
                head,
                train_eval_loader,
                train_metrics_limit,
                amp_enabled=args.amp,
                class_weights=class_weights,
                collect_confusion=args.log_eval_per_class,
                num_classes=len(MELD_LABELS),
                label_names=MELD_LABELS,
            )
            metrics_logger.log(
                epoch=epoch,
                global_step=global_step,
                phase="train_eval",
                stats=train_metrics_stats,
                hp_state=hp_controller.state(),
            )
            _process_metrics(train_metrics_stats, f"Train@epoch {epoch}", apply_early_stop=False)

        if early_stop_triggered:
            break

    if early_stop_triggered:
        print(f"Stopped early after epoch {final_epoch} due to sustained eval degradation.")

    save_path = _resolve_save_path(args)
    target = head.module if isinstance(head, torch.nn.DataParallel) else head
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": target.state_dict(), "args": dict(vars(args))}, save_path)
    print(f"Saved fusion head checkpoint to {save_path}")

    metrics_logger.close()


def _resolve_eval_batches(eval_batches: int):
    return None if eval_batches <= 0 else eval_batches


def _resolve_save_path(args: argparse.Namespace) -> Path:
    if args.save_path:
        return Path(args.save_path)
    default_dir = Path(args.root) / "checkpoints"
    return default_dir / "clap_fusion_head.pt"


if __name__ == "__main__":
    main()
