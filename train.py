from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from rlnet.data import FrameSequenceDataset, discover_records, split_records
from rlnet.metrics import classification_metrics
from rlnet.model import RLNet
from rlnet.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a paper-inspired RLNet baseline on UADFV frame sequences.")
    parser.add_argument("--dataset-root", type=Path, default=Path("UADFV"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rlnet"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--bidirectional", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def build_loader(dataset: FrameSequenceDataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def run_epoch(
    model: RLNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    epoch: int | None = None,
    phase: str = "train",
) -> tuple[float, dict[str, float]]:
    is_training = optimizer is not None
    model.train(is_training)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    total_loss = 0.0

    progress = tqdm(
        loader,
        desc=f"{phase} epoch {epoch}" if epoch is not None else phase,
        leave=False,
    )

    for frames, labels, _video_ids in progress:
        frames = frames.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        autocast_device = device.type if device.type in {"cuda", "cpu"} else "cpu"
        with autocast(device_type=autocast_device, enabled=device.type == "cuda"):
            logits = model(frames)
            loss = criterion(logits, labels)

        if is_training and optimizer is not None:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * frames.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
        seen_samples = sum(batch.size(0) for batch in all_labels)
        progress.set_postfix(loss=f"{total_loss / max(seen_samples, 1):.4f}")

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    metrics = classification_metrics(logits, labels)
    mean_loss = total_loss / max(len(loader.dataset), 1)
    return mean_loss, metrics


def build_optimizer(model: RLNet, args: argparse.Namespace) -> torch.optim.Optimizer:
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.") and parameter.requires_grad
    ]
    parameter_groups: list[dict] = []
    if backbone_parameters:
        parameter_groups.append({"params": backbone_parameters, "lr": args.backbone_learning_rate})
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.learning_rate})
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = discover_records(args.dataset_root)
    train_records, val_records = split_records(records, test_size=args.val_size, seed=args.seed)

    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]
    if args.max_val_samples is not None:
        val_records = val_records[: args.max_val_samples]

    train_dataset = FrameSequenceDataset(
        train_records,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        train=True,
    )
    val_dataset = FrameSequenceDataset(
        val_records,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        train=False,
    )

    train_loader = build_loader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    val_loader = build_loader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    device = torch.device(args.device)
    model = RLNet(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
        bidirectional=args.bidirectional,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, args)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=args.scheduler_patience,
        factor=args.scheduler_factor,
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float | int]] = []
    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / "best.pt"
    serialized_config = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    print(
        "training_config="
        f"pretrained_backbone={args.pretrained_backbone} "
        f"freeze_backbone={args.freeze_backbone} "
        f"unfreeze_epoch={args.unfreeze_epoch} "
        f"learning_rate={args.learning_rate} "
        f"backbone_learning_rate={args.backbone_learning_rate}"
    )

    for epoch in range(1, args.epochs + 1):
        if args.freeze_backbone and args.unfreeze_epoch > 0 and epoch == args.unfreeze_epoch:
            model.set_backbone_trainable(True)
            optimizer = build_optimizer(model, args)
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=args.scheduler_patience,
                factor=args.scheduler_factor,
            )
            print(f"unfroze_backbone_at_epoch={epoch}")

        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            phase="train",
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model,
                val_loader,
                criterion,
                device,
                epoch=epoch,
                phase="val",
            )

        epoch_result = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "val_f1": val_metrics["f1"],
            "head_lr": optimizer.param_groups[-1]["lr"],
        }
        history.append(epoch_result)
        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"train_f1={train_metrics['f1']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"head_lr={optimizer.param_groups[-1]['lr']:.2e}"
        )
        scheduler.step(val_loss)

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": serialized_config,
                    "val_metrics": val_metrics,
                    "train_metrics": train_metrics,
                },
                best_checkpoint_path,
            )

    save_json(output_dir / "history.json", {"epochs": history})
    save_json(
        output_dir / "split.json",
        {
            "train_ids": [record.video_id for record in train_records],
            "val_ids": [record.video_id for record in val_records],
        },
    )
    print(f"saved_best_checkpoint={best_checkpoint_path}")


if __name__ == "__main__":
    main()
