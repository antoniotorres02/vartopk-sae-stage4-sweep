"""Fast ConvNeXt Stage4 comparison: TopK no-norm versus original Variable TopK."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from tfm_sae_evals.config import ExternalPaths
from tfm_sae_evals.experiments.convnext_stage4_features import (
    STAGE_DIM,
    STAGE_FEATURE_IDX,
    STAGE_HW,
    STAGE_NAME,
    StageData,
    classify_from_stage_map,
    get_stage_data,
    load_convnext_checkpoint,
    maps_to_vectors,
    vectors_to_maps,
)
from tfm_sae_evals.models import OriginalVariableTopKSparseAutoencoder, TopKSparseAutoencoder
from tfm_sae_evals.models.sae import unwrap_model
from tfm_sae_evals.models.variable_topk_loss import compute_discrete_kl, compute_prefix_expected_mse


SEED = 42
METHODS = ("topk_nonorm", "variable_topk_original")
DEFAULT_RESULTS_DIR = Path("results/convnext_stage4_only_topk_versus_topk")
DEFAULT_OUTPUT_DIR = Path("outputs/convnext_stage4_only_topk_versus_topk")
RESULTS_CSV = "stage4_only_topk_versus_topk_results.csv"
REPORT_MD = "stage4_only_topk_versus_topk_report.md"
CSV_FIELDNAMES = [
    "stage",
    "method",
    "status",
    "error",
    "d",
    "hidden_dim",
    "hidden_multiplier",
    "k",
    "k_max",
    "target_l0",
    "lambda_prior",
    "beta",
    "train_images",
    "val_images",
    "test_images",
    "train_vectors",
    "best_epoch",
    "epochs_trained",
    "stopped_reason",
    "runtime_seconds",
    "peak_memory_mb",
    "test_nmse",
    "test_mse",
    "l0",
    "dead_pct",
    "top1_agreement",
    "original_top1_accuracy",
    "reconstructed_top1_accuracy",
    "logit_kl",
    "suffix_max_abs_diff",
    "checkpoint_path",
]


class StageVectorDataset(torch.utils.data.Dataset):
    def __init__(self, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        self.vectors = vectors
        self.mean = mean.squeeze(0)
        self.std = std.squeeze(0)

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def __getitem__(self, idx: int):
        vector = self.vectors[idx].float()
        return (vector - self.mean) / self.std, torch.tensor(0, dtype=torch.long)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_vector_loader(vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, shuffle: bool):
    dataset = StageVectorDataset(vectors, mean=mean, std=std)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def forward_sae(model, batch: torch.Tensor):
    out = model(batch, return_aux=True)
    missing = {"encoded", "reconstruction", "k_eval"}.difference(out)
    if missing:
        raise ValueError(f"SAE output missing keys: {sorted(missing)}")
    return out


def evaluate_reconstruction_mse(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    mse_sum = 0.0
    elements = 0
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            out = forward_sae(model, xb)
            mse_sum += F.mse_loss(out["reconstruction"], xb, reduction="sum").item()
            elements += xb.numel()
    return {"mse": mse_sum / max(elements, 1)}


def loss_for_batch(model, xb: torch.Tensor, method: str, args: argparse.Namespace):
    out = forward_sae(model, xb)
    hard = F.mse_loss(out["reconstruction"], xb)
    if method == "topk_nonorm":
        return hard, out

    expected_mse, _ = compute_prefix_expected_mse(out["prefix_reconstruction"], xb, out["k_probs"])
    expected = expected_mse.mean()
    kl = compute_discrete_kl(out["k_probs"], out["k_log_probs"], args.vtk_lambda_prior).mean()
    budget = ((out["expected_k"].mean() - args.vtk_target_l0) / max(args.vtk_target_l0, 1.0)).pow(2)
    loss = (
        args.vtk_expected_weight * expected
        + args.vtk_hard_weight * hard
        + args.vtk_beta * kl
        + args.vtk_budget_weight * budget
    )
    return loss, out


def train_with_early_stopping(
    model,
    method: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    deadline: float,
):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_scheduler_factor,
        patience=args.lr_scheduler_patience,
        min_lr=args.lr_scheduler_min,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(unwrap_model(model).state_dict())
    best_val = float("inf")
    best_ema = float("inf")
    best_epoch = 0
    no_improve = 0
    stop_reason = "max_epochs"
    epoch_idx = -1

    for epoch_idx in range(args.epochs):
        model.train()
        running_loss = 0.0
        n_samples = 0
        timed_out = False
        for xb, _ in train_loader:
            if time.perf_counter() >= deadline:
                timed_out = True
                break
            xb = xb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss, _out = loss_for_batch(model, xb, method, args)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * xb.shape[0]
            n_samples += xb.shape[0]

        if n_samples == 0:
            stop_reason = "time_budget_before_epoch"
            break

        val = evaluate_reconstruction_mse(model, val_loader, device)["mse"]
        ema = val if epoch_idx == 0 else args.ema_alpha * val + (1.0 - args.ema_alpha) * ema
        scheduler.step(val)
        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(unwrap_model(model).state_dict())
            best_epoch = epoch_idx + 1
        if ema < best_ema - args.min_delta:
            best_ema = ema
            no_improve = 0
        else:
            no_improve += 1

        log_this = epoch_idx == 0 or (epoch_idx + 1) % args.log_every == 0 or no_improve >= args.patience or timed_out
        if log_this:
            print(
                f"    ep {epoch_idx + 1:03d}/{args.epochs} "
                f"loss={running_loss / max(n_samples, 1):.6f} "
                f"val_mse={val:.6f} ema={ema:.6f} best_ep={best_epoch} "
                f"pat={no_improve}/{args.patience}",
                flush=True,
            )
        if timed_out:
            stop_reason = f"time_budget_{args.max_total_train_seconds}s"
            break
        if no_improve >= args.patience:
            stop_reason = f"early_stop_patience_{args.patience}"
            break

    unwrap_model(model).load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "epochs_trained": max(epoch_idx + 1, 0),
        "stopped_reason": stop_reason,
        "best_val_mse": best_val,
    }


@torch.inference_mode()
def compute_dead_pct(model, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, device) -> float:
    hidden_dim = int(getattr(unwrap_model(model), "hidden_dim"))
    fire = torch.zeros(hidden_dim, dtype=torch.bool)
    model.eval()
    for start in range(0, vectors.shape[0], batch_size):
        xb = ((vectors[start : start + batch_size].float() - mean) / std).to(device, non_blocking=True)
        out = forward_sae(model, xb)
        fire |= (out["encoded"].detach().cpu() > 0).any(dim=0)
    return 100.0 * (1.0 - fire.float().mean().item())


@torch.inference_mode()
def reconstruct_vectors(model, vectors: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, batch_size: int, device):
    chunks = []
    l0_sum = 0.0
    n = 0
    for start in range(0, vectors.shape[0], batch_size):
        raw = vectors[start : start + batch_size].float()
        xb = ((raw - mean) / std).to(device, non_blocking=True)
        out = forward_sae(model, xb)
        rec = out["reconstruction"].detach().cpu() * std + mean
        chunks.append(rec)
        l0_sum += out["k_eval"].detach().float().sum().item()
        n += xb.shape[0]
    return torch.cat(chunks, dim=0), l0_sum, n


@torch.inference_mode()
def evaluate_downstream(cnn_model, model, maps, labels, mean, std, args: argparse.Namespace, device: torch.device):
    model.eval()
    mse_sum = 0.0
    total_elements = 0
    l0_sum = 0.0
    l0_n = 0
    agree = 0
    correct_orig = 0
    correct_rec = 0
    total_images = 0
    kl_sum = 0.0
    var = maps.float().var(unbiased=False).item()
    for start in range(0, maps.shape[0], args.eval_image_batch_size):
        batch_maps = maps[start : start + args.eval_image_batch_size].float()
        batch_labels = labels[start : start + batch_maps.shape[0]]
        vectors = maps_to_vectors(batch_maps)
        rec_vectors, batch_l0_sum, batch_l0_n = reconstruct_vectors(
            model,
            vectors,
            mean,
            std,
            batch_size=args.eval_vector_batch_size,
            device=device,
        )
        rec_maps = vectors_to_maps(rec_vectors, tuple(batch_maps.shape))
        mse_sum += F.mse_loss(rec_maps, batch_maps, reduction="sum").item()
        total_elements += batch_maps.numel()
        l0_sum += batch_l0_sum
        l0_n += batch_l0_n
        logits_orig = classify_from_stage_map(cnn_model, batch_maps.to(device))
        logits_rec = classify_from_stage_map(cnn_model, rec_maps.to(device))
        labels_device = batch_labels.to(device)
        pred_orig = logits_orig.argmax(dim=1)
        pred_rec = logits_rec.argmax(dim=1)
        correct_orig += (pred_orig == labels_device).sum().item()
        correct_rec += (pred_rec == labels_device).sum().item()
        agree += (pred_orig == pred_rec).sum().item()
        total_images += labels_device.shape[0]
        kl = F.kl_div(F.log_softmax(logits_rec, dim=1), F.softmax(logits_orig, dim=1), reduction="batchmean")
        kl_sum += kl.item() * labels_device.shape[0]
    mse = mse_sum / max(total_elements, 1)
    return {
        "test_mse": mse,
        "test_nmse": mse / max(var, 1e-12),
        "l0": l0_sum / max(l0_n, 1),
        "top1_agreement": agree / max(total_images, 1),
        "original_top1_accuracy": correct_orig / max(total_images, 1),
        "reconstructed_top1_accuracy": correct_rec / max(total_images, 1),
        "logit_kl": kl_sum / max(total_images, 1),
    }


def build_model(method: str, args: argparse.Namespace, device: torch.device):
    hidden_dim = int(args.hidden_multiplier * STAGE_DIM)
    if method == "topk_nonorm":
        return TopKSparseAutoencoder(STAGE_DIM, hidden_dim, args.k, normalize_decoder=False).to(device)
    if method == "variable_topk_original":
        k_max = max(1, min(hidden_dim, int(round(args.vtk_kmax_factor * args.vtk_target_l0))))
        return OriginalVariableTopKSparseAutoencoder(STAGE_DIM, hidden_dim, k_max=k_max).to(device)
    raise ValueError(f"Unsupported method: {method}")


def completed_keys(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return {row["method"] for row in csv.DictReader(handle) if row.get("status") == "ok"}


def append_row(row: dict[str, object], results_csv: Path) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists()
    with results_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDNAMES})


def checkpoint_name(method: str) -> str:
    return f"{STAGE_NAME}_{method}.pt"


def run_one(cnn_model, method: str, stage_data: StageData, args: argparse.Namespace, device: torch.device, deadline: float):
    hidden_dim = int(args.hidden_multiplier * STAGE_DIM)
    train_vectors = maps_to_vectors(stage_data.train_maps)
    val_vectors = maps_to_vectors(stage_data.val_maps)
    mean = train_vectors.mean(dim=0, keepdim=True)
    std = train_vectors.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    batch_size = args.vtk_batch_size if method == "variable_topk_original" else args.batch_size
    train_loader = make_vector_loader(train_vectors, mean, std, batch_size=batch_size, shuffle=True)
    val_loader = make_vector_loader(val_vectors, mean, std, batch_size=batch_size, shuffle=False)
    model = build_model(method, args, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    print(
        f"\n[{STAGE_NAME} / {method}] d={STAGE_DIM} hidden={hidden_dim} "
        f"train_vectors={len(train_vectors)} batch={batch_size}",
        flush=True,
    )
    t0 = time.perf_counter()
    model, train_info = train_with_early_stopping(model, method, train_loader, val_loader, args, device, deadline)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        peak_memory_mb = 0.0
    runtime = time.perf_counter() - t0

    eval_metrics = evaluate_downstream(
        cnn_model,
        model,
        stage_data.test_maps,
        stage_data.test_labels,
        mean,
        std,
        args,
        device,
    )
    dead_pct = compute_dead_pct(model, train_vectors, mean, std, batch_size=args.dead_eval_batch_size, device=device)

    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / checkpoint_name(method)
    k_max = int(getattr(unwrap_model(model), "k_max", 0)) if method == "variable_topk_original" else ""
    torch.save(
        {
            "stage": STAGE_NAME,
            "method": method,
            "config": {
                "d": STAGE_DIM,
                "hidden_dim": hidden_dim,
                "hidden_multiplier": args.hidden_multiplier,
                "k": args.k if method == "topk_nonorm" else None,
                "k_max": k_max if method == "variable_topk_original" else None,
                "target_l0": args.vtk_target_l0 if method == "variable_topk_original" else None,
                "train_images": args.train_images,
                "val_images": args.val_images,
                "test_images": args.test_images,
            },
            "train_info": train_info,
            "eval_metrics": eval_metrics,
            "model_state_dict": unwrap_model(model).state_dict(),
            "mean": mean,
            "std": std,
        },
        ckpt_path,
    )

    row = {
        "stage": STAGE_NAME,
        "method": method,
        "status": "ok",
        "error": "",
        "d": STAGE_DIM,
        "hidden_dim": hidden_dim,
        "hidden_multiplier": args.hidden_multiplier,
        "k": args.k if method == "topk_nonorm" else "",
        "k_max": k_max,
        "target_l0": args.vtk_target_l0 if method == "variable_topk_original" else "",
        "lambda_prior": args.vtk_lambda_prior if method == "variable_topk_original" else "",
        "beta": args.vtk_beta if method == "variable_topk_original" else "",
        "train_images": args.train_images,
        "val_images": args.val_images,
        "test_images": args.test_images,
        "train_vectors": len(train_vectors),
        "best_epoch": train_info["best_epoch"],
        "epochs_trained": train_info["epochs_trained"],
        "stopped_reason": train_info["stopped_reason"],
        "runtime_seconds": round(runtime, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "dead_pct": round(dead_pct, 4),
        "suffix_max_abs_diff": stage_data.suffix_max_abs_diff,
        "checkpoint_path": str(ckpt_path),
        **{key: round(value, 8) for key, value in eval_metrics.items()},
    }
    append_row(row, args.results_dir / RESULTS_CSV)
    write_report(args)
    print(
        f"[{STAGE_NAME} / {method}] done runtime={runtime:.1f}s "
        f"NMSE={eval_metrics['test_nmse']:.4f} L0={eval_metrics['l0']:.2f} "
        f"agree={eval_metrics['top1_agreement']:.3f} KL={eval_metrics['logit_kl']:.4f}",
        flush=True,
    )


def write_report(args: argparse.Namespace) -> None:
    results_csv = args.results_dir / RESULTS_CSV
    report_path = args.results_dir / REPORT_MD
    if not results_csv.exists():
        return
    with results_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: METHODS.index(r["method"]) if r.get("method") in METHODS else 999)
    by_method = {row["method"]: row for row in rows if row.get("status") == "ok"}
    topk_mse = float(by_method["topk_nonorm"]["test_mse"]) if "topk_nonorm" in by_method else None

    lines = [
        "# ConvNeXt Stage4 TopK Versus Original Variable TopK\n\n",
        "Pilot comparison on a deliberately reduced CUB subset so the first pass can stay under the training time budget.\n\n",
        "## Feature Source\n\n",
        "| Field | Value |\n",
        "| --- | --- |\n",
        f"| Hook | `backbone.features[{STAGE_FEATURE_IDX}]` |\n",
        f"| Map shape | `[N, {STAGE_DIM}, {STAGE_HW}, {STAGE_HW}]` |\n",
        f"| SAE vectorization | `[N, {STAGE_DIM}, {STAGE_HW}, {STAGE_HW}] -> [N * {STAGE_HW * STAGE_HW}, {STAGE_DIM}]` |\n",
        f"| Train/val/test images | `{args.train_images}` / `{args.val_images}` / `{args.test_images}` |\n\n",
        "## Architectures\n\n",
        "| Method | Selector | Decoder normalization | Notes |\n",
        "| --- | --- | --- | --- |\n",
        f"| `topk_nonorm` | Fixed `k={args.k}` | Disabled | Matched TopK baseline without decoder column normalization. |\n",
        "| `variable_topk_original` | `Linear(d -> k_max)` over the normalized SAE input | Disabled by architecture | Original Variable TopK baseline in this repo. |\n\n",
        "## Results\n\n",
        "| Method | Runtime s | Best epoch | Epochs | MSE | NMSE | L0 | Dead % | Top-1 agree | Orig acc | Recon acc | Logit KL | MSE vs TopK |\n",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    ]
    for row in rows:
        if row.get("status") != "ok":
            continue
        mse = float(row["test_mse"])
        rel = "" if topk_mse is None else f"{100.0 * (mse - topk_mse) / topk_mse:+.2f}%"
        lines.append(
            f"| `{row['method']}` | {float(row['runtime_seconds']):.1f} | {row['best_epoch']} | {row['epochs_trained']} | "
            f"{mse:.6f} | {float(row['test_nmse']):.6f} | {float(row['l0']):.2f} | "
            f"{float(row['dead_pct']):.2f} | {float(row['top1_agreement']):.4f} | "
            f"{float(row['original_top1_accuracy']):.4f} | {float(row['reconstructed_top1_accuracy']):.4f} | "
            f"{float(row['logit_kl']):.6f} | {rel} |\n"
        )
    lines.extend(
        [
            "\n## Training Configuration\n\n",
            "| Field | Value |\n",
            "| --- | --- |\n",
            f"| Seed | `{SEED}` |\n",
            f"| Optimizer | `Adam(lr={args.lr})` |\n",
            f"| Max epochs | `{args.epochs}` |\n",
            f"| Total training time budget | `{args.max_total_train_seconds}s` |\n",
            f"| Early stopping | `patience={args.patience}`, EMA alpha `{args.ema_alpha}` |\n",
            f"| TopK batch size | `{args.batch_size}` vectors |\n",
            f"| Variable TopK batch size | `{args.vtk_batch_size}` vectors |\n",
            f"| Hidden dimension | `{args.hidden_multiplier}d = {args.hidden_multiplier * STAGE_DIM}` |\n",
            f"| TopK k | `{args.k}` |\n",
            f"| Variable target L0 | `{args.vtk_target_l0}` |\n",
            f"| Variable k_max | `{round(args.vtk_kmax_factor * args.vtk_target_l0)}` |\n",
            f"| lambda_prior | `{args.vtk_lambda_prior}` |\n",
            f"| beta | `{args.vtk_beta}` |\n",
            f"| budget weight | `{args.vtk_budget_weight}` |\n",
        ]
    )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text("".join(lines), encoding="utf-8")


def run_compare(args: argparse.Namespace) -> int:
    args.output_dir = Path(args.output_dir)
    args.results_dir = Path(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    paths = ExternalPaths.from_toml(args.config)
    missing = paths.missing()
    if missing:
        raise FileNotFoundError("Missing external paths: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    results_csv = args.results_dir / RESULTS_CSV
    print(f"Device: {device}", flush=True)
    print(f"Results: {results_csv}", flush=True)
    print(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, sort_keys=True), flush=True)

    cnn_model = load_convnext_checkpoint(paths.convnext_checkpoint, device)
    done = set() if args.fresh else completed_keys(results_csv)
    stage_data = get_stage_data(cnn_model, paths, args, device)
    train_deadline = time.perf_counter() + args.max_total_train_seconds
    for method in args.methods:
        if method in done:
            print(f"[skip] {STAGE_NAME}/{method} already completed", flush=True)
            continue
        if time.perf_counter() >= train_deadline:
            print(f"[skip] {STAGE_NAME}/{method} skipped because the training time budget is exhausted", flush=True)
            continue
        run_one(cnn_model, method, stage_data, args, device, deadline=train_deadline)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_report(args)
    print(f"Done. Wrote {results_csv}", flush=True)
    return 0


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "convnext-stage4-compare-only-topk-versus-topk",
        help="Fast Stage4 comparison: TopK no-norm vs original Variable TopK.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/external_paths.example.toml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--fresh", action="store_true", help="Ignore completed rows and rerun selected methods.")
    parser.add_argument("--fresh-cache", action="store_true")
    parser.add_argument("--train-images", type=int, default=256)
    parser.add_argument("--val-images", type=int, default=64)
    parser.add_argument("--test-images", type=int, default=96)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--eval-image-batch-size", type=int, default=16)
    parser.add_argument("--eval-vector-batch-size", type=int, default=4096)
    parser.add_argument("--dead-eval-batch-size", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--vtk-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-total-train-seconds", type=float, default=900.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5)
    parser.add_argument("--lr-scheduler-patience", type=int, default=4)
    parser.add_argument("--lr-scheduler-min", type=float, default=1e-5)
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--hidden-multiplier", type=int, default=4)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--vtk-target-l0", type=float, default=64.0)
    parser.add_argument("--vtk-kmax-factor", type=float, default=1.5)
    parser.add_argument("--vtk-lambda-prior", type=float, default=0.003)
    parser.add_argument("--vtk-beta", type=float, default=3e-4)
    parser.add_argument("--vtk-hard-weight", type=float, default=1.0)
    parser.add_argument("--vtk-expected-weight", type=float, default=1.0)
    parser.add_argument("--vtk-budget-weight", type=float, default=0.1)
    parser.add_argument("--suffix-check-images", type=int, default=16)
    parser.set_defaults(func=run_compare)
