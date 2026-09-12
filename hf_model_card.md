---
license: mit
tags:
- sparse-autoencoder
- sae
- topk
- variable-topk
- jumprelu
- interpretability
- cub
- convnext
- activations
---

# ConvNeXt Stage-4 SAE checkpoints (fixed TopK · Variable TopK · L1 · JumpReLU)

Checkpoints of the 44 sparse-autoencoder runs compared in the TFM memory
*Toward Neurosymbolic Sparse Autoencoders: A Bayesian Framework for Adaptive
Concept Discovery*, together with the cached ConvNeXt-Tiny Stage-4 activations and
the frozen CUB classifier they were trained on.

**Code that trains and evaluates these models:
[github.com/antoniotorres02/vartopk-sae-stage4-sweep](https://github.com/antoniotorres02/vartopk-sae-stage4-sweep)**
(reproduces the sweep with `bash run_sweep.sh` and verifies these files with
`python scripts/verify_published_checkpoints.py`).

## Files

| Path | Files | Size | Contents |
| --- | --- | ---: | --- |
| `checkpoints/geomspace_sweep/` | 40 | 724 MB | the sweep: fixed TopK and Variable TopK (9 targets each) + L1 and JumpReLU (11 targets each) |
| `checkpoints/logspace_reuse/` | 4 | 72 MB | TopK and Variable TopK at targets 8 and 256, reused from the preceding log-space sweep to complete the 4 × 11 grid |
| `features/stage4_maps_subset512.pt` | 1 | 111 MB | cached Stage-4 maps and labels for the 512-image train/val/test subsets (`float16`, `[512, 768, 7, 7]` + `[512]`) |
| `backbone/cub_convnext_tiny_classifier.pt` | 1 | 107 MB | frozen ConvNeXt-Tiny CUB classifier, the model being explained |
| `results/` | 4 | — | results CSV (44 rows), report, hyperparameter record, protocol appendix |
| `SHA256SUMS` | 1 | — | digests of every checkpoint, the cache and the backbone |

File naming: `stage4_<method>_k<target>.pt` with `method` in
`topk_nonorm`, `variable_topk_original`, `l1_sae`, `jumprelu_sae` and `target`
from `round(geomspace(8, 256, 11))` = 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256.

## What each checkpoint contains

```python
import torch

ckpt = torch.load("checkpoints/geomspace_sweep/stage4_variable_topk_original_k64.pt",
                  map_location="cpu", weights_only=False)
ckpt["method"]        # 'variable_topk_original'
ckpt["config"]        # {'d': 768, 'hidden_dim': 3072, 'k_max': 64, 'target_l0': 64.0, ...}
ckpt["train_info"]    # {'best_epoch': 36, 'epochs_trained': 69, 'stopped_reason': 'early_stop_patience_30', ...}
ckpt["eval_metrics"]  # {'test_mse': ..., 'test_nmse': ..., 'l0': ..., 'top1_agreement': ...,
                      #  'reconstructed_top1_accuracy': ..., 'original_top1_accuracy': 0.90234375, 'logit_kl': ...}
ckpt["mean"], ckpt["std"]        # activation normalization statistics used at training time
ckpt["model_state_dict"]         # weights of the SAE
```

The model is rebuilt from `ckpt["method"]` + `ckpt["config"]` by
`build_model(method, args, device)` in
`src/tfm_sae_evals/experiments/convnext_stage4_compare.py` of the code repository.
Load with `weights_only=False`: the checkpoints store plain Python configuration
alongside tensors.

Cached activations:

```python
cache = torch.load("features/stage4_maps_subset512.pt", map_location="cpu", weights_only=False)
cache["test_maps"].shape    # torch.Size([512, 768, 7, 7])  float16
cache["test_labels"].shape  # torch.Size([512])             int64
cache["train_maps"], cache["val_maps"]  # same layout
```

## Recorded results

Every checkpoint's `eval_metrics` matches its row in
`results/convnext_stage4_geomspace_sweep/stage4_arch_compare_results.csv`
(verified for all 44 rows at `1e-6` tolerance). Summary of the comparison at the
11 shared nominal targets:

| Comparison (Variable TopK vs fixed TopK) | Outcome |
| --- | --- |
| lower average `L0` | **11 / 11** |
| lower test MSE | **10 / 11** |
| higher top-1 agreement | 9 / 11 |
| higher reconstructed top-1 accuracy | 5 / 11 |

Frozen ConvNeXt-Tiny top-1 accuracy on the 512-image test subset: **0.90234375**.
Decoder normalization is disabled for the two TopK families and enabled for L1 and
JumpReLU.

## License

MIT for the code and the released artifacts. CUB-200-2011 and its processed splits
keep their original terms.
