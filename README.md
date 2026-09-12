# ConvNeXt Stage-4 SAE sweep: fixed TopK, Variable TopK, L1 and JumpReLU

Code that trains and evaluates the four sparse-autoencoder (SAE) families compared
in the TFM memory *Toward Neurosymbolic Sparse Autoencoders: A Bayesian Framework
for Adaptive Concept Discovery*. Each SAE reconstructs ConvNeXt-Tiny Stage-4
activations from CUB-200-2011 and is scored on reconstruction error, achieved
sparsity, and preservation of the frozen classifier's behaviour.

**Trained artifacts (44 checkpoints + cached activations + frozen backbone):
[Hugging Face · antoniotorres02/vartopk-sae-stage4-checkpoints](https://huggingface.co/antoniotorres02/vartopk-sae-stage4-checkpoints)**

Reference numbers for every run live in
[`results/convnext_stage4_geomspace_sweep/stage4_arch_compare_results.csv`](results/convnext_stage4_geomspace_sweep/stage4_arch_compare_results.csv).

---

## Scope of this repository

This is an **isolated copy of the sweep only**. The implementation under `src/`,
`tests/`, `pyproject.toml` and `requirements.txt` is copied verbatim from the
private development repository; nothing in the training or evaluation path was
rewritten for publication. What is *not* here: the CAM/interpretability
evaluations, the LLM (Pythia) experiments, the Oxford-Pets transfer runs, the
full-dataset convergence campaign and the exploratory notebooks of the parent
project.

## Experiment at a glance

| Field | Value |
| --- | --- |
| Explained model | ConvNeXt-Tiny classifier trained on CUB-200-2011 (`cub_convnext_tiny_classifier.pt`) |
| Hook | `backbone.features[7]` (last stage, before global average pooling) |
| Activation map | `[N, 768, 7, 7]` |
| SAE training vectors | `[N, 768, 7, 7] -> [49N, 768]` (each spatial position is one example) |
| SAE input / hidden dim | `d = 768`, `4d = 3072` |
| Families | fixed TopK, Variable TopK (V²k-SAE), L1 SAE, JumpReLU SAE |
| Nominal targets | `round(geomspace(8, 256, 11))` = 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256 |
| Data subset | first 512 images of each CUB split (train/val/test) → 25,088 vectors per split |
| Optimizer | Adam, `lr = 1e-3`, ≤ 300 epochs |
| Early stopping | validation MSE EMA, `patience = 30`; `ReduceLROnPlateau(0.5, patience 10, min 1e-5)` |
| Seed | 42 |
| Batch size | 4096 vectors (TopK, L1, JumpReLU); 512 (Variable TopK, larger memory footprint) |
| Variable TopK hyperparameters | `beta = 3e-4`, `lambda_prior = 3e-3`, `Kmax = nominal target`, expected-MSE weight 1.0 |
| Sparse baselines | per-target `lambda` in [`config/sparse_hyperparameters.csv`](config/sparse_hyperparameters.csv); JumpReLU `theta_init = 0.01`, bandwidth `0.05` |
| Measured cost (this campaign) | ≈ 196 min total, ≤ 2.9 GB peak GPU memory |

Decoder normalization is disabled for the two TopK families and enabled for L1 and
JumpReLU, because disabling it degraded their empirical performance.

## Results recorded by this sweep

44 completed runs, all with `status = ok`. Comparing Variable TopK against fixed
TopK at each of the 11 shared nominal targets:

| Comparison | Outcome |
| --- | --- |
| Lower average `L0` | **11 / 11** |
| Lower test MSE | **10 / 11** |
| Higher top-1 agreement with the original ConvNeXt prediction | 9 / 11 |
| Higher reconstructed top-1 accuracy | 5 / 11 |

Original ConvNeXt top-1 accuracy on this 512-image test subset: **0.90234375**
(51 % granularity step, `1/512`). These numbers reproduce Table III of the memory;
see also `stage4_arch_compare_report.md` and `hyperparameters.txt` in the results
folder.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or, for the package + CLI entry point:
pip install -e .
```

Requires Python ≥ 3.12 (see `pyproject.toml`) and PyTorch with CUDA for the
timings above; the code falls back to CPU with `--cpu`.

## Data and artifacts

1. **CUB-200-2011**. Raw images and the processed splits:

   | Key | Expected layout |
   | --- | --- |
   | `cub_raw_dir` | `CUB_200_2011/` with `images/` and class metadata |
   | `cub_processed_dir` | `class_attr_data_10/` with `train.pkl`, `val.pkl`, `test.pkl` |

   The processed split layout is the one distributed with the Concept Bottleneck
   Models release (`yewsiang/ConceptBottleneck`, `CUB_processed/class_attr_data_10`).

2. **Frozen backbone and cached activations** from Hugging Face:

   ```bash
   bash scripts/prepare_artifacts.sh            # downloads into ./artifacts
   ```

3. **Configuration**: copy the example and edit the four keys (all must exist,
   `sae-evals check-paths` verifies them):

   ```bash
   cp config/external_paths.example.toml config/external_paths.toml
   ```

   ```toml
   cub_raw_dir        = "/path/to/CUB_200_2011/CUB_200_2011"
   cub_processed_dir  = "/path/to/CUB_processed/class_attr_data_10"
   convnext_checkpoint = "artifacts/backbone/cub_convnext_tiny_classifier.pt"
   cub_feature_cache   = "artifacts/features/stage4_maps_subset512.pt"
   ```

   `cub_feature_cache` is used by other experiments of the parent project and only
   has to point at an existing file here. The Stage-4 cache this sweep actually
   reads is `outputs/convnext_stage4_geomspace_sweep/features/stage4_maps_train512_val512_test512.pt`;
   `scripts/prepare_artifacts.sh` places it there, so the feature-extraction pass is
   skipped and the sweep starts directly from the cached maps.

## Reproduce the sweep

```bash
bash run_sweep.sh          # 44 runs, one CLI invocation
```

`run_sweep.sh` expands to:

```bash
PYTHONPATH=src python -m tfm_sae_evals.cli.main convnext-stage4-compare \
  --config config/external_paths.toml \
  --fresh \
  --methods topk_nonorm variable_topk_original l1_sae jumprelu_sae \
  --subset-size 512 --epochs 300 \
  --sweep-logspace --sweep-min-k 8 --sweep-max-k 256 --sweep-points 11 \
  --vtk-kmax-factor 1.0 --vtk-lambda-prior 0.003 --vtk-beta 0.0003 \
  --vtk-hard-weight 0.0 --vtk-expected-weight 1.0 --vtk-budget-weight 0.0 \
  --sparse-hyperparams-csv config/sparse_hyperparameters.csv \
  --output-dir outputs/convnext_stage4_geomspace_sweep \
  --results-dir results/convnext_stage4_geomspace_sweep
```

The `--sweep-*` flags step through the geometric target list; the per-target
regularization coefficients of the sparse baselines come from
`config/sparse_hyperparameters.csv` (they were selected in a preliminary
coefficient pilot; only the final values are kept here, and they match the
`sparsity_lambda` column of the recorded results). Rows already completed are
skipped unless `--fresh` is given. The command writes checkpoints under
`outputs/<run>/checkpoints/`, per-run rows to the results CSV, and the Pareto /
accuracy / agreement figures to the results folder.

## Verify the published checkpoints

Each checkpoint stores the metrics that produced its CSV row, so the published
files can be checked against the reported table:

```bash
hf download antoniotorres02/vartopk-sae-stage4-checkpoints --local-dir artifacts
python scripts/verify_published_checkpoints.py \
  --checkpoints-root artifacts/checkpoints \
  --sha256sums artifacts/SHA256SUMS
```

Expected output ends with:

```
44/44 rows match their checkpoint metrics.
46 file digests verified, 0 failed.
Verification OK.
```

The same script also works against the original development layout
(`--checkpoints-root` pointing at the sweep `outputs/.../checkpoints` folder).

## Published artifacts (Hugging Face)

| Path | Files | Size | Contents |
| --- | --- | ---: | --- |
| `checkpoints/geomspace_sweep/` | 40 | 724 MB | the sweep itself: TopK / Variable TopK (9 targets each) + L1 / JumpReLU (11 targets each) |
| `checkpoints/logspace_reuse/` | 4 | 72 MB | TopK and Variable TopK at targets 8 and 256, reused from the preceding log-space sweep to complete the 4 × 11 grid |
| `features/stage4_maps_subset512.pt` | 1 | 111 MB | cached Stage-4 maps and labels for the 512-image train/val/test subsets (float16) |
| `backbone/cub_convnext_tiny_classifier.pt` | 1 | 107 MB | frozen ConvNeXt-Tiny CUB classifier (the explained model) |
| `results/` | 3 | — | results CSV, report and hyperparameter record |
| `SHA256SUMS` | 1 | — | digests of every checkpoint, the cache and the backbone |

Each `.pt` checkpoint is a dictionary with `method`, `config` (dimensions, `k` /
`k_max` / `target_l0`), `model_state_dict`, the activation-normalization statistics
`mean` / `std`, `train_info` (best epoch, epochs trained, stopping reason) and
`eval_metrics` (test MSE/NMSE, `L0`, top-1 agreement, reconstructed accuracy, logit KL).
Models are rebuilt by `build_model(method, args, device)` in
`src/tfm_sae_evals/experiments/convnext_stage4_compare.py`.

## Repository layout

```
config/
  external_paths.example.toml          paths to CUB, backbone and caches
  sparse_hyperparameters.csv           per-target lambda for L1 and JumpReLU
docs/
  convnext_stage4_variable_topk_architectures.md
results/convnext_stage4_geomspace_sweep/
  stage4_arch_compare_results.csv      44 rows: the recorded numbers
  stage4_arch_compare_report.md        summary, metrics and paired statistics
  hyperparameters.txt                  full hyperparameter record
  experiment_appendix.md               protocol description
  *.png / *.pdf                        Pareto, accuracy and agreement figures
run_sweep.sh                           one-command reproduction of the sweep
scripts/
  prepare_artifacts.sh                 download HF artifacts into ./artifacts
  verify_published_checkpoints.py      checkpoint metrics vs results CSV
src/tfm_sae_evals/                     the package (verbatim copy)
tests/                                 pytest suite of the package
SHA256SUMS                             digests of the published artifacts
```

## License

MIT (see `LICENSE`). Third-party data and models keep their own terms: CUB-200-2011
and the processed splits follow their original licences, and the published
checkpoints are research artifacts of this project.
