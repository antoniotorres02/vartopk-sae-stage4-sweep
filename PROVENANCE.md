# Provenance record

This repository publishes, as a self-contained unit, the ConvNeXt Stage-4 SAE
sweep behind the TFM memory. It was assembled by copying files, not by
re-implementing them.

## Source of the code

| Field | Value |
| --- | --- |
| Development repository | `tfm_sae_evals_clean` (private) |
| Git worktree | `.worktrees/convnext-stage4-logspace-sweep` |
| Commit at sweep time | `414ec9d` — *Add variable TopK SAE evaluations* |
| Working-tree state | the sweep-mode flags and the `sae.py` model definitions were **uncommitted** at that revision; the copied files are the working-tree versions that produced the checkpoints |
| Remote of the source repo | `github.com/antoniotorres02/vtopk_sae_evals` (private) |

Files copied verbatim: `src/tfm_sae_evals/**`, `tests/**`, `pyproject.toml`,
`requirements.txt`, `config/external_paths.example.toml`,
`docs/convnext_stage4_variable_topk_architectures.md`, and the sweep outputs under
`results/convnext_stage4_geomspace_sweep/`. Nothing in the training, model or
evaluation code was modified for publication. Only added files are new
(`run_sweep.sh`, `config/sparse_hyperparameters.csv`, `scripts/*`, `README.md`,
this record, `SHA256SUMS`, `LICENSE`, `.gitignore`).

## Source of the artifacts

| Artifact | Original location |
| --- | --- |
| 40 checkpoints | `<source>/outputs/convnext_stage4_geomspace_sweep/checkpoints/` |
| 4 reused checkpoints (targets 8 and 256) | `<source>/outputs/convnext_stage4_logspace_sweep/checkpoints/` |
| Stage-4 cache (`stage4_maps_subset512.pt`) | `<source>/outputs/convnext_stage4_geomspace_sweep/features/` |
| Frozen backbone | `tfm_sae_evals_clean/data/feature_explorer/checkpoints/cub_convnext_tiny_classifier.pt` |

The four reused checkpoints are the ones referenced by the results CSV for the
`topk_nonorm` and `variable_topk_original` rows at targets 8 and 256; their
`checkpoint_path` column points into the log-space sweep folder. They are
published under `checkpoints/logspace_reuse/` so that the 4 × 11 grid is complete.

## Consistency checks run before publication

1. **Backbone identity.** Feeding the cached 512-image test maps through the
   published backbone reproduces the accuracy recorded by the sweep for the
   original (unreconstructed) classifier:
   `0.90234375` — identical to the `original_top1_accuracy` column of every row.
2. **Checkpoint metrics.** `scripts/verify_published_checkpoints.py` compares the
   `eval_metrics` embedded in each checkpoint with its CSV row
   (test MSE, NMSE, `L0`, top-1 agreement, reconstructed accuracy, logit KL;
   tolerance `1e-6`): **44/44 rows match**.
3. **Sparse hyperparameters.** `config/sparse_hyperparameters.csv` was derived
   from the campaign's `hyperparameters.txt`; loading it with the experiment's own
   `load_sparse_hyperparams()` reproduces the `sparsity_lambda` of all 22 sparse
   runs (0 discrepancies).
4. **Package tests.** `pytest tests` passes in this copy of the package.

## Reproduction fidelity notes

* The original campaign combined the CLI sweep mode with per-target invocations;
  `run_sweep.sh` reproduces the same 44 runs in one invocation using the sweep
  mode plus the recorded per-target coefficients. Timings may differ.
* The sweep's `stage4_arch_compare_report.md` reflects the arguments of the last
  invocation of the campaign, not the whole sweep; the authoritative record of the
  44 runs is `stage4_arch_compare_results.csv` and `hyperparameters.txt`.
* Checkpoints are `torch.save` dictionaries; load them with
  `weights_only=False` (they contain plain Python configuration values).
