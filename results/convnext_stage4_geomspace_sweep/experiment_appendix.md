* [ ] Appendix: ConvNeXt Stage4 SAE Sweep Setup

This appendix documents the experimental setup used for the ConvNeXt Stage4
SAE comparison stored in `results/convnext_stage4_geomspace_sweep/`. The run
compares fixed TopK, Variable TopK, L1 SAE, and JumpReLU SAE models on
ConvNeXt-Tiny Stage4 activation maps.

## Feature Source

The explained model is a ConvNeXt-Tiny classifier trained on CUB. The sparse
autoencoders are trained on activations extracted from the last ConvNeXt feature
stage, immediately before global average pooling and the classifier suffix.

| Field                        | Value                               |
| ---------------------------- | ----------------------------------- |
| Backbone                     | ConvNeXt-Tiny                       |
| Dataset                      | CUB processed split                 |
| Hook                         | `backbone.features[7]`            |
| Stage name                   | `stage4`                          |
| Activation map shape         | `[N, 768, 7, 7]`                  |
| SAE input vectorization      | `[N, 768, 7, 7] -> [N * 49, 768]` |
| SAE input dimension          | `768`                             |
| Hidden dimension             | `3072`                            |
| Hidden multiplier            | `4`                               |
| Number of classifier classes | `200`                             |

For each image split, the activation maps are cached as `float16` tensors. SAE
training is performed on vectorized spatial positions. Evaluation reconstructs
the full Stage4 activation map, replays the frozen ConvNeXt suffix, and measures
both reconstruction and downstream prediction metrics.

## Data Subset

The experiment uses a fixed subset size of 512 examples per split. The subset is
deterministic: it takes the first 512 examples of each split rather than a
random or stratified sample.

| Split      | Images | SAE vectors |
| ---------- | -----: | ----------: |
| Train      |    512 |      25,088 |
| Validation |    512 |      25,088 |
| Test       |    512 |      25,088 |

The top-1 metrics therefore have a granularity of `1 / 512 = 0.1953125`
percentage points.

## Compared Architectures

| Method key                 | Description                                                   | Sparsity control                                           | Decoder normalization |
| -------------------------- | ------------------------------------------------------------- | ---------------------------------------------------------- | --------------------- |
| `topk_nonorm`            | Fixed TopK sparse autoencoder                                 | Fixed `K`                                                | Disabled              |
| `variable_topk_original` | Variable TopK with a learned distribution over prefix lengths | `Kmax`, prior regularization                             | Disabled              |
| `l1_sae`                 | ReLU SAE with L1 activation penalty                           | L1 regularization coefficient                              | Enabled               |
| `jumprelu_sae`           | JumpReLU SAE with expected-L0 surrogate penalty               | L0-surrogate regularization coefficient and threshold gate | Enabled               |

TopK fixes the number of active features directly. Variable TopK uses an
explicit `Kmax`. L1 SAE and JumpReLU SAE do not have an explicit `Kmax`; their
effective sparsity is controlled indirectly through regularization strength.
Consequently, comparisons should be read as functions of measured effective
`L0`, not only as functions of the nominal target.

## Training Configuration

| Field                              | Value                |
| ---------------------------------- | -------------------- |
| Seed                               | `42`               |
| Device                             | CUDA, when available |
| Optimizer                          | Adam                 |
| Learning rate                      | `1e-3`             |
| LR scheduler                       | ReduceLROnPlateau    |
| LR scheduler factor                | `0.5`              |
| LR scheduler patience              | `10`               |
| Minimum LR                         | `1e-5`             |
| Max epochs                         | `300`              |
| Early stopping patience            | `30`               |
| EMA alpha for early stopping       | `0.3`              |
| Minimum improvement delta          | `1e-6`             |
| TopK/L1/JumpReLU vector batch size | `4096`             |
| Variable TopK vector batch size    | `512`              |
| Image extraction batch size        | `32`               |
| Evaluation image batch size        | `16`               |
| Evaluation vector batch size       | `4096`             |
| Dead-feature evaluation batch size | `4096`             |
| Suffix replay check images         | `32`               |

The training objective for fixed TopK is reconstruction MSE. For L1 SAE and
JumpReLU SAE, the validation objective used for early stopping includes the
same sparsity penalty as training.

## Sweep Values

The nominal sweep values are generated on a geometric scale:

```python
np.round(np.geomspace(8, 256, num=11)).astype(int)
```

This gives:

```text
8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256
```

For fixed TopK, the nominal target is the actual fixed `K`. For Variable TopK,
the nominal target sets `Kmax` because `kmax_factor = 1.0`. For L1 SAE and
JumpReLU SAE, the nominal target is used only to organize the sweep and select
regularization coefficients; the measured effective `L0` can differ from the
nominal target.

## Method-Specific Hyperparameters

### Fixed TopK

| Nominal target | Fixed K |
| -------------: | ------: |
|              8 |       8 |
|             11 |      11 |
|             16 |      16 |
|             23 |      23 |
|             32 |      32 |
|             45 |      45 |
|             64 |      64 |
|             91 |      91 |
|            128 |     128 |
|            181 |     181 |
|            256 |     256 |

### Variable TopK

| Hyperparameter      |      Value |
| ------------------- | ---------: |
| `kmax_factor`     |    `1.0` |
| `lambda_prior`    |  `0.003` |
| `beta`            | `0.0003` |
| Expected MSE weight |    `1.0` |
| Hard MSE weight     |    `0.0` |
| Budget weight       |    `0.0` |

| Nominal target | Kmax | Effective L0 |
| -------------: | ---: | -----------: |
|              8 |    8 |         6.58 |
|             11 |   11 |         8.80 |
|             16 |   16 |        12.93 |
|             23 |   23 |        19.24 |
|             32 |   32 |        29.18 |
|             45 |   45 |        43.20 |
|             64 |   64 |        60.23 |
|             91 |   91 |        83.77 |
|            128 |  128 |       116.30 |
|            181 |  181 |       177.47 |
|            256 |  256 |       245.39 |

### L1 SAE

The L1 objective is:

```text
MSE + lambda_L1 * mean(sum(abs(hidden_activations)))
```

| Nominal target | L1 lambda | Effective L0 |
| -------------: | --------: | -----------: |
|              8 |    0.0300 |        13.66 |
|             11 |    0.0350 |        11.72 |
|             16 |    0.0250 |        19.03 |
|             23 |    0.0180 |        37.20 |
|             32 |    0.0140 |        57.64 |
|             45 |    0.0115 |        77.76 |
|             64 |    0.0100 |        97.80 |
|             91 |    0.0085 |       122.41 |
|            128 |    0.0073 |       146.41 |
|            181 |    0.0062 |       167.88 |
|            256 |    0.0052 |       213.65 |

### JumpReLU SAE

The JumpReLU objective is:

```text
MSE + lambda_JumpReLU * mean(L0_surrogate)
```

The threshold parameter is initialized from `theta_init`, but is represented in
the model as a learnable log-threshold.

| Hyperparameter |    Value |
| -------------- | -------: |
| `theta_init` | `0.01` |
| `bandwidth`  | `0.05` |

| Nominal target | JumpReLU lambda | Theta init | Bandwidth | Effective L0 |
| -------------: | --------------: | ---------: | --------: | -----------: |
|              8 |         0.02500 |       0.01 |      0.05 |         8.39 |
|             11 |         0.01800 |       0.01 |      0.05 |        11.69 |
|             16 |         0.01200 |       0.01 |      0.05 |        16.87 |
|             23 |         0.00800 |       0.01 |      0.05 |        25.23 |
|             32 |         0.00600 |       0.01 |      0.05 |        30.54 |
|             45 |         0.00450 |       0.01 |      0.05 |        40.43 |
|             64 |         0.00330 |       0.01 |      0.05 |        54.52 |
|             91 |         0.00220 |       0.01 |      0.05 |        80.74 |
|            128 |         0.00150 |       0.01 |      0.05 |       112.07 |
|            181 |         0.00100 |       0.01 |      0.05 |       156.71 |
|            256 |         0.00065 |       0.01 |      0.05 |       223.56 |

## Metrics

All reported metrics are computed on the 512-image test subset.

| Metric                       | Definition                                                                                                             |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Effective L0                 | Mean number of active SAE features per vectorized Stage4 position.                                                     |
| Test MSE                     | MSE between reconstructed and original Stage4 maps in the unnormalized activation space.                               |
| Test NMSE                    | Test MSE divided by the variance of the original test Stage4 maps.                                                     |
| Dead percentage              | Percentage of hidden features that never activate on the training vectors.                                             |
| Top-1 agreement              | Fraction of test images for which the reconstructed suffix prediction matches the original ConvNeXt suffix prediction. |
| Original Top-1 accuracy      | Accuracy of the original ConvNeXt predictions against labels on the test subset.                                       |
| Reconstructed Top-1 accuracy | Accuracy after replacing Stage4 maps with SAE reconstructions and replaying the ConvNeXt suffix.                       |

The original ConvNeXt top-1 accuracy on this subset is `90.234375%`.

## Result Summary

| Method                     | Nominal target | Effective L0 | Test MSE | Top-1 agreement (%) | Recon. Top-1 accuracy (%) |
| -------------------------- | -------------: | -----------: | -------: | ------------------: | ------------------------: |
| `topk_nonorm`            |              8 |         8.00 | 0.051286 |               89.06 |                     84.38 |
| `topk_nonorm`            |             11 |        11.00 | 0.049568 |               89.84 |                     84.77 |
| `topk_nonorm`            |             16 |        16.00 | 0.047622 |               91.41 |                     85.94 |
| `topk_nonorm`            |             23 |        23.00 | 0.045580 |               91.80 |                     88.48 |
| `topk_nonorm`            |             32 |        32.00 | 0.043412 |               93.16 |                     87.50 |
| `topk_nonorm`            |             45 |        45.00 | 0.040817 |               94.34 |                     87.70 |
| `topk_nonorm`            |             64 |        64.00 | 0.038080 |               94.73 |                     87.70 |
| `topk_nonorm`            |             91 |        91.00 | 0.034700 |               95.70 |                     88.09 |
| `topk_nonorm`            |            128 |       128.00 | 0.031068 |               95.51 |                     88.87 |
| `topk_nonorm`            |            181 |       181.00 | 0.026914 |               96.48 |                     89.45 |
| `topk_nonorm`            |            256 |       256.00 | 0.022633 |               97.46 |                     89.06 |
| `variable_topk_original` |              8 |         6.58 | 0.051157 |               88.48 |                     84.18 |
| `variable_topk_original` |             11 |         8.80 | 0.049355 |               89.26 |                     85.55 |
| `variable_topk_original` |             16 |        12.93 | 0.047115 |               92.38 |                     86.91 |
| `variable_topk_original` |             23 |        19.24 | 0.044483 |               90.04 |                     86.13 |
| `variable_topk_original` |             32 |        29.18 | 0.042212 |               91.99 |                     86.91 |
| `variable_topk_original` |             45 |        43.20 | 0.039549 |               94.92 |                     87.70 |
| `variable_topk_original` |             64 |        60.23 | 0.036787 |               94.34 |                     87.89 |
| `variable_topk_original` |             91 |        83.77 | 0.033652 |               95.12 |                     87.30 |
| `variable_topk_original` |            128 |       116.30 | 0.030586 |               94.73 |                     88.87 |
| `variable_topk_original` |            181 |       177.47 | 0.026621 |               95.12 |                     89.06 |
| `variable_topk_original` |            256 |       245.39 | 0.022794 |               96.09 |                     89.06 |
| `l1_sae`                 |              8 |        13.66 | 0.073942 |                4.69 |                      4.30 |
| `l1_sae`                 |             11 |        11.72 | 0.081359 |                0.00 |                      0.00 |
| `l1_sae`                 |             16 |        19.03 | 0.064078 |               54.30 |                     52.93 |
| `l1_sae`                 |             23 |        37.20 | 0.048641 |               88.67 |                     83.20 |
| `l1_sae`                 |             32 |        57.64 | 0.041687 |               91.80 |                     85.94 |
| `l1_sae`                 |             45 |        77.76 | 0.037642 |               93.16 |                     87.50 |
| `l1_sae`                 |             64 |        97.80 | 0.035001 |               93.36 |                     87.11 |
| `l1_sae`                 |             91 |       122.41 | 0.032377 |               93.95 |                     88.28 |
| `l1_sae`                 |            128 |       146.41 | 0.030502 |               95.31 |                     88.48 |
| `l1_sae`                 |            181 |       167.88 | 0.029309 |               94.92 |                     87.70 |
| `l1_sae`                 |            256 |       213.65 | 0.026230 |               96.09 |                     88.67 |
| `jumprelu_sae`           |              8 |         8.39 | 0.068184 |               31.45 |                     29.69 |
| `jumprelu_sae`           |             11 |        11.69 | 0.063296 |               57.42 |                     53.71 |
| `jumprelu_sae`           |             16 |        16.87 | 0.059425 |               76.76 |                     72.85 |
| `jumprelu_sae`           |             23 |        25.23 | 0.055803 |               80.27 |                     76.56 |
| `jumprelu_sae`           |             32 |        30.54 | 0.052098 |               87.11 |                     84.18 |
| `jumprelu_sae`           |             45 |        40.43 | 0.049638 |               89.65 |                     85.16 |
| `jumprelu_sae`           |             64 |        54.52 | 0.046663 |               91.21 |                     87.30 |
| `jumprelu_sae`           |             91 |        80.74 | 0.041903 |               93.95 |                     89.06 |
| `jumprelu_sae`           |            128 |       112.07 | 0.037239 |               94.34 |                     89.45 |
| `jumprelu_sae`           |            181 |       156.71 | 0.031958 |               95.51 |                     89.65 |
| `jumprelu_sae`           |            256 |       223.56 | 0.026860 |               95.90 |                     89.84 |

## Generated Artifacts

The following plots are generated in both PNG and PDF format in
`results/convnext_stage4_geomspace_sweep/`:

| File stem                                            | Description                                                                     |
| ---------------------------------------------------- | ------------------------------------------------------------------------------- |
| `stage4_arch_compare_sparse_mse_l0`                | Test MSE versus effective L0 for TopK, Variable TopK, L1 SAE, and JumpReLU SAE. |
| `stage4_arch_compare_sparse_top1_accuracy_l0`      | Reconstructed top-1 accuracy versus effective L0 for the same baselines.        |
| `stage4_arch_compare_sparse_top1_agreement_l0`     | Top-1 agreement with the original ConvNeXt prediction versus effective L0.      |
| `stage4_arch_compare_mse_l0`                       | TopK versus Variable TopK MSE sweep.                                            |
| `stage4_arch_compare_vartopk_relative_improvement` | Relative MSE improvement of Variable TopK over interpolated TopK.               |

The raw result table is stored in `stage4_arch_compare_results.csv`.

## Caveats Specific to This Setup

- The evaluation uses only the first 512 examples from each split. This is not a
  random or stratified subset.
- Only one seed is reported.
- L1 SAE and JumpReLU SAE are compared by measured effective L0. Their nominal
  target values do not fix the number of active features.
- Variable TopK uses `budget_weight = 0.0`; the nominal target sets `Kmax`, but
  there is no explicit budget penalty forcing the expected L0 to match the
  target.
- The batch size differs between Variable TopK (`512` vectors) and the other
  SAEs (`4096` vectors), so the number of optimizer steps per epoch is not
  identical across all methods.
- Several L1 SAE runs reach the maximum epoch budget, so the L1 curve should be
  interpreted as the result of this specific hyperparameter sweep rather than a
  fully optimized L1 Pareto frontier.
- Test MSE is computed in unnormalized Stage4 activation space, whereas the SAE
  training loader normalizes vectors with train-set mean and standard deviation.
- The top-1 metrics are discrete over 512 images; changes smaller than a few
  tenths of a percentage point should not be overinterpreted.

## Reproducibility Notes

The experiment code path is
`src/tfm_sae_evals/experiments/convnext_stage4_compare.py`. The plotting script
for the stored result directory is
`results/convnext_stage4_geomspace_sweep/generate_plots.py`.

The key command pattern for the TopK and Variable TopK sweep was:

```bash
PYTHONPATH=src .venv/bin/python -m tfm_sae_evals.cli.main \
  convnext-stage4-compare \
  --config config/external_paths.example.toml \
  --methods topk_nonorm variable_topk_original \
  --subset-size 512 \
  --epochs 300 \
  --sweep-logspace \
  --sweep-min-k 8 \
  --sweep-max-k 256 \
  --sweep-points 11 \
  --vtk-kmax-factor 1 \
  --vtk-budget-weight 0 \
  --vtk-hard-weight 0
```

The additional L1 SAE and JumpReLU SAE baselines were added using the same data
extraction and evaluation setup, with per-target regularization coefficients
listed above.
