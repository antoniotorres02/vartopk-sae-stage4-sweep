# ConvNeXt Stage4 Variable TopK Architectures

The Stage4 comparison uses ConvNeXt-Tiny CUB maps from `backbone.features[7]`,
before global average pooling and the classifier.

```text
[N, 768, 7, 7] -> [N * 7 * 7, 768]
```

Each spatial location in the `7 x 7` map becomes one SAE training vector with
768 channels.

## Compared Architectures

| Architecture | K selector input | K selector | Decoder normalization | Notes |
| --- | --- | --- | --- | --- |
| `topk_nonorm` | Fixed `k=64` | None | Disabled | Matched fixed-K baseline. |
| `variable_topk_original` | Normalized SAE input vector `x` | `Linear(d -> k_max)` | Disabled by architecture | Original Variable TopK baseline in this clean repo. |
| `variable_topk_topvals_mlp_nonorm` | `concat(log1p(topk_vals), cumulative_topk_mass)` | `MLP(2*k_max -> 256 -> k_max)` | Disabled | Variant used in the Stage4 timing pilot. |

## Key Difference

The original Variable TopK model predicts K directly from the input vector:

```text
k_logits = Linear(x)
```

The TopVals MLP variant first computes the encoder activations, sorts the top
`k_max` activations, and predicts K from activation evidence:

```text
topk_vals = topk(ReLU(encoder(x)), k_max)
selector_input = concat(log1p(topk_vals), cumulative_topk_mass)
k_logits = MLP(selector_input)
```

This gives the selector direct access to how much useful activation mass is
available in the sorted latent prefix. The Stage4 timing pilot result should
therefore be attributed to `variable_topk_topvals_mlp_nonorm`, not to the
original linear-selector Variable TopK architecture.
