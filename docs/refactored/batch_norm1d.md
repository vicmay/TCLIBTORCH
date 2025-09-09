# torch::batch_norm1d

Applies **Batch Normalization** over a mini-batch of 1-D inputs (a 2D or 3D tensor where channel dimension is C).

Batch Normalization stabilizes and accelerates training by normalizing the activations of each feature channel to zero mean and unit variance, followed by learnable scale (`gamma`) and shift (`beta`) parameters (if `affine` is true).

---

## Syntax

### Positional (legacy)
```tcl
# torch::batch_norm1d num_features ?eps? ?momentum? ?affine? ?trackRunningStats?
set layer [torch::batch_norm1d 128]
```

### Named parameters (modern)
```tcl
# torch::batch_norm1d -numFeatures value ?-eps value? ?-momentum value? ?-affine bool? ?-trackRunningStats bool?
set layer [torch::batch_norm1d -numFeatures 128 -eps 1e-5 -momentum 0.1 -affine true -trackRunningStats true]
```

### camelCase alias
```tcl
set layer [torch::batchNorm1d -numFeatures 128]
```

Both syntaxes are fully supported and equivalent.

---

## Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `numFeatures` | `int` | **required** | Number of feature channels `C`. |
| `eps` | `double` | `1e-5` | Value added to denominator for numerical stability. |
| `momentum` | `double` | `0.1` | Momentum for running mean/variance statistics. |
| `affine` | `bool` | `true` | If `true`, layer has learnable scale (`gamma`) and shift (`beta`) parameters. |
| `trackRunningStats` | `bool` | `true` | If `true`, layer keeps running statistics during training and uses them in evaluation mode. |

### Positional order
1. `num_features`  
2. `eps`  
3. `momentum`  
4. `affine`  
5. `trackRunningStats`

---

## Examples

### Create and use a BatchNorm1d layer
```tcl
# Input tensor shape: (batch, channels, length) => (4, 128, 10)
set x      [torch::randn {4 128 10} float32 cpu false]
set bn     [torch::batch_norm1d 128]              ;# positional
set y_pos  [torch::layer_forward $bn $x]          ;# forward pass

# Same layer using named parameters and camelCase alias
set bn2    [torch::batchNorm1d -numFeatures 128]
set y_named [torch::layer_forward $bn2 $x]
```

### Migrating existing code
```tcl
# BEFORE
set bn [torch::batch_norm1d 64]

# AFTER (named parameters)
set bn [torch::batch_norm1d -numFeatures 64]
```
Both calls create equivalent layers; no code changes are required elsewhere.

---

## Error handling
* Missing required arguments raises a descriptive error.  
* Unknown parameters raise `Unknown parameter: -foo`.

---

## See also
* `torch::batch_norm2d` – BatchNorm over 4-D input (N,C,H,W)
* `torch::layer_norm` – Normalization over the last N dimensions

---

## Version history
* Added dual-syntax support and camelCase alias during API modernization. 