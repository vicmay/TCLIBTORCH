# `torch::fractional_maxpool2d` / `torch::fractionalMaxpool2d`

2-D fractional max pooling operation with random subsampling.
Supports both legacy positional syntax and the new named-parameter + camelCase API.

---
## 📜 Legacy Positional Syntax (Backward-Compatible)
```tcl
# torch::fractional_maxpool2d input kernel_size ?output_ratio?
set y [torch::fractional_maxpool2d $x {2 2} {0.5 0.5}]
```

## 🆕 Named-Parameter Syntax
```tcl
set y [torch::fractional_maxpool2d \
    -input        $x        ;# input tensor
    -kernel_size  {2 2}     ;# pooling kernel size
    -output_ratio {0.5 0.5} ;# output size ratio (default {0.5 0.5})
]
```

## 🆕 camelCase Alias
`torch::fractionalMaxpool2d` is a direct alias and accepts **both** syntaxes:
```tcl
# Positional with camelCase alias
set y [torch::fractionalMaxpool2d $x {2 2}]
# Named-parameter with camelCase alias
set y [torch::fractionalMaxpool2d -input $x -kernelSize {2 2} -outputRatio {0.6 0.6}]
```

---
## Parameters
| Name | Positional Index | Named Flag | Type | Default | Description |
|------|------------------|------------|------|---------|-------------|
| input | 1 | `-input` | tensor | – | Input tensor `N x C x H x W` |
| kernel_size | 2 | `-kernel_size` / `-kernelSize` | list(2) | – | Size of pooling window `{kH kW}` |
| output_ratio | 3 | `-output_ratio` / `-outputRatio` | list(2) | `{0.5 0.5}` | Output size as ratio of input `{rH rW}` |

### Parameter Details
- **kernel_size**: Two positive integers specifying pooling window size
- **output_ratio**: Two positive doubles between 0 and 1 specifying output size ratio
- Output size is calculated as `input_size * ratio` for each dimension

---
## Return Value
A new tensor representing the fractional max pooling result with random subsampling.

---
## Examples

### Basic Usage (Positional)
```tcl
set x [torch::randn -shape {1 1 8 8}]
set y [torch::fractional_maxpool2d $x {2 2}]
```

### Basic Usage (Named)
```tcl
set x [torch::randn -shape {1 1 8 8}]
set y [torch::fractional_maxpool2d -input $x -kernel_size {2 2} -output_ratio {0.6 0.6}]
```

### camelCase Alias
```tcl
set y [torch::fractionalMaxpool2d -input $x -kernelSize {3 3} -outputRatio {0.3 0.7}]
```

### Asymmetric Pooling
```tcl
# Different kernel sizes and output ratios for height/width
set x [torch::randn -shape {1 1 12 16}]
set y [torch::fractional_maxpool2d -input $x -kernel_size {2 3} -output_ratio {0.4 0.8}]
```

### Batch Processing
```tcl
# Works with batched inputs
set x [torch::randn -shape {4 3 16 16}]
set y [torch::fractional_maxpool2d -input $x -kernel_size {2 2}]
```

---
## Migration Guide
1. **Basic**: `torch::fractional_maxpool2d $input {2 2}` → `torch::fractional_maxpool2d -input $input -kernel_size {2 2}`
2. **With ratio**: `torch::fractional_maxpool2d $input {2 2} {0.5 0.5}` → `torch::fractional_maxpool2d -input $input -kernel_size {2 2} -output_ratio {0.5 0.5}`
3. **camelCase**: Use `-kernelSize` and `-outputRatio` for modern style

---
## Algorithm Details
- **Fractional pooling** randomly selects pooling regions rather than using fixed grids
- **Random sampling** is performed internally for region selection
- **Output size** is determined by the `output_ratio` parameter
- Each pooling operation uses different random samples, making results non-deterministic

---
## Error Handling
The command validates:
* Presence of `input` and `kernel_size`
* Tensor names existing in global storage
* `kernel_size` is a list of 2 positive integers
* `output_ratio` is a list of 2 positive doubles
* Parameter pairs for named syntax

Common errors:
```tcl
# ❌ Invalid kernel size
torch::fractional_maxpool2d -input $x -kernel_size {2 2 2}  ;# Error: Expected 2 integers

# ❌ Invalid output ratio
torch::fractional_maxpool2d -input $x -kernel_size {2 2} -output_ratio {0.0 0.5}  ;# Error: Must be positive

# ❌ Missing required parameter
torch::fractional_maxpool2d -input $x  ;# Error: kernel_size missing
```

---
## Performance Notes
- Fractional max pooling introduces randomness and may be slower than regular max pooling
- Random sampling is performed on GPU when input tensor is on GPU
- Output size should be smaller than input size for meaningful pooling

---
## Test Coverage
See `tests/refactored/fractional_maxpool2d_test.tcl` for:
* Positional & named syntax functionality
* camelCase alias equivalence
* Parameter validation & error cases
* Various kernel sizes and output ratios
* Batch processing and edge cases

---
© LibTorch TCL Extension – Dual Syntax API Modernization 