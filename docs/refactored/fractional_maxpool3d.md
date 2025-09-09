# `torch::fractional_maxpool3d` / `torch::fractionalMaxpool3d`

3-D fractional max pooling operation with random subsampling.
Supports both legacy positional syntax and the new named-parameter + camelCase API.

---
## 📜 Legacy Positional Syntax (Backward-Compatible)
```tcl
# torch::fractional_maxpool3d input kernel_size ?output_ratio?
set y [torch::fractional_maxpool3d $x {2 2 2} {0.5 0.5 0.5}]
```

## 🆕 Named-Parameter Syntax
```tcl
set y [torch::fractional_maxpool3d \
    -input        $x            ;# input tensor
    -kernel_size  {2 2 2}       ;# pooling kernel size
    -output_ratio {0.5 0.5 0.5} ;# output size ratio (default {0.5 0.5 0.5})
]
```

## 🆕 camelCase Alias
`torch::fractionalMaxpool3d` is a direct alias and accepts **both** syntaxes:
```tcl
# Positional with camelCase alias
set y [torch::fractionalMaxpool3d $x {2 2 2}]
# Named-parameter with camelCase alias
set y [torch::fractionalMaxpool3d -input $x -kernelSize {2 2 2} -outputRatio {0.6 0.6 0.6}]
```

---
## Parameters
| Name | Positional Index | Named Flag | Type | Default | Description |
|------|------------------|------------|------|---------|-------------|
| input | 1 | `-input` | tensor | – | Input tensor `N x C x D x H x W` |
| kernel_size | 2 | `-kernel_size` / `-kernelSize` | list(3) | – | Size of pooling window `{kD kH kW}` |
| output_ratio | 3 | `-output_ratio` / `-outputRatio` | list(3) | `{0.5 0.5 0.5}` | Output size as ratio of input `{rD rH rW}` |

### Parameter Details
- **kernel_size**: Three positive integers specifying pooling window size
- **output_ratio**: Three positive doubles between 0 and 1 specifying output size ratio
- Output size is calculated as `input_size * ratio` for each dimension

---
## Return Value
A new tensor representing the fractional max pooling result with random subsampling.

---
## Examples

### Basic Usage (Positional)
```tcl
set x [torch::randn -shape {1 1 8 8 8}]
set y [torch::fractional_maxpool3d $x {2 2 2}]
```

### Basic Usage (Named)
```tcl
set x [torch::randn -shape {1 1 8 8 8}]
set y [torch::fractional_maxpool3d -input $x -kernel_size {2 2 2} -output_ratio {0.6 0.6 0.6}]
```

### camelCase Alias
```tcl
set y [torch::fractionalMaxpool3d -input $x -kernelSize {3 3 3} -outputRatio {0.3 0.7 0.8}]
```

### Asymmetric Pooling
```tcl
# Different kernel sizes and output ratios for depth/height/width
set x [torch::randn -shape {1 1 12 16 12}]
set y [torch::fractional_maxpool3d -input $x -kernel_size {2 3 2} -output_ratio {0.4 0.8 0.7}]
```

### Video Processing
```tcl
# Works with video-like data (batch, channels, depth, height, width)
set x [torch::randn -shape {4 3 16 16 16}]
set y [torch::fractional_maxpool3d -input $x -kernel_size {2 2 2}]
```

---
## Migration Guide
1. **Basic**: `torch::fractional_maxpool3d $input {2 2 2}` → `torch::fractional_maxpool3d -input $input -kernel_size {2 2 2}`
2. **With ratio**: `torch::fractional_maxpool3d $input {2 2 2} {0.5 0.5 0.5}` → `torch::fractional_maxpool3d -input $input -kernel_size {2 2 2} -output_ratio {0.5 0.5 0.5}`
3. **camelCase**: Use `-kernelSize` and `-outputRatio` for modern style

---
## Algorithm Details
- **Fractional pooling** randomly selects pooling regions rather than using fixed grids
- **Random sampling** is performed internally for region selection across 3D space
- **Output size** is determined by the `output_ratio` parameter
- Each pooling operation uses different random samples, making results non-deterministic

---
## Error Handling
The command validates:
* Presence of `input` and `kernel_size`
* Tensor names existing in global storage
* `kernel_size` is a list of 3 positive integers
* `output_ratio` is a list of 3 positive doubles
* Parameter pairs for named syntax

Common errors:
```tcl
# ❌ Invalid kernel size (wrong dimensions)
torch::fractional_maxpool3d -input $x -kernel_size {2 2}  ;# Error: Expected 3 integers

# ❌ Invalid output ratio
torch::fractional_maxpool3d -input $x -kernel_size {2 2 2} -output_ratio {0.0 0.5 0.5}  ;# Error: Must be positive

# ❌ Missing required parameter
torch::fractional_maxpool3d -input $x  ;# Error: kernel_size missing
```

---
## Performance Notes
- Fractional max pooling introduces randomness and may be slower than regular max pooling
- Random sampling is performed on GPU when input tensor is on GPU
- Output size should be smaller than input size for meaningful pooling
- 3D operations are more computationally intensive than 2D equivalents

---
## Use Cases
- **3D Medical Imaging**: Downsampling volumetric data (CT, MRI scans)
- **Video Processing**: Temporal-spatial pooling in video analysis
- **3D Computer Vision**: Feature extraction from 3D point clouds or voxel data
- **Scientific Simulation**: Reducing resolution of 3D simulation data

---
## Test Coverage
See `tests/refactored/fractional_maxpool3d_test.tcl` for:
* Positional & named syntax functionality
* camelCase alias equivalence
* Parameter validation & error cases
* Various kernel sizes and output ratios
* Batch processing and edge cases
* 3D-specific test scenarios

---
© LibTorch TCL Extension – Dual Syntax API Modernization 