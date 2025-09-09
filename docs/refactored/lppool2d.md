# torch::lppool2d / torch::lpPool2d

Applies 2D LP (Lp norm) pooling over a 2D input signal composed of several 2D planes. LP pooling computes the Lp norm of elements in each 2D pooling window.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::lppool2d input norm_type kernel_size ?stride? ?ceil_mode?
```

### Modern Syntax (Named Parameters)
```tcl
torch::lppool2d -input <tensor> -normType <double> -kernelSize <int|list> ?-stride <int|list>? ?-ceilMode <bool>?
torch::lpPool2d -input <tensor> -normType <double> -kernelSize <int|list> ?-stride <int|list>? ?-ceilMode <bool>?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` / `-input` | tensor | Yes | - | Input tensor of shape (N, C, H, W) |
| `norm_type` / `-normType` | double | Yes | - | Type of norm (e.g., 1.0 for L1, 2.0 for L2) |
| `kernel_size` / `-kernelSize` | int or list | Yes | - | Size of pooling window (int or {H, W}) |
| `stride` / `-stride` | int or list | No | `kernel_size` | Stride of pooling window (int or {H, W}) |
| `ceil_mode` / `-ceilMode` | boolean | No | `false` | Use ceil instead of floor for output shape |

## Returns

Returns a tensor handle representing the pooled tensor with reduced spatial dimensions.

## Mathematical Background

2D LP pooling computes the Lp norm over 2D spatial regions:

```
output[i,j] = (∑∑|x[m,n]|^p)^(1/p)
```

Where:
- `p` is the norm type (`norm_type`)
- The double sum is over all elements `(m,n)` in the 2D pooling window
- Special cases:
  - `p = 1`: L1 norm (sum of absolute values)
  - `p = 2`: L2 norm (Euclidean norm, energy preservation)
  - `p → ∞`: L∞ norm (maximum absolute value)

## Examples

### Basic 2D LP Pooling

```tcl
# L2 pooling (Euclidean norm) with square kernel
set input [torch::randn -shape {1 3 8 8}]
set output [torch::lppool2d $input 2.0 3]

# L1 pooling (Manhattan norm)
set input [torch::randn -shape {1 3 8 8}]
set output [torch::lppool2d $input 1.0 2]

# Large norm (approximates 2D max pooling)
set input [torch::randn -shape {1 3 8 8}]
set output [torch::lppool2d $input 100.0 2]
```

### Modern Syntax Examples

```tcl
# L2 pooling with named parameters
set input [torch::randn -shape {1 3 10 10}]
set output [torch::lppool2d -input $input -normType 2.0 -kernelSize 3 -stride 2]

# Non-square kernel and stride
set input [torch::randn -shape {1 3 12 16}]
set output [torch::lppool2d -input $input -normType 2.0 -kernelSize {3 4} -stride {2 3}]

# camelCase alias (recommended)
set input [torch::randn -shape {1 3 8 8}]
set output [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2 -ceilMode true]
```

### Different Kernel Configurations

```tcl
set input [torch::randn -shape {2 3 12 12}]

# Square kernel (3x3)
set square_output [torch::lpPool2d -input $input -normType 2.0 -kernelSize 3]

# Rectangular kernel (3x2)
set rect_output [torch::lpPool2d -input $input -normType 2.0 -kernelSize {3 2}]

# Different strides for height and width
set asym_output [torch::lpPool2d -input $input -normType 2.0 -kernelSize {2 2} -stride {1 2}]
```

### Norm Type Variations

```tcl
set input [torch::randn -shape {1 3 8 8}]

# L1 norm (sum of absolute values)
set l1_output [torch::lpPool2d -input $input -normType 1.0 -kernelSize 2]

# L2 norm (Euclidean norm, energy preservation)
set l2_output [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2]

# L∞ approximation (large p value)
set linf_output [torch::lpPool2d -input $input -normType 50.0 -kernelSize 2]

# Fractional norm
set frac_output [torch::lpPool2d -input $input -normType 1.5 -kernelSize 2]
```

### Advanced Configuration

```tcl
set input [torch::randn -shape {4 64 32 32}]

# Overlapping windows for feature extraction
set overlapping [torch::lpPool2d -input $input -normType 2.0 -kernelSize 3 -stride 1]

# Non-overlapping downsampling
set downsampled [torch::lpPool2d -input $input -normType 2.0 -kernelSize 4 -stride 4]

# Ceiling mode for complete input coverage
set ceil_pooled [torch::lpPool2d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode true]
```

## Output Shape Calculation

For input shape `(N, C, H_in, W_in)`, the output shape is `(N, C, H_out, W_out)` where:

```
H_out = floor((H_in - kernel_h) / stride_h) + 1    (if ceil_mode = false)
W_out = floor((W_in - kernel_w) / stride_w) + 1

H_out = ceil((H_in - kernel_h) / stride_h) + 1     (if ceil_mode = true)
W_out = ceil((W_in - kernel_w) / stride_w) + 1
```

### Shape Examples

```tcl
# Square input and kernel
# Input: (1, 3, 8, 8), kernel: 2x2, stride: 2x2
# Output: (1, 3, 4, 4)

# Rectangular input with non-square kernel
# Input: (1, 3, 10, 12), kernel: 3x2, stride: 2x3
# H_out = floor((10-3)/2) + 1 = 4
# W_out = floor((12-2)/3) + 1 = 4
# Output: (1, 3, 4, 4)

set input [torch::randn -shape {1 3 10 12}]
set output [torch::lpPool2d -input $input -normType 2.0 -kernelSize {3 2} -stride {2 3}]
puts [torch::tensor_shape $output]  ;# {1 3 4 4}
```

## Computer Vision Applications

### 1. Feature Map Downsampling
```tcl
# Reduce spatial resolution while preserving important features
set feature_maps [torch::randn -shape {32 256 64 64}]
set downsampled [torch::lpPool2d -input $feature_maps -normType 2.0 -kernelSize 2 -stride 2]
puts [torch::tensor_shape $downsampled]  ;# {32 256 32 32}
```

### 2. Texture Analysis
```tcl
# L1 norm is robust for texture features
set texture_image [torch::randn -shape {1 3 128 128}]
set texture_features [torch::lpPool2d -input $texture_image -normType 1.0 -kernelSize 4]
```

### 3. Energy-Based Pooling
```tcl
# L2 norm preserves energy content in image patches
set image_patches [torch::randn -shape {1 64 32 32}]
set energy_pooled [torch::lpPool2d -input $image_patches -normType 2.0 -kernelSize 4 -stride 4]
```

### 4. Multi-Scale Feature Extraction
```tcl
set input [torch::randn -shape {1 128 64 64}]

# Different scales with different kernel sizes
set scale1 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2]    # Fine
set scale2 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 4]    # Medium
set scale3 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 8]    # Coarse
```

## Comparison with Other 2D Pooling Methods

| Method | Operation | Characteristics | Use Cases |
|--------|-----------|-----------------|-----------|
| **LP Pool (p=1)** | Sum of absolute values | Linear, robust to outliers | Texture analysis, robust features |
| **LP Pool (p=2)** | Euclidean norm | Energy preservation | Signal processing, smooth transitions |
| **Max Pool** | Maximum value | Non-linear, sparse | Edge detection, translation invariance |
| **Avg Pool** | Arithmetic mean | Linear, smooth | General downsampling, noise reduction |

```tcl
set input [torch::randn -shape {1 3 8 8}]

# Compare different pooling methods
set lp1_pool [torch::lpPool2d -input $input -normType 1.0 -kernelSize 2]
set lp2_pool [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2]
set max_pool [torch::maxpool2d $input 2]
set avg_pool [torch::avgpool2d $input 2]

puts "LP1 shape: [torch::tensor_shape $lp1_pool]"
puts "LP2 shape: [torch::tensor_shape $lp2_pool]"
puts "Max shape: [torch::tensor_shape $max_pool]"
puts "Avg shape: [torch::tensor_shape $avg_pool]"
```

## Advanced Usage Patterns

### Asymmetric Pooling for Rectangular Features
```tcl
# Horizontal features (wide kernels)
set horizontal_pool [torch::lpPool2d -input $input -normType 2.0 -kernelSize {1 4}]

# Vertical features (tall kernels)
set vertical_pool [torch::lpPool2d -input $input -normType 2.0 -kernelSize {4 1}]
```

### Progressive Downsampling
```tcl
set input [torch::randn -shape {1 64 64 64}]

# Stage 1: 64x64 -> 32x32
set stage1 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2 -stride 2]

# Stage 2: 32x32 -> 16x16
set stage2 [torch::lpPool2d -input $stage1 -normType 2.0 -kernelSize 2 -stride 2]

# Stage 3: 16x16 -> 8x8
set stage3 [torch::lpPool2d -input $stage2 -normType 2.0 -kernelSize 2 -stride 2]
```

### Overlapping Windows for Dense Features
```tcl
# Dense feature extraction with overlapping windows
set dense_features [torch::lpPool2d -input $input -normType 2.0 -kernelSize 3 -stride 1]
```

## Error Handling

```tcl
# Invalid norm type
catch {torch::lpPool2d -input $input -normType -1.0 -kernelSize 3} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: ... normType must be positive"

# Invalid kernel size list
catch {torch::lpPool2d -input $input -normType 2.0 -kernelSize {3 2 1}} error_msg
puts $error_msg  ;# "List must have length 2"

# Missing required parameters
catch {torch::lpPool2d -input $input -normType 2.0} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: ... kernelSize must be positive"
```

## Performance Considerations

1. **Kernel Size Impact**: Larger kernels process more elements per output pixel
2. **Stride Configuration**: Larger strides reduce computational cost and output size
3. **Norm Type**: Higher p values require more computation (powers and roots)
4. **Memory Usage**: 2D operations require more memory than 1D equivalents
5. **Non-Square Kernels**: May have different cache performance characteristics

### Performance Tips
```tcl
# Efficient: Power-of-2 kernel sizes often perform better
set efficient [torch::lpPool2d -input $input -normType 2.0 -kernelSize 4 -stride 4]

# Efficient: L2 norm is well-optimized
set optimized [torch::lpPool2d -input $input -normType 2.0 -kernelSize 2]

# Less efficient: Non-power-of-2 or very large kernels
set less_efficient [torch::lpPool2d -input $input -normType 3.7 -kernelSize 13]
```

## Related Commands

- `torch::lppool1d` - 1D LP pooling
- `torch::lppool3d` - 3D LP pooling
- `torch::maxpool2d` - 2D max pooling
- `torch::avgpool2d` - 2D average pooling
- `torch::adaptive_avgpool2d` - Adaptive 2D average pooling

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy)
set output [torch::lppool2d $input 2.0 {3 2} {2 1} true]

# NEW (Modern - equivalent)
set output [torch::lpPool2d -input $input -normType 2.0 -kernelSize {3 2} -stride {2 1} -ceilMode true]
```

### Parameter Mapping

| Legacy Position | Modern Parameter | Description |
|----------------|------------------|-------------|
| 1st argument | `-input` | Input tensor |
| 2nd argument | `-normType` | LP norm type (1.0, 2.0, etc.) |
| 3rd argument | `-kernelSize` | Pooling window size (int or {H, W}) |
| 4th argument | `-stride` | Stride (optional, default: kernelSize) |
| 5th argument | `-ceilMode` | Ceiling mode (optional, default: false) |

### Kernel Size Specifications

```tcl
# Square kernel - both syntaxes
set square1 [torch::lppool2d $input 2.0 3]                    # Legacy
set square2 [torch::lpPool2d -input $input -normType 2.0 -kernelSize 3]  # Modern

# Rectangular kernel - both syntaxes  
set rect1 [torch::lppool2d $input 2.0 {3 2}]                  # Legacy
set rect2 [torch::lpPool2d -input $input -normType 2.0 -kernelSize {3 2}]  # Modern
```

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Both syntaxes supported, camelCase (`torch::lpPool2d`) recommended

## Notes

- Both `torch::lppool2d` and `torch::lpPool2d` refer to the same implementation
- The camelCase alias `torch::lpPool2d` is recommended for new code
- Legacy positional syntax remains fully supported for backward compatibility
- 2D LP pooling is differentiable and suitable for training neural networks
- Kernel size can be specified as single integer (square) or list {H, W} (rectangular)
- For very large p values, LP pooling approximates max pooling behavior
- Memory usage scales with kernel size and input dimensions 