# torch::lppool1d / torch::lpPool1d

Applies 1D LP (Lp norm) pooling over an input signal. LP pooling computes the Lp norm of elements in each pooling window.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::lppool1d input norm_type kernel_size ?stride? ?ceil_mode?
```

### Modern Syntax (Named Parameters)
```tcl
torch::lppool1d -input <tensor> -normType <double> -kernelSize <int> ?-stride <int>? ?-ceilMode <bool>?
torch::lpPool1d -input <tensor> -normType <double> -kernelSize <int> ?-stride <int>? ?-ceilMode <bool>?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` / `-input` | tensor | Yes | - | Input tensor of shape (N, C, L) |
| `norm_type` / `-normType` | double | Yes | - | Type of norm (e.g., 1.0 for L1, 2.0 for L2) |
| `kernel_size` / `-kernelSize` | integer | Yes | - | Size of the pooling window |
| `stride` / `-stride` | integer | No | `kernel_size` | Stride of the pooling window |
| `ceil_mode` / `-ceilMode` | boolean | No | `false` | Use ceil instead of floor to compute output shape |

## Returns

Returns a tensor handle representing the pooled tensor with reduced spatial dimensions.

## Mathematical Background

LP pooling computes the Lp norm over spatial regions:

```
output[i] = (∑|x[j]|^p)^(1/p)
```

Where:
- `p` is the norm type (`norm_type`)
- The sum is over all elements `j` in the pooling window
- Special cases:
  - `p = 1`: L1 norm (sum of absolute values)
  - `p = 2`: L2 norm (Euclidean norm)
  - `p → ∞`: L∞ norm (maximum absolute value)

## Examples

### Basic LP Pooling

```tcl
# L2 pooling (Euclidean norm)
set input [torch::randn -shape {1 4 8}]
set output [torch::lppool1d $input 2.0 3]

# L1 pooling (Manhattan norm)
set input [torch::randn -shape {1 4 8}]
set output [torch::lppool1d $input 1.0 3]

# Large norm (approximates max pooling)
set input [torch::randn -shape {1 4 8}]
set output [torch::lppool1d $input 100.0 3]
```

### Modern Syntax Examples

```tcl
# L2 pooling with named parameters
set input [torch::randn -shape {1 4 10}]
set output [torch::lppool1d -input $input -normType 2.0 -kernelSize 3 -stride 2]

# L1 pooling with ceil mode
set input [torch::randn -shape {1 4 9}]
set output [torch::lppool1d -input $input -normType 1.0 -kernelSize 3 -ceilMode true]

# camelCase alias (recommended)
set input [torch::randn -shape {1 4 12}]
set output [torch::lpPool1d -input $input -normType 2.0 -kernelSize 4 -stride 2]
```

### Different Norm Types

```tcl
set input [torch::randn -shape {2 3 8}]

# L1 norm (sum of absolute values)
set l1_output [torch::lpPool1d -input $input -normType 1.0 -kernelSize 2]

# L2 norm (Euclidean norm)
set l2_output [torch::lpPool1d -input $input -normType 2.0 -kernelSize 2]

# L3 norm
set l3_output [torch::lpPool1d -input $input -normType 3.0 -kernelSize 2]

# Fractional norm
set frac_output [torch::lpPool1d -input $input -normType 1.5 -kernelSize 2]
```

### Stride and Output Shape Control

```tcl
set input [torch::randn -shape {1 4 12}]

# Default stride (equals kernel size)
set output1 [torch::lpPool1d -input $input -normType 2.0 -kernelSize 3]

# Custom stride for overlapping windows
set output2 [torch::lpPool1d -input $input -normType 2.0 -kernelSize 3 -stride 1]

# Larger stride for downsampling
set output3 [torch::lpPool1d -input $input -normType 2.0 -kernelSize 2 -stride 4]

puts [torch::tensor_shape $output1]  ;# {1 4 4}
puts [torch::tensor_shape $output2]  ;# {1 4 10}
puts [torch::tensor_shape $output3]  ;# {1 4 3}
```

### Ceiling Mode Effects

```tcl
set input [torch::randn -shape {1 4 9}]

# Floor mode (default)
set floor_output [torch::lpPool1d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode false]

# Ceil mode (may produce larger output)
set ceil_output [torch::lpPool1d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode true]

puts [torch::tensor_shape $floor_output]  ;# May be {1 4 3}
puts [torch::tensor_shape $ceil_output]   ;# May be {1 4 4}
```

## Common Use Cases

### 1. Feature Extraction
```tcl
# Robust feature pooling with L2 norm
set features [torch::randn -shape {32 256 64}]
set pooled_features [torch::lpPool1d -input $features -normType 2.0 -kernelSize 4]
```

### 2. Signal Processing
```tcl
# Energy-based pooling for audio signals
set audio_signal [torch::randn -shape {1 1 16000}]
set energy_pooled [torch::lpPool1d -input $audio_signal -normType 2.0 -kernelSize 1024 -stride 512]
```

### 3. Downsampling with Norm Preservation
```tcl
# Preserve signal magnitude while downsampling
set high_res [torch::randn -shape {1 64 1024}]
set low_res [torch::lpPool1d -input $high_res -normType 2.0 -kernelSize 4 -stride 4]
```

### 4. Robust Pooling
```tcl
# L1 norm is less sensitive to outliers than L2
set noisy_signal [torch::randn -shape {1 32 128}]
set robust_pooled [torch::lpPool1d -input $noisy_signal -normType 1.0 -kernelSize 3]
```

## Output Shape Calculation

For input shape `(N, C, L_in)`, the output shape is `(N, C, L_out)` where:

```
L_out = floor((L_in - kernel_size) / stride) + 1        (if ceil_mode = false)
L_out = ceil((L_in - kernel_size) / stride) + 1         (if ceil_mode = true)
```

### Examples

```tcl
# Input: (1, 4, 10), kernel_size=3, stride=2
# L_out = floor((10 - 3) / 2) + 1 = floor(3.5) + 1 = 4
# Output shape: (1, 4, 4)

# Input: (1, 4, 9), kernel_size=3, stride=2, ceil_mode=true
# L_out = ceil((9 - 3) / 2) + 1 = ceil(3) + 1 = 4
# Output shape: (1, 4, 4)
```

## Comparison with Other Pooling Methods

| Method | Operation | Characteristics |
|--------|-----------|-----------------|
| **LP Pool (p=1)** | Sum of absolute values | Linear, robust to outliers |
| **LP Pool (p=2)** | Euclidean norm | Preserves energy, smooth gradients |
| **Max Pool** | Maximum value | Non-linear, sparse gradients |
| **Avg Pool** | Arithmetic mean | Linear, uniform weighting |

```tcl
set input [torch::randn -shape {1 1 8}]

# Compare different pooling methods
set lp1_pool [torch::lpPool1d -input $input -normType 1.0 -kernelSize 2]
set lp2_pool [torch::lpPool1d -input $input -normType 2.0 -kernelSize 2]
set max_pool [torch::maxpool1d $input 2]
set avg_pool [torch::avgpool1d $input 2]
```

## Error Handling

```tcl
# Invalid norm type
catch {torch::lpPool1d -input $input -normType -1.0 -kernelSize 3} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: ... normType must be positive"

# Missing required parameters
catch {torch::lpPool1d -input $input -normType 2.0} error_msg
puts $error_msg  ;# "Required parameters missing or invalid: ... kernelSize must be positive"

# Invalid tensor
catch {torch::lpPool1d -input invalid_tensor -normType 2.0 -kernelSize 3} error_msg
puts $error_msg  ;# "Invalid input tensor name"
```

## Performance Considerations

1. **Norm Type Impact**: Higher norm values (p > 2) require more computation
2. **Kernel Size**: Larger kernels process more elements per output
3. **Stride**: Larger strides reduce output size and computation
4. **Memory**: LP pooling requires temporary storage for norm computation

## Related Commands

- `torch::maxpool1d` - 1D max pooling
- `torch::avgpool1d` - 1D average pooling
- `torch::lppool2d` - 2D LP pooling
- `torch::lppool3d` - 3D LP pooling
- `torch::adaptive_avgpool1d` - Adaptive average pooling

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy)
set output [torch::lppool1d $input 2.0 3 2 true]

# NEW (Modern - equivalent)
set output [torch::lpPool1d -input $input -normType 2.0 -kernelSize 3 -stride 2 -ceilMode true]
```

### Parameter Mapping

| Legacy Position | Modern Parameter | Description |
|----------------|------------------|-------------|
| 1st argument | `-input` | Input tensor |
| 2nd argument | `-normType` | LP norm type (1.0, 2.0, etc.) |
| 3rd argument | `-kernelSize` | Pooling window size |
| 4th argument | `-stride` | Stride (optional, default: kernelSize) |
| 5th argument | `-ceilMode` | Ceiling mode (optional, default: false) |

### Benefits of Modern Syntax

1. **Self-documenting**: Parameter names clarify purpose
2. **Flexible ordering**: Parameters can be specified in any order
3. **Extensible**: Easy to add new parameters
4. **Consistent**: Matches PyTorch API conventions
5. **Type safety**: Better error messages

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Both syntaxes supported, camelCase (`torch::lpPool1d`) recommended

## Notes

- Both `torch::lppool1d` and `torch::lpPool1d` refer to the same implementation
- The camelCase alias `torch::lpPool1d` is recommended for new code
- Legacy positional syntax remains fully supported for backward compatibility
- LP pooling is differentiable and suitable for training neural networks
- For `norm_type` approaching infinity, LP pooling approximates max pooling 