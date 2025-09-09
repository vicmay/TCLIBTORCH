# torch::cross_map_lrn2d

Apply cross-map local response normalization (LRN) to 2D feature maps for neural network processing.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::cross_map_lrn2d -input tensor_handle -size size -alpha alpha -beta beta -k k
torch::crossMapLrn2d -input tensor_handle -size size -alpha alpha -beta beta -k k  # camelCase alias
```

### Positional Parameters (Legacy)
```tcl
torch::cross_map_lrn2d tensor_handle size alpha beta k
torch::crossMapLrn2d tensor_handle size alpha beta k  # camelCase alias
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | string | Yes | - | Handle/name of the input tensor (4D: NCHW format) |
| `-size` | integer | No | 5 | Size of the local neighborhood for normalization |
| `-alpha` | double | No | 1e-4 | Multiplicative factor for normalization |
| `-beta` | double | No | 0.75 | Exponent for normalization |
| `-k` | double | No | 1.0 | Additive factor for normalization |

## Returns

Returns a tensor handle containing the normalized output with the same shape and data type as the input.

## Description

Cross-map Local Response Normalization (LRN) is a normalization technique that normalizes across feature maps at the same spatial location. It implements lateral inhibition inspired by neurobiology, where activated neurons suppress the activity of nearby neurons.

The operation is defined as:
```
output[i] = input[i] / (k + alpha/size * sum(input[j]^2 for j in neighborhood))^beta
```

Where the neighborhood is defined across channels at the same spatial location.

## Examples

### Basic Usage
```tcl
# Create a 4D tensor (batch=1, channels=3, height=4, width=4)
set input [torch::randn {1 3 4 4}]

# Apply cross-map LRN using named syntax (recommended)
set output [torch::cross_map_lrn2d -input $input -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
puts "Output shape: [torch::tensor_shape $output]"

# Apply using positional syntax (legacy)
set output [torch::cross_map_lrn2d $input 5 1e-4 0.75 1.0]

# Using camelCase alias
set output [torch::crossMapLrn2d -input $input -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
```

### Using Default Parameters
```tcl
set input [torch::randn {2 16 32 32}]

# Use default parameters (size=5, alpha=1e-4, beta=0.75, k=1.0)
set output [torch::cross_map_lrn2d -input $input]
puts "Normalized tensor: $output"
```

### Different Parameter Values
```tcl
set input [torch::randn {1 64 8 8}]

# Stronger normalization with larger alpha
set strong_norm [torch::cross_map_lrn2d -input $input -alpha 1e-3]

# Larger neighborhood size
set large_neighborhood [torch::cross_map_lrn2d -input $input -size 7]

# Different beta for softer normalization
set soft_norm [torch::cross_map_lrn2d -input $input -beta 0.5]

# Custom k value
set custom_k [torch::cross_map_lrn2d -input $input -k 2.0]
```

### In a CNN Architecture
```tcl
# Create convolutional layers
set conv1 [torch::conv2d -inChannels 3 -outChannels 96 -kernelSize 11 -stride 4]
set conv2 [torch::conv2d -inChannels 96 -outChannels 256 -kernelSize 5 -stride 1 -padding 2]

# Forward pass through network
set input [torch::randn {32 3 224 224}]
set conv1_out [torch::module_forward $conv1 $input]
set relu1_out [torch::relu $conv1_out]

# Apply cross-map LRN after first convolution (AlexNet style)
set lrn1_out [torch::cross_map_lrn2d -input $relu1_out -size 5 -alpha 1e-4 -beta 0.75 -k 2.0]

# Continue with pooling
set pool1_out [torch::maxpool2d -input $lrn1_out -kernelSize 3 -stride 2]
```

### Batch Processing
```tcl
# Process multiple images in a batch
set batch_input [torch::randn {8 64 16 16}]  # batch_size=8, channels=64

# Apply normalization to entire batch
set normalized_batch [torch::cross_map_lrn2d -input $batch_input]
puts "Batch output shape: [torch::tensor_shape $normalized_batch]"
```

### Comparing Different Configurations
```tcl
set input [torch::randn {1 32 10 10}]

# Standard configuration (AlexNet)
set alexnet_lrn [torch::cross_map_lrn2d -input $input -size 5 -alpha 1e-4 -beta 0.75 -k 2.0]

# Milder normalization
set mild_lrn [torch::cross_map_lrn2d -input $input -size 3 -alpha 1e-5 -beta 0.5 -k 1.0]

# Stronger normalization
set strong_lrn [torch::cross_map_lrn2d -input $input -size 7 -alpha 1e-3 -beta 1.0 -k 1.0]
```

## Mathematical Details

### Normalization Formula
For input tensor `x` and output tensor `y`:
```
y[n,i,h,w] = x[n,i,h,w] / (k + (alpha/size) * sum(x[n,j,h,w]^2 for j in range))^beta
```

Where:
- `n` is the batch index
- `i` is the channel index  
- `h,w` are spatial coordinates
- `range` is the neighborhood around channel `i`

### Neighborhood Definition
The neighborhood includes channels `max(0, i-floor(size/2))` to `min(C-1, i+floor(size/2))` where `C` is the total number of channels.

### Parameter Effects
- **size**: Larger values create more smoothing across channels
- **alpha**: Higher values increase normalization strength
- **beta**: Controls the exponent of the divisive normalization
- **k**: Prevents division by zero and controls baseline

## Input Requirements

### Tensor Shape
- **Required**: 4D tensor in NCHW format (batch, channels, height, width)
- **Minimum**: Any positive dimensions
- **Typical**: `[N, C, H, W]` where `N` ≥ 1, `C` ≥ 1, `H` ≥ 1, `W` ≥ 1

### Data Types
- Supports: `float32`, `float64`, `float16`
- Preserves input data type in output

### Device Support
- CPU tensors: Full support
- CUDA tensors: Full support (if CUDA available)
- Preserves input device in output

## Error Handling

### Invalid Input Tensor
```tcl
# Non-existent tensor
catch {torch::cross_map_lrn2d -input "invalid_tensor"} error
puts $error  # Output: Tensor not found
```

### Missing Required Parameters
```tcl
# Missing input tensor
catch {torch::cross_map_lrn2d -size 5} error
puts $error  # Output: Required parameter missing: -input tensor_name

# Wrong number of positional arguments
catch {torch::cross_map_lrn2d $tensor 5} error
puts $error  # Output: Wrong number of arguments for positional syntax
```

### Invalid Parameter Values
```tcl
# Invalid size
catch {torch::cross_map_lrn2d -input $tensor -size 0} error
puts $error  # Output: Required parameter missing: -input tensor_name

# Invalid parameter type
catch {torch::cross_map_lrn2d -input $tensor -alpha "invalid"} error
puts $error  # Output: Invalid alpha parameter
```

### Unknown Parameters
```tcl
catch {torch::cross_map_lrn2d -input $tensor -unknown_param value} error
puts $error  # Output: Unknown parameter: -unknown_param
```

## Performance Considerations

### Memory Usage
- Output tensor has same size as input tensor
- Temporary memory proportional to input size
- No additional persistent memory requirements

### Computational Complexity
- Time complexity: O(N × C × H × W × size)
- Most efficient with moderate neighborhood sizes (3-7)
- GPU acceleration available for CUDA tensors

### Optimization Tips
```tcl
# Use smaller neighborhoods for better performance
set fast_lrn [torch::cross_map_lrn2d -input $tensor -size 3]

# Batch multiple operations together
set batch_tensor [torch::cat [list $tensor1 $tensor2 $tensor3] 0]
set batch_result [torch::cross_map_lrn2d -input $batch_tensor]
```

## Migration Guide

### From Positional to Named Parameters

**Before (Positional)**:
```tcl
set output [torch::cross_map_lrn2d $input 5 1e-4 0.75 1.0]
```

**After (Named Parameters)**:
```tcl
set output [torch::cross_map_lrn2d -input $input -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
```

**Or using camelCase**:
```tcl
set output [torch::crossMapLrn2d -input $input -size 5 -alpha 1e-4 -beta 0.75 -k 1.0]
```

### Using Defaults
**Before**: Required to specify all parameters
```tcl
set output [torch::cross_map_lrn2d $input 5 1e-4 0.75 1.0]
```

**After**: Can use convenient defaults
```tcl
set output [torch::cross_map_lrn2d -input $input]  # Uses all defaults
set output [torch::cross_map_lrn2d -input $input -size 3]  # Partial override
```

## Historical Context

Local Response Normalization was popularized by AlexNet (2012) and was commonly used in early CNN architectures. While batch normalization and layer normalization have largely replaced LRN in modern architectures, it remains useful for:

- Reproducing classical CNN architectures
- Research comparing normalization techniques  
- Specific applications requiring cross-channel competition
- Scenarios where batch statistics are not available

## Related Commands

- [`torch::local_response_norm`](local_response_norm.md) - General local response normalization
- [`torch::batch_norm2d`](batch_norm2d.md) - Modern batch normalization alternative
- [`torch::layer_norm`](layer_norm.md) - Layer normalization alternative
- [`torch::group_norm`](group_norm.md) - Group normalization alternative
- [`torch::instance_norm2d`](instance_norm2d.md) - Instance normalization alternative

## See Also

- [Normalization Techniques Guide](../normalization.md)
- [CNN Architecture Patterns](../cnn_patterns.md)
- [Historical CNN Architectures](../historical_cnns.md)
- [Performance Optimization](../performance.md) 