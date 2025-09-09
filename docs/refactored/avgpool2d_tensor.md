# torch::avgpool2d / torch::avgPool2d (Tensor Operation)

Creates a 2D average pooling operation for downsampling 2D input tensors by computing the average value in each pooling window.

## Description

The `avgpool2d` command creates a 2D average pooling operation that applies average pooling over a 2D input signal composed of several input channels. Average pooling reduces the spatial dimensions (height and width) of feature maps while smoothing the signal by computing the average value from each 2D pooling window.

This operation is commonly used in Convolutional Neural Networks for:
- **Spatial dimensionality reduction**: Reducing the size of feature maps to decrease computation
- **Signal smoothing**: Creating smoother feature representations by averaging neighboring values
- **Translation invariance**: Making the network more robust to small translations in the input
- **Noise reduction**: Reducing noise in feature maps through local averaging

Average pooling is often preferred over max pooling when you want to preserve more information about the spatial distribution of features rather than just the maximum activation.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
# Basic usage with square kernel
torch::avgpool2d <input> <kernel_size>

# With explicit stride
torch::avgpool2d <input> <kernel_size> <stride>

# With padding
torch::avgpool2d <input> <kernel_size> <stride> <padding>

# With all parameters
torch::avgpool2d <input> <kernel_size> <stride> <padding> <count_include_pad>
```

### Named Parameter Syntax (Recommended)
```tcl
torch::avgpool2d -input <tensor> -kernel_size <size>
torch::avgpool2d -tensor <tensor> -kernelSize <size>

# With optional parameters
torch::avgpool2d -input <tensor> -kernel_size <size> -stride <stride> -padding <padding> -count_include_pad <bool>

# Using alternative parameter names
torch::avgpool2d -tensor <tensor> -kernelSize <size> -stride <stride> -padding <padding> -countIncludePad <bool>
```

### CamelCase Alias
```tcl
torch::avgPool2d <same_parameters_as_above>
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` / `tensor` | tensor | **Required** | Input 4D tensor with shape `[batch_size, channels, height, width]` |
| `kernel_size` / `kernelSize` | int or list | **Required** | Size of the pooling window. Can be a single integer for square kernels or `{height, width}` for rectangular kernels |
| `stride` | int or list | `kernel_size` | Stride of the pooling window. Can be a single integer or `{height, width}`. Defaults to `kernel_size` |
| `padding` | int or list | `0` | Zero-padding added to both sides. Can be a single integer or `{height, width}` |
| `count_include_pad` / `countIncludePad` | bool | `true` | Whether to include zero-padding in the averaging calculation |

## Input Requirements

- **Input tensor**: Must be a 4D tensor with shape `[batch_size, channels, height, width]`
- **Data types**: Supports floating-point types (float32, float64)
- **Device**: Can be CPU or CUDA tensors

## Output

Returns a new tensor with average pooled values. The output shape is:
```
[batch_size, channels, output_height, output_width]
```

Where:
- `output_height = floor((input_height + 2*padding_height - kernel_height) / stride_height) + 1`
- `output_width = floor((input_width + 2*padding_width - kernel_width) / stride_width) + 1`

## Examples

### Basic 2D Average Pooling
```tcl
# Create a 4D input tensor (batch=1, channels=1, height=4, width=4)
set input [torch::ones -shape {1 1 4 4} -dtype float32]

# Apply 2x2 average pooling
set result [torch::avgpool2d $input 2]
puts "Result shape: [torch::tensor_shape $result]"  ;# {1 1 2 2}
```

### Named Parameter Syntax
```tcl
# Using named parameters (recommended)
set input [torch::randn -shape {2 3 8 8} -dtype float32]
set result [torch::avgpool2d -input $input -kernel_size 3 -stride 2 -padding 1]
puts "Result shape: [torch::tensor_shape $result]"  ;# {2 3 4 4}
```

### Rectangular Kernel
```tcl
# Using rectangular pooling window
set input [torch::randn -shape {1 1 6 8} -dtype float32]
set result [torch::avgpool2d $input {2 3} {1 2}]
puts "Result shape: [torch::tensor_shape $result]"  ;# {1 1 5 3}
```

### Different Stride and Padding
```tcl
# Custom stride and padding
set input [torch::randn -shape {1 1 10 10} -dtype float32]
set result [torch::avgpool2d -input $input -kernel_size 3 -stride 3 -padding 1]
puts "Result shape: [torch::tensor_shape $result]"  ;# {1 1 4 4}
```

### Controlling Zero-padding in Averaging
```tcl
# Exclude padding from averaging (more accurate averages)
set input [torch::randn -shape {1 1 4 4} -dtype float32]
set result1 [torch::avgpool2d -input $input -kernel_size 3 -padding 1 -count_include_pad true]
set result2 [torch::avgpool2d -input $input -kernel_size 3 -padding 1 -count_include_pad false]

# result1 includes padding zeros in average calculation
# result2 excludes padding zeros (recommended for most cases)
```

### CamelCase Alias
```tcl
# Using camelCase alias
set input [torch::randn -shape {1 1 6 6} -dtype float32]
set result [torch::avgPool2d -input $input -kernelSize 2 -stride 2]
```

## Mathematical Background

Average pooling computes the arithmetic mean of values in each pooling window:

For a pooling window W at position (i,j):
```
output[i,j] = (1/|W|) × Σ input[m,n] for all (m,n) in W
```

Where |W| is the size of the pooling window (or effective size when `count_include_pad=false`).

### Padding Behavior
- **count_include_pad=true**: Zero-padding values are included in the average calculation
- **count_include_pad=false**: Only actual input values are averaged (recommended)

Example with 3x3 pooling and padding=1 on a 2x2 input:
```
Input:    Padded:     count_include_pad=true:  count_include_pad=false:
[1 2]     [0 0 0 0]   (0+0+0+0+1)/9 = 0.11    1/1 = 1.0
[3 4]     [0 1 2 0]   
          [0 3 4 0]
          [0 0 0 0]
```

## Common Use Cases

### 1. CNN Feature Map Downsampling
```tcl
# Downsample feature maps in CNN
set features [torch::randn -shape {32 64 28 28} -dtype float32]  ;# CIFAR-10 style
set pooled [torch::avgpool2d $features 2 2]  ;# Reduce to 14x14
```

### 2. Global Average Pooling (GAP)
```tcl
# Convert spatial features to single values per channel
set features [torch::randn -shape {1 512 7 7} -dtype float32]
set gap [torch::avgpool2d $features 7]  ;# Results in 1x512x1x1
```

### 3. Multi-scale Processing
```tcl
# Different scales of the same input
set input [torch::randn -shape {1 3 32 32} -dtype float32]
set scale1 [torch::avgpool2d $input 2]    ;# 16x16
set scale2 [torch::avgpool2d $input 4]    ;# 8x8
set scale3 [torch::avgpool2d $input 8]    ;# 4x4
```

### 4. Medical Image Processing
```tcl
# Reduce resolution of medical images while preserving intensity distributions
set medical_image [torch::randn -shape {1 1 512 512} -dtype float32]
set downsampled [torch::avgpool2d -input $medical_image -kernel_size 4 -stride 4]
# Results in 128x128 image with averaged intensities
```

## Performance Considerations

1. **Memory efficiency**: Average pooling reduces memory usage by decreasing spatial dimensions
2. **Computational cost**: O(N × C × H_out × W_out × K_h × K_w) where K_h, K_w are kernel dimensions
3. **GPU acceleration**: Fully supported on CUDA devices for large-scale processing
4. **Gradient flow**: Provides smoother gradients compared to max pooling during backpropagation

## Comparison with Max Pooling

| Aspect | Average Pooling | Max Pooling |
|--------|----------------|-------------|
| **Information preservation** | Better (retains all spatial information) | Sparse (only maximum values) |
| **Noise robustness** | Excellent (smooths noise) | Poor (may amplify noise) |
| **Sharp feature detection** | Poor (blurs edges) | Excellent (preserves strong features) |
| **Gradient flow** | Smooth gradients | Sparse gradients |
| **Use cases** | Global features, classification | Local features, object detection |

## Error Handling

The command provides clear error messages for common issues:

```tcl
# Invalid tensor name
catch {torch::avgpool2d "invalid_tensor" 2} msg
# Returns: "Invalid input tensor name"

# Missing required parameters
catch {torch::avgpool2d $input} msg  
# Returns error about missing kernel_size

# Invalid parameter names in named syntax
catch {torch::avgpool2d -input $input -invalid_param 2} msg
# Returns error about unknown parameter
```

## Relationship to PyTorch

This command corresponds to PyTorch's `torch.nn.functional.avg_pool2d()` function:

```python
# PyTorch equivalent
import torch.nn.functional as F
result = F.avg_pool2d(input, kernel_size, stride, padding, count_include_pad)
```

## See Also

- **torch::maxpool2d** - 2D max pooling operation
- **torch::avgpool1d** - 1D average pooling operation  
- **torch::avgpool3d** - 3D average pooling operation
- **torch::adaptive_avgpool2d** - Adaptive 2D average pooling
- **torch::conv2d** - 2D convolution operation

---

*This documentation covers the direct tensor operation `torch::avgpool2d`. For the PyTorch module layer version, see the `AvgPool2d` layer documentation.* 