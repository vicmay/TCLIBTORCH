# torch::maxpool2d / torch::maxPool2d

Creates a 2D max pooling layer for downsampling input feature maps by taking the maximum value in each pooling window.

## Description

The `maxpool2d` command creates a 2D max pooling layer that applies maximum pooling over a 2D input signal composed of several input channels. Max pooling reduces the spatial dimensions of feature maps while preserving the most important features by selecting the maximum value from each pooling window.

This layer is commonly used in Convolutional Neural Networks (CNNs) for:
- **Spatial dimensionality reduction**: Reducing the size of feature maps to decrease computation
- **Translation invariance**: Making the network more robust to small translations in the input
- **Feature extraction**: Highlighting the most prominent features in each region
- **Overfitting prevention**: Providing a form of regularization through downsampling

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::maxpool2d kernel_size ?stride? ?padding?
```

### Named Parameter Syntax (New)
```tcl
torch::maxpool2d -kernelSize kernel_size ?-stride stride? ?-padding padding?
torch::maxpool2d -kernel_size kernel_size ?-stride stride? ?-padding padding?
```

### CamelCase Alias
```tcl
torch::maxPool2d kernel_size ?stride? ?padding?
torch::maxPool2d -kernelSize kernel_size ?-stride stride? ?-padding padding?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kernel_size` | integer | **required** | Size of the pooling window (e.g., 2 for 2×2 window) |
| `stride` | integer | `kernel_size` | Step size for sliding the pooling window |
| `padding` | integer | `0` | Zero-padding added to both sides of the input |

## Parameter Details

### Kernel Size
- **Required parameter** - must be a positive integer
- Defines the size of the square pooling window
- Common values: 2 (for 2×2), 3 (for 3×3)
- Larger kernels reduce output size more aggressively

### Stride
- **Optional parameter** - defaults to `kernel_size` if not specified
- Determines how much the pooling window moves at each step
- Stride = kernel_size → non-overlapping windows (typical usage)
- Stride < kernel_size → overlapping windows
- Stride > kernel_size → some input regions are skipped

### Padding
- **Optional parameter** - defaults to 0 (no padding)
- Adds zeros around the input borders
- Useful for maintaining output size or handling border effects
- Must be non-negative

## Return Value

Returns a layer handle (string) that can be used with:
- `torch::layer_forward` - Apply the pooling to input tensors
- `torch::sequential` - Add to sequential models
- Other layer management functions

## Output Size Calculation

The output spatial dimensions are calculated as:
```
output_size = floor((input_size + 2×padding - kernel_size) / stride) + 1
```

For example, with input 32×32, kernel_size=2, stride=2, padding=0:
```
output_size = floor((32 + 0 - 2) / 2) + 1 = floor(30/2) + 1 = 15 + 1 = 16
```

## Examples

### Basic Usage

```tcl
# Create a simple 2×2 max pooling layer
set pool [torch::maxpool2d 2]

# Create input tensor: 1 batch, 1 channel, 4×4 spatial
set input [torch::ones -shape {1 1 4 4} -dtype float32]

# Apply pooling
set output [torch::layer_forward $pool $input]
set shape [torch::tensor_shape $output]
puts "Output shape: $shape"  # Output: {1 1 2 2}
```

### Named Parameter Syntax

```tcl
# Using named parameters for clarity
set pool [torch::maxpool2d -kernelSize 2 -stride 2 -padding 0]

# Alternative parameter name
set pool [torch::maxpool2d -kernel_size 2 -stride 1 -padding 1]

# CamelCase alias
set pool [torch::maxPool2d -kernelSize 3 -stride 2]
```

### Different Pooling Configurations

```tcl
# Non-overlapping 2×2 pooling (most common)
set pool1 [torch::maxpool2d 2]

# Overlapping 3×3 pooling with stride 2
set pool2 [torch::maxpool2d 3 2]

# Large 4×4 pooling for aggressive downsampling
set pool3 [torch::maxpool2d 4 4]

# Pooling with padding to maintain certain dimensions
set pool4 [torch::maxpool2d 3 2 1]
```

### CNN Architecture Example

```tcl
# Create a simple CNN with alternating conv and pooling layers
set conv1 [torch::conv2d -inChannels 3 -outChannels 32 -kernelSize 3]
set pool1 [torch::maxpool2d 2]  # Reduces spatial size by half

set conv2 [torch::conv2d -inChannels 32 -outChannels 64 -kernelSize 3]
set pool2 [torch::maxpool2d 2]  # Reduces spatial size by half again

# Build sequential model
set model [torch::sequential]
torch::sequential_add $model $conv1
torch::sequential_add $model $pool1
torch::sequential_add $model $conv2
torch::sequential_add $model $pool2

# Process image: 3×224×224 → ... → 64×54×54 (after both pooling layers)
set image [torch::randn -shape {1 3 224 224}]
set features [torch::layer_forward $model $image]
```

### Different Input Sizes

```tcl
# Small feature maps
set small_input [torch::randn -shape {1 64 8 8}]
set pool [torch::maxpool2d 2]
set small_output [torch::layer_forward $pool $small_input]  # Shape: {1 64 4 4}

# Large feature maps
set large_input [torch::randn -shape {1 128 128 128}]
set aggressive_pool [torch::maxpool2d 4]
set large_output [torch::layer_forward $aggressive_pool $large_input]  # Shape: {1 128 32 32}
```

### Batch Processing

```tcl
# Multiple images in a batch
set batch_input [torch::randn -shape {8 3 32 32}]  # 8 images, 3 channels, 32×32
set pool [torch::maxpool2d 2]
set batch_output [torch::layer_forward $pool $batch_input]  # Shape: {8 3 16 16}
```

## Mathematical Properties

### Max Operation
For each pooling window, the output is:
```
output[i,j] = max(input[i×stride:(i×stride)+kernel_size, j×stride:(j×stride)+kernel_size])
```

### Translation Invariance
Max pooling provides local translation invariance - small shifts in the input may not change the output if the maximum values remain within their respective pooling windows.

### Information Loss
Max pooling discards spatial information but preserves the strongest activations, which often correspond to the most important features.

## Common Use Cases

### Image Classification
```tcl
# Typical CNN for image classification
set conv_pool_block {
    set conv [torch::conv2d -inChannels $in_ch -outChannels $out_ch -kernelSize 3 -padding 1]
    set relu [torch::relu]
    set pool [torch::maxpool2d 2]  # Standard 2×2 max pooling
    # Add to sequential model...
}
```

### Feature Extraction
```tcl
# Feature pyramid-like structure
set feature_extractor [torch::sequential]

# Fine features (less pooling)
set conv1 [torch::conv2d -inChannels 3 -outChannels 64 -kernelSize 3]
torch::sequential_add $feature_extractor $conv1

# Coarse features (more pooling)
set pool1 [torch::maxpool2d 2]
torch::sequential_add $feature_extractor $pool1

set conv2 [torch::conv2d -inChannels 64 -outChannels 128 -kernelSize 3]
torch::sequential_add $feature_extractor $conv2

set pool2 [torch::maxpool2d 2]
torch::sequential_add $feature_extractor $pool2
```

### Object Detection Backbone
```tcl
# ResNet-like backbone with pooling layers
set backbone_pool [torch::maxpool2d 3 2 1]  # 3×3 kernel, stride 2, padding 1

# Apply after initial convolution
set input [torch::randn -shape {1 64 112 112}]
set pooled [torch::layer_forward $backbone_pool $input]  # Shape: {1 64 56 56}
```

## Comparison with Other Pooling Types

### vs Average Pooling
- **Max pooling**: Preserves sharp features, provides some translation invariance
- **Average pooling**: Smooths features, better for global feature aggregation

### vs Adaptive Pooling
- **Fixed pooling**: Output size depends on input size and pooling parameters
- **Adaptive pooling**: Output size is fixed regardless of input size

### vs Strided Convolution
- **Max pooling**: No learnable parameters, simple maximum operation
- **Strided convolution**: Learnable parameters, more flexible downsampling

## Performance Considerations

### Memory Usage
Max pooling reduces memory requirements by decreasing spatial dimensions:
```tcl
# Memory reduction example
# Input: 1×512×32×32 = 524,288 elements
# After 2×2 pooling: 1×512×16×16 = 131,072 elements (4× reduction)
```

### Computational Efficiency
Max pooling is computationally efficient - only requires comparison operations, no arithmetic.

### GPU Acceleration
Max pooling operations are highly optimized on GPUs with efficient parallel implementations.

## Error Handling

### Invalid Kernel Size
```tcl
catch {torch::maxpool2d 0} result
puts $result  # Output: kernelSize must be > 0

catch {torch::maxpool2d -2} result
puts $result  # Output: kernelSize must be > 0
```

### Missing Parameters
```tcl
catch {torch::maxpool2d} result
puts $result  # Output: Usage error

catch {torch::maxpool2d -kernelSize} result
puts $result  # Output: Missing value for parameter
```

### Unknown Parameters
```tcl
catch {torch::maxpool2d -invalidParam 2} result
puts $result  # Output: Unknown parameter: -invalidParam
```

### Too Many Arguments
```tcl
catch {torch::maxpool2d 2 1 0 extra} result
puts $result  # Output: Invalid number of arguments
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set pool [torch::maxpool2d 2 2 0]
```

**New (Named Parameters):**
```tcl
set pool [torch::maxpool2d -kernelSize 2 -stride 2 -padding 0]
# or
set pool [torch::maxpool2d -kernel_size 2 -stride 2 -padding 0]
```

**New (CamelCase Alias):**
```tcl
set pool [torch::maxPool2d 2 2 0]
set pool [torch::maxPool2d -kernelSize 2 -stride 2 -padding 0]
```

### Backward Compatibility

All existing code continues to work:
```tcl
# These all work
set pool1 [torch::maxpool2d 2]
set pool2 [torch::maxpool2d 2 1]
set pool3 [torch::maxpool2d 2 1 0]
set pool4 [torch::maxpool2d -kernelSize 2]
set pool5 [torch::maxPool2d 2]
```

## Related Commands

### Layer Operations
- `torch::layer_forward` - Apply the pooling layer to input tensors
- `torch::sequential` - Create sequential models containing pooling layers
- `torch::layer_to` - Move pooling layers to different devices

### Other Pooling Types
- `torch::avgpool2d` - Average pooling (when available)
- `torch::maxpool1d` - 1D max pooling (when available)
- `torch::maxpool3d` - 3D max pooling (when available)
- `torch::adaptive_max_pool2d` - Adaptive max pooling

### Convolutional Layers
- `torch::conv2d` - 2D convolution layers (commonly used before pooling)
- `torch::conv1d` - 1D convolution layers
- `torch::conv3d` - 3D convolution layers

### Activation Functions
- `torch::relu` - ReLU activation (commonly used with pooling)
- `torch::leaky_relu` - Leaky ReLU activation
- `torch::sigmoid` - Sigmoid activation

## Notes

1. **Default Stride**: When stride is not specified, it defaults to the kernel size, creating non-overlapping pooling windows.

2. **Square Kernels Only**: This implementation uses square pooling kernels (kernel_size × kernel_size).

3. **Channel Independence**: Max pooling operates independently on each channel.

4. **Batch Support**: The layer works with batched inputs of shape [batch_size, channels, height, width].

5. **No Learnable Parameters**: Max pooling layers have no weights or biases to learn.

6. **Deterministic**: Given the same input, max pooling always produces the same output.

7. **Non-differentiable**: The max operation is technically non-differentiable at points where multiple values are equal, but gradients are typically assigned to one of the maximum elements during backpropagation.

## Best Practices

1. **Use 2×2 pooling with stride 2** for standard downsampling
2. **Apply pooling after activation functions** in typical CNN architectures
3. **Consider the output size** when designing network architectures
4. **Use padding carefully** - it can affect the effective receptive field
5. **Monitor feature map sizes** throughout your network to avoid excessive downsampling
6. **Test with different pooling configurations** to find optimal performance for your task 