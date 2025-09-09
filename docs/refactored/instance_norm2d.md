# torch::instance_norm2d

Applies 2D Instance Normalization to a tensor. Instance normalization normalizes across the spatial dimensions for each instance (sample) and channel independently, making it particularly effective for image processing, style transfer, and computer vision tasks.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::instance_norm2d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
torch::instance_norm2d -tensor tensor ?-epsilon epsilon? ?-momentum momentum? ?-affine affine? ?-trackRunningStats track_stats?
```

### Positional Parameters (Legacy)
```tcl
torch::instance_norm2d tensor ?eps? ?momentum? ?affine? ?track_running_stats?
```

### CamelCase Alias
```tcl
torch::instanceNorm2d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` or `-tensor` | tensor | Yes | - | Input tensor (4D: [N, C, H, W]) |
| `-eps` or `-epsilon` | float | No | 1e-5 | Small value for numerical stability |
| `-momentum` | float | No | 0.1 | Momentum for running statistics |
| `-affine` | bool | No | true | Whether to use learnable affine parameters |
| `-track_running_stats` or `-trackRunningStats` | bool | No | true | Whether to track running statistics |

## Returns

Returns a new tensor with the same shape as input, containing the instance-normalized values.

## Description

Instance normalization applies normalization over each spatial location independently for each sample and channel. For a 4D input tensor with shape [N, C, H, W]:

- **N**: Batch size
- **C**: Number of channels  
- **H**: Height (spatial dimension)
- **W**: Width (spatial dimension)

The normalization is computed as:
```
output = (input - mean) / sqrt(var + eps)
```

Where `mean` and `var` are computed per instance and channel across the spatial dimensions (H×W).

### Key Features
- **Per-instance normalization**: Each sample in the batch is normalized independently
- **Per-channel normalization**: Each channel is normalized independently
- **Spatial normalization**: Statistics computed across spatial dimensions (H×W)
- **Style transfer**: Particularly effective for neural style transfer
- **Real-time processing**: Suitable for inference and real-time applications

## Examples

### Basic Usage

```tcl
# Create a 4D tensor (batch=2, channels=3, height=4, width=5)
set tensor [torch::ones -shape {2 3 4 5}]

# Apply instance normalization (named parameters)
set result [torch::instance_norm2d -input $tensor]
puts [torch::tensor_shape $result]
;# Output: {2 3 4 5}

# Using positional syntax (legacy)
set result2 [torch::instance_norm2d $tensor]
puts [torch::tensor_shape $result2]
;# Output: {2 3 4 5}

# Using camelCase alias
set result3 [torch::instanceNorm2d -input $tensor]
puts [torch::tensor_shape $result3]
;# Output: {2 3 4 5}
```

### Image Processing

```tcl
# RGB images (batch=4, channels=3, height=32, width=32)
set rgb_images [torch::ones -shape {4 3 32 32}]
set norm_rgb [torch::instance_norm2d -input $rgb_images]
puts [torch::tensor_shape $norm_rgb]
;# Output: {4 3 32 32}

# Grayscale images (batch=8, channels=1, height=28, width=28)
set gray_images [torch::ones -shape {8 1 28 28}]
set norm_gray [torch::instance_norm2d -input $gray_images]
puts [torch::tensor_shape $norm_gray]
;# Output: {8 1 28 28}

# High-resolution images (batch=2, channels=3, height=224, width=224)
set hires_images [torch::ones -shape {2 3 224 224}]
set norm_hires [torch::instance_norm2d -input $hires_images]
puts [torch::tensor_shape $norm_hires]
;# Output: {2 3 224 224}
```

### Custom Parameters

```tcl
# Create input tensor
set tensor [torch::ones -shape {4 8 16 32}]

# Custom epsilon for numerical stability
set result [torch::instance_norm2d -input $tensor -eps 1e-4]

# Custom momentum for running statistics
set result2 [torch::instance_norm2d -input $tensor -momentum 0.2]

# Disable affine transformation
set result3 [torch::instance_norm2d -input $tensor -affine 0]

# Disable running statistics tracking
set result4 [torch::instance_norm2d -input $tensor -track_running_stats 0]
```

### Complete Configuration

```tcl
# Full parameter specification
set tensor [torch::ones -shape {2 4 8 16}]
set result [torch::instance_norm2d \
    -input $tensor \
    -eps 1e-4 \
    -momentum 0.15 \
    -affine 1 \
    -track_running_stats 1]
```

### Style Transfer Example

```tcl
# Typical usage in neural style transfer
set content_features [torch::ones -shape {1 512 32 32}]  ;# Content features
set style_features [torch::ones -shape {1 512 32 32}]    ;# Style features

# Normalize content features
set norm_content [torch::instance_norm2d -input $content_features -eps 1e-5]

# Normalize style features  
set norm_style [torch::instance_norm2d -input $style_features -eps 1e-5]

# Instance norm is particularly effective for style transfer
# as it removes instance-specific contrast information
puts "Content normalized: [torch::tensor_shape $norm_content]"
puts "Style normalized: [torch::tensor_shape $norm_style]"
```

## Parameter Aliases

The function supports multiple parameter names for flexibility:

- `-input` and `-tensor`: Both specify the input tensor
- `-eps` and `-epsilon`: Both specify the epsilon value
- `-track_running_stats` and `-trackRunningStats`: Both specify running statistics tracking

## Error Handling

The function provides clear error messages for invalid inputs:

```tcl
# Missing input tensor
catch {torch::instance_norm2d} msg
puts $msg
;# Output: Required parameters missing: input tensor required

# Invalid epsilon (must be positive)
set tensor [torch::ones -shape {2 3 4 5}]
catch {torch::instance_norm2d -input $tensor -eps -1.0} msg
puts $msg
;# Output: Invalid eps: must be positive number

# Invalid momentum (must be non-negative)
catch {torch::instance_norm2d -input $tensor -momentum -0.1} msg
puts $msg
;# Output: Invalid momentum: must be number >= 0

# Unknown parameter
catch {torch::instance_norm2d -input $tensor -unknown_param value} msg
puts $msg
;# Output: Unknown parameter: -unknown_param

# Unsupported tensor type
set int_tensor [torch::ones -shape {2 3 4 5} -dtype int64]
catch {torch::instance_norm2d $int_tensor} msg
puts $msg
;# Output: Error in instance_norm2d: "batch_norm" not implemented for 'Long'
```

## Technical Details

### Input Requirements
- **Shape**: Input must be a 4D tensor with shape [N, C, H, W]
- **N**: Batch size (number of samples)
- **C**: Number of channels
- **H**: Height (spatial dimension)
- **W**: Width (spatial dimension)

### Computation
1. **Per-instance/channel statistics**: Mean and variance computed independently for each (batch, channel) pair
2. **Spatial reduction**: Statistics computed across H×W dimensions
3. **Normalization**: `(x - mean) / sqrt(var + eps)`
4. **Affine transformation**: Optional learnable scale and shift parameters
5. **Running statistics**: Optional tracking for inference

### Performance Considerations
- **Memory**: O(N×C×H×W) for input tensor
- **Computation**: O(N×C×H×W) for normalization
- **Parallelization**: Highly parallelizable across batch and channel dimensions
- **Efficiency**: More efficient than batch norm for small batch sizes

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::instance_norm2d $tensor
torch::instance_norm2d $tensor $eps
torch::instance_norm2d $tensor $eps $momentum
torch::instance_norm2d $tensor $eps $momentum $affine $track_stats

# New named parameter syntax
torch::instance_norm2d -input $tensor
torch::instance_norm2d -input $tensor -eps $eps
torch::instance_norm2d -input $tensor -eps $eps -momentum $momentum
torch::instance_norm2d -input $tensor -eps $eps -momentum $momentum -affine $affine -track_running_stats $track_stats
```

### Advantages of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Partial specification**: Can specify only needed parameters
- **Error prevention**: Less likely to pass arguments in wrong order
- **Extensibility**: Easy to add new parameters without breaking existing code

## Related Functions

- `torch::instance_norm1d`: 1D instance normalization
- `torch::instance_norm3d`: 3D instance normalization
- `torch::batch_norm2d`: Batch normalization (different normalization strategy)
- `torch::layer_norm`: Layer normalization
- `torch::group_norm`: Group normalization

## Use Cases

### Neural Style Transfer
Instance normalization is the standard choice for style transfer networks:
```tcl
# Style transfer generator layer
set conv_output [torch::conv2d -input $input -weight $conv_weight]
set normalized [torch::instance_norm2d -input $conv_output -eps 1e-5]
set activated [torch::relu -input $normalized]
```

### Real-time Image Processing
Suitable for real-time applications due to independence from batch statistics:
```tcl
# Process single image in real-time
set single_image [torch::ones -shape {1 3 256 256}]
set processed [torch::instance_norm2d -input $single_image]
```

### Generative Adversarial Networks
Commonly used in GANs for image generation:
```tcl
# Generator network layer
set feature_maps [torch::ones -shape {4 64 32 32}]
set normalized [torch::instance_norm2d -input $feature_maps -eps 1e-5]
```

### Computer Vision Preprocessing
Normalize images before feeding to neural networks:
```tcl
# Preprocess images for vision model
set images [torch::ones -shape {8 3 224 224}]
set preprocessed [torch::instance_norm2d -input $images -eps 1e-5]
```

## Mathematical Background

Instance normalization computes statistics independently for each instance and channel:

- **Mean**: `μ_nc = (1/HW) Σ_hw x_nchw`
- **Variance**: `σ²_nc = (1/HW) Σ_hw (x_nchw - μ_nc)²`
- **Normalization**: `y_nchw = (x_nchw - μ_nc) / √(σ²_nc + ε)`

This differs from:
- **Batch Norm**: Statistics computed across batch dimension
- **Layer Norm**: Statistics computed across channel and spatial dimensions
- **Group Norm**: Statistics computed across channel groups

### Advantages over Batch Normalization
- **Batch size independence**: Works well with small or single batch sizes
- **Style transfer**: Removes instance-specific contrast information
- **Real-time inference**: No dependency on batch statistics
- **Stable training**: More stable for generative models

## See Also

- [torch::instance_norm1d](instance_norm1d.md) - 1D instance normalization
- [torch::instance_norm3d](instance_norm3d.md) - 3D instance normalization
- [torch::batch_norm2d](batch_norm2d.md) - Batch normalization
- [Normalization Guide](../guides/normalization.md) - Complete normalization comparison 