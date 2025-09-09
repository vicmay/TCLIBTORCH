# torch::instance_norm1d

Applies 1D Instance Normalization to a tensor. Instance normalization normalizes across the spatial dimensions for each instance (sample) and channel independently, which is particularly useful for style transfer and other generative tasks.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::instance_norm1d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
torch::instance_norm1d -tensor tensor ?-epsilon epsilon? ?-momentum momentum? ?-affine affine? ?-trackRunningStats track_stats?
```

### Positional Parameters (Legacy)
```tcl
torch::instance_norm1d tensor ?eps? ?momentum? ?affine? ?track_running_stats?
```

### CamelCase Alias
```tcl
torch::instanceNorm1d -input tensor ?-eps epsilon? ?-momentum momentum? ?-affine affine? ?-track_running_stats track_stats?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` or `-tensor` | tensor | Yes | - | Input tensor (3D: [N, C, L]) |
| `-eps` or `-epsilon` | float | No | 1e-5 | Small value for numerical stability |
| `-momentum` | float | No | 0.1 | Momentum for running statistics |
| `-affine` | bool | No | true | Whether to use learnable affine parameters |
| `-track_running_stats` or `-trackRunningStats` | bool | No | true | Whether to track running statistics |

## Returns

Returns a new tensor with the same shape as input, containing the instance-normalized values.

## Description

Instance normalization applies normalization over each spatial location independently for each sample and channel. For a 3D input tensor with shape [N, C, L]:

- **N**: Batch size
- **C**: Number of channels  
- **L**: Length (spatial dimension)

The normalization is computed as:
```
output = (input - mean) / sqrt(var + eps)
```

Where `mean` and `var` are computed per instance and channel across the spatial dimension.

### Key Features
- **Per-instance normalization**: Each sample in the batch is normalized independently
- **Per-channel normalization**: Each channel is normalized independently
- **Spatial normalization**: Statistics computed across spatial dimensions
- **Style transfer**: Particularly effective for style transfer tasks
- **Real-time processing**: Suitable for inference and style transfer

## Examples

### Basic Usage

```tcl
# Create a 3D tensor (batch=2, channels=3, length=4)
set tensor [torch::ones -shape {2 3 4}]

# Apply instance normalization (named parameters)
set result [torch::instance_norm1d -input $tensor]
puts [torch::tensor_shape $result]
;# Output: {2 3 4}

# Using positional syntax (legacy)
set result2 [torch::instance_norm1d $tensor]
puts [torch::tensor_shape $result2]
;# Output: {2 3 4}

# Using camelCase alias
set result3 [torch::instanceNorm1d -input $tensor]
puts [torch::tensor_shape $result3]
;# Output: {2 3 4}
```

### Custom Parameters

```tcl
# Create input tensor
set tensor [torch::ones -shape {4 8 16}]

# Custom epsilon for numerical stability
set result [torch::instance_norm1d -input $tensor -eps 1e-4]

# Custom momentum for running statistics
set result2 [torch::instance_norm1d -input $tensor -momentum 0.2]

# Disable affine transformation
set result3 [torch::instance_norm1d -input $tensor -affine 0]

# Disable running statistics tracking
set result4 [torch::instance_norm1d -input $tensor -track_running_stats 0]
```

### Complete Configuration

```tcl
# Full parameter specification
set tensor [torch::ones -shape {2 4 8}]
set result [torch::instance_norm1d \
    -input $tensor \
    -eps 1e-4 \
    -momentum 0.15 \
    -affine 1 \
    -track_running_stats 1]
```

### Style Transfer Example

```tcl
# Typical usage in style transfer
set content_features [torch::ones -shape {1 512 64}]  ;# Content features
set style_features [torch::ones -shape {1 512 64}]    ;# Style features

# Normalize content features
set norm_content [torch::instance_norm1d -input $content_features -eps 1e-5]

# Normalize style features  
set norm_style [torch::instance_norm1d -input $style_features -eps 1e-5]
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
catch {torch::instance_norm1d} msg
puts $msg
;# Output: Required parameters missing: input tensor required

# Invalid epsilon (must be positive)
set tensor [torch::ones -shape {2 3 4}]
catch {torch::instance_norm1d -input $tensor -eps -1.0} msg
puts $msg
;# Output: Invalid eps: must be positive number

# Invalid momentum (must be non-negative)
catch {torch::instance_norm1d -input $tensor -momentum -0.1} msg
puts $msg
;# Output: Invalid momentum: must be number >= 0

# Unknown parameter
catch {torch::instance_norm1d -input $tensor -unknown_param value} msg
puts $msg
;# Output: Unknown parameter: -unknown_param
```

## Technical Details

### Input Requirements
- **Shape**: Input must be a 3D tensor with shape [N, C, L]
- **N**: Batch size (number of samples)
- **C**: Number of channels
- **L**: Spatial length dimension

### Computation
1. **Per-instance/channel statistics**: Mean and variance computed independently for each (batch, channel) pair
2. **Normalization**: `(x - mean) / sqrt(var + eps)`
3. **Affine transformation**: Optional learnable scale and shift parameters
4. **Running statistics**: Optional tracking for inference

### Performance Considerations
- **Memory**: O(N×C×L) for input tensor
- **Computation**: O(N×C×L) for normalization
- **Parallelization**: Highly parallelizable across batch and channel dimensions

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::instance_norm1d $tensor
torch::instance_norm1d $tensor $eps
torch::instance_norm1d $tensor $eps $momentum
torch::instance_norm1d $tensor $eps $momentum $affine $track_stats

# New named parameter syntax
torch::instance_norm1d -input $tensor
torch::instance_norm1d -input $tensor -eps $eps
torch::instance_norm1d -input $tensor -eps $eps -momentum $momentum
torch::instance_norm1d -input $tensor -eps $eps -momentum $momentum -affine $affine -track_running_stats $track_stats
```

### Advantages of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Partial specification**: Can specify only needed parameters
- **Error prevention**: Less likely to pass arguments in wrong order
- **Extensibility**: Easy to add new parameters without breaking existing code

## Related Functions

- `torch::instance_norm2d`: 2D instance normalization
- `torch::instance_norm3d`: 3D instance normalization
- `torch::batch_norm1d`: Batch normalization (different normalization strategy)
- `torch::layer_norm`: Layer normalization
- `torch::group_norm`: Group normalization

## Use Cases

### Style Transfer
Instance normalization is particularly effective for neural style transfer:
```tcl
# Normalize feature maps in style transfer network
set features [torch::conv1d -input $input -weight $conv_weight]
set normalized [torch::instance_norm1d -input $features -eps 1e-5]
```

### Real-time Processing
Suitable for real-time applications due to independence from batch statistics:
```tcl
# Process single sample in real-time
set single_sample [torch::ones -shape {1 64 128}]
set processed [torch::instance_norm1d -input $single_sample]
```

### Generative Models
Commonly used in generative adversarial networks and autoencoders:
```tcl
# Generator network layer
set generated [torch::instance_norm1d -input $conv_output -eps 1e-5]
```

## Mathematical Background

Instance normalization computes statistics independently for each instance and channel:

- **Mean**: `μ_nc = (1/L) Σ_l x_ncl`
- **Variance**: `σ²_nc = (1/L) Σ_l (x_ncl - μ_nc)²`
- **Normalization**: `y_ncl = (x_ncl - μ_nc) / √(σ²_nc + ε)`

This differs from:
- **Batch Norm**: Statistics computed across batch dimension
- **Layer Norm**: Statistics computed across channel and spatial dimensions
- **Group Norm**: Statistics computed across channel groups

## See Also

- [torch::instance_norm2d](instance_norm2d.md) - 2D instance normalization
- [torch::instance_norm3d](instance_norm3d.md) - 3D instance normalization
- [torch::batch_norm1d](batch_norm1d.md) - Batch normalization
- [Normalization Guide](../guides/normalization.md) - Complete normalization comparison 