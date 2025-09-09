# torch::group_norm

## Overview
Creates a Group Normalization layer that applies normalization across channels by dividing them into groups. Group normalization is useful for small batch sizes where batch normalization may not work well, providing normalization that is independent of the batch size.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::group_norm -num_groups NUM_GROUPS -num_channels NUM_CHANNELS ?-eps EPSILON?
torch::group_norm -numGroups NUM_GROUPS -numChannels NUM_CHANNELS ?-eps EPSILON?
```

### Positional Syntax (Legacy)
```tcl
torch::group_norm NUM_GROUPS NUM_CHANNELS ?EPSILON?
```

### camelCase Alias
```tcl
torch::groupNorm -num_groups NUM_GROUPS -num_channels NUM_CHANNELS ?-eps EPSILON?
torch::groupNorm -numGroups NUM_GROUPS -numChannels NUM_CHANNELS ?-eps EPSILON?
```

## Parameters

### Required Parameters
- **`-num_groups`** or **`-numGroups`**: Number of groups to divide channels into
  - Type: Integer
  - Range: > 0
  - Must be a divisor of `num_channels`
  - Controls how channels are grouped for normalization

- **`-num_channels`** or **`-numChannels`**: Number of input channels
  - Type: Integer
  - Range: > 0
  - Must be divisible by `num_groups`
  - Total number of channels in the input tensor

### Optional Parameters
- **`-eps`**: Small value added to denominator for numerical stability
  - Type: Float
  - Default: `1e-5`
  - Range: > 0
  - Prevents division by zero during normalization

## Return Value
Returns a group normalization layer handle that can be used with `torch::layer_forward` or other layer operations.

## Examples

### Basic Group Normalization
```tcl
# Create a group norm layer with 2 groups for 4 channels
set gn [torch::group_norm -num_groups 2 -num_channels 4]
puts "Group norm layer created: $gn"

# Same using positional syntax
set gn_pos [torch::group_norm 2 4]
puts "Group norm layer (positional): $gn_pos"
```

### Group Normalization with Custom Epsilon
```tcl
# Create with custom epsilon value
set gn [torch::group_norm -numGroups 4 -numChannels 16 -eps 1e-6]
puts "Group norm with custom eps: $gn"

# Same using positional syntax
set gn_pos [torch::group_norm 4 16 1e-6]
puts "Group norm (positional with eps): $gn_pos"
```

### Using camelCase Alias
```tcl
# Using camelCase alias
set gn [torch::groupNorm -numGroups 8 -numChannels 32]
puts "Group norm (camelCase): $gn"

# Mixed parameter styles
set gn_mixed [torch::groupNorm -num_groups 2 -numChannels 6 -eps 1e-4]
puts "Group norm (mixed style): $gn_mixed"
```

### Different Group Configurations
```tcl
# Single group (equivalent to instance normalization)
set instance_norm [torch::group_norm -num_groups 1 -num_channels 16]

# Each channel is its own group (equivalent to layer normalization)
set layer_norm [torch::group_norm -num_groups 16 -num_channels 16]

# Typical group normalization (2-8 groups)
set typical_gn [torch::group_norm -num_groups 4 -num_channels 32]
```

### Parameter Variations
```tcl
# All equivalent ways to create the same layer
set gn1 [torch::group_norm -num_groups 2 -num_channels 4]
set gn2 [torch::group_norm -numGroups 2 -numChannels 4]
set gn3 [torch::groupNorm -num_groups 2 -num_channels 4]
set gn4 [torch::groupNorm -numGroups 2 -numChannels 4]
set gn5 [torch::group_norm 2 4]
```

## Group Normalization Explained

### How it Works
Group normalization divides channels into groups and normalizes within each group:
- Input shape: `(N, C, H, W)` where N=batch, C=channels, H=height, W=width
- Channels are divided into `num_groups` groups
- Each group has `C/num_groups` channels
- Normalization is applied independently to each group

### Mathematical Formula
For each group:
```
y = (x - μ) / √(σ² + ε)
```
Where:
- `μ` is the mean of the group
- `σ²` is the variance of the group
- `ε` is the epsilon value for numerical stability

### Group Configurations
```tcl
# Example with 8 channels and different group counts
set channels 8

# 1 group - Instance normalization (per-channel normalization)
set instance [torch::group_norm -num_groups 1 -num_channels $channels]

# 2 groups - 4 channels per group
set group2 [torch::group_norm -num_groups 2 -num_channels $channels]

# 4 groups - 2 channels per group  
set group4 [torch::group_norm -num_groups 4 -num_channels $channels]

# 8 groups - 1 channel per group (layer normalization)
set layer [torch::group_norm -num_groups 8 -num_channels $channels]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::group_norm} msg
puts $msg  ;# "numGroups and numChannels must be > 0"

# Invalid number of groups
catch {torch::group_norm -numGroups 0 -numChannels 4} msg
puts $msg  ;# "numGroups and numChannels must be > 0"

# Invalid number of channels
catch {torch::group_norm -numGroups 2 -numChannels 0} msg
puts $msg  ;# "numGroups and numChannels must be > 0"

# Non-integer parameters
catch {torch::group_norm -numGroups 2.5 -numChannels 4} msg
puts $msg  ;# "Invalid numGroups value"

# Invalid epsilon
catch {torch::group_norm -numGroups 2 -numChannels 4 -eps "not_a_number"} msg
puts $msg  ;# "Invalid eps value"

# Unknown parameter
catch {torch::group_norm -numGroups 2 -numChannels 4 -unknown_param value} msg
puts $msg  ;# "Unknown parameter: -unknown_param"
```

## Common Use Cases

### Convolutional Neural Networks
```tcl
# After convolutional layers
set conv [torch::conv2d -in_channels 3 -out_channels 16 -kernel_size 3]
set gn [torch::group_norm -num_groups 2 -num_channels 16]
# Apply: conv -> group_norm -> activation
```

### Small Batch Training
```tcl
# Group norm works well with small batches (even batch_size=1)
set gn [torch::group_norm -num_groups 4 -num_channels 32]
# Unlike batch norm, group norm is independent of batch size
```

### Residual Networks
```tcl
# In residual blocks
set conv1 [torch::conv2d -in_channels 64 -out_channels 64 -kernel_size 3 -padding 1]
set gn1 [torch::group_norm -num_groups 8 -num_channels 64]
set conv2 [torch::conv2d -in_channels 64 -out_channels 64 -kernel_size 3 -padding 1]
set gn2 [torch::group_norm -num_groups 8 -num_channels 64]
```

## Practical Guidelines

### Choosing Number of Groups
- **Small groups (2-8)**: Good general choice, balances normalization and independence
- **Single group**: Equivalent to instance normalization
- **Groups = Channels**: Equivalent to layer normalization
- **Rule of thumb**: 4-8 groups for most cases, ensure `channels % groups == 0`

### Epsilon Selection
- **Default (1e-5)**: Works for most cases
- **Smaller (1e-6, 1e-8)**: For higher precision or when values are small
- **Larger (1e-4, 1e-3)**: For numerical stability with specific data ranges

### Channel Divisibility
```tcl
# Ensure channels are divisible by groups
set channels 32
set valid_groups [list 1 2 4 8 16 32]  ;# All divisors of 32

# This will work
set gn [torch::group_norm -num_groups 8 -num_channels 32]  ;# 32/8 = 4 channels per group

# This might cause issues if not divisible
# torch::group_norm -num_groups 5 -num_channels 32  ;# 32/5 = 6.4 (not integer)
```

## Performance Considerations

- **Memory**: Group norm has minimal memory overhead
- **Computation**: Slightly more expensive than batch norm, less than layer norm
- **Gradient Flow**: Provides good gradient flow for deep networks
- **Batch Independence**: Works with any batch size, including batch_size=1

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set gn [torch::group_norm 2 4]
set gn [torch::group_norm 4 16 1e-6]

# New named syntax
set gn [torch::group_norm -num_groups 2 -num_channels 4]
set gn [torch::group_norm -numGroups 4 -numChannels 16 -eps 1e-6]
```

### Parameter Mapping

| Positional | Named | Alternative |
|------------|-------|-------------|
| `num_groups` | `-num_groups` | `-numGroups` |
| `num_channels` | `-num_channels` | `-numChannels` |
| `eps` | `-eps` | N/A |

## See Also

- [torch::batchnorm2d](batchnorm2d.md) - Batch normalization
- [torch::layer_norm](layer_norm.md) - Layer normalization
- [torch::instance_norm](instance_norm.md) - Instance normalization
- [torch::conv2d](conv2d.md) - Convolutional layers
- [torch::layer_forward](layer_forward.md) - Forward pass through layers

## Implementation Notes

- Based on PyTorch's `torch.nn.GroupNorm`
- Supports all standard tensor data types (float32, float64, etc.)
- Maintains learnable scale and shift parameters (affine transformation)
- Efficient implementation using LibTorch's native group normalization
- Thread-safe for concurrent usage
- Automatic gradient support for training

## Mathematical Background

Group normalization was introduced to address limitations of batch normalization with small batch sizes. It provides:

1. **Batch Independence**: Normalization doesn't depend on batch statistics
2. **Stable Training**: Works well with small or variable batch sizes
3. **Good Generalization**: Often performs better than batch norm in certain scenarios
4. **Flexibility**: Number of groups can be tuned based on the problem

The key insight is that normalization along the batch dimension (batch norm) may not always be optimal, especially with small batches or when batch statistics are not representative of the true distribution.
