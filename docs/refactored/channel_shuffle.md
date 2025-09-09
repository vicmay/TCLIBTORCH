# torch::channel_shuffle

## Overview

Performs channel shuffle operation on input tensors. This operation divides channels into groups and rearranges them, commonly used in ShuffleNet architectures for efficient neural networks.

**Status**: ✅ **REFACTORED** - Supports both snake_case and camelCase syntax with named parameters

## Syntax

### Current Syntax (Recommended)
```tcl
# Named parameters (recommended)
torch::channel_shuffle -input tensor_handle -groups num_groups
torch::channelShuffle -input tensor_handle -groups num_groups

# Alternative parameter names
torch::channel_shuffle -tensor tensor_handle -groups num_groups
```

### Legacy Syntax (Backward Compatible)
```tcl
# Positional parameters (still supported)
torch::channel_shuffle tensor_handle num_groups
torch::channelShuffle tensor_handle num_groups
```

## Parameters

### Named Parameters
- **`-input tensor_handle`** (required): Input tensor handle
  - Alternative: **`-tensor tensor_handle`**
  - Must be a valid tensor with at least 4 dimensions (N, C, H, W)
  
- **`-groups num_groups`** (required): Number of groups for channel shuffling
  - Type: Integer
  - Must be > 0 and should divide the number of input channels evenly
  - Default: None (required parameter)

### Legacy Positional Parameters
1. **`tensor_handle`**: Input tensor handle
2. **`num_groups`**: Number of groups for channel shuffling

## Return Value

Returns a handle to the channel-shuffled tensor with the same shape as the input tensor.

## Examples

### Basic Usage
```tcl
# Create a tensor with 6 channels
set input [torch::zeros {1 6 4 4}]

# Named parameter syntax (recommended)
set result [torch::channel_shuffle -input $input -groups 2]

# Legacy syntax (still works)
set result [torch::channel_shuffle $input 2]

# CamelCase alias
set result [torch::channelShuffle -input $input -groups 2]
```

### Advanced Examples
```tcl
# Multiple batch and larger channel count
set input [torch::zeros {2 12 8 8}]
set result [torch::channel_shuffle -input $input -groups 3]

# Using tensor alias parameter
set input [torch::zeros {1 16 4 4}]
set result [torch::channel_shuffle -tensor $input -groups 8]

# Parameter order flexibility
set result [torch::channel_shuffle -groups 4 -input $input]
```

### Integration with Neural Networks
```tcl
# Example in a ShuffleNet-style block
proc shufflenet_unit {input channels groups} {
    # 1x1 convolution
    set conv1 [torch::conv2d $input $weight1 $bias1 1 0]
    
    # Channel shuffle
    set shuffled [torch::channel_shuffle -input $conv1 -groups $groups]
    
    # 3x3 depthwise convolution
    set conv2 [torch::conv2d $shuffled $weight2 $bias2 1 1 1 $channels]
    
    return $conv2
}
```

## Mathematical Description

Channel shuffle operation works as follows:

1. **Input tensor shape**: `(N, C, H, W)`
2. **Reshape**: `(N, groups, C/groups, H, W)`
3. **Transpose**: `(N, C/groups, groups, H, W)`
4. **Reshape back**: `(N, C, H, W)`

The operation rearranges channels to enable information flow between different channel groups in efficient neural network architectures.

## Error Handling

### Common Errors
```tcl
# Missing required parameters
torch::channel_shuffle
# Error: Usage: torch::channel_shuffle input groups | torch::channelShuffle -input tensor -groups num_groups

# Invalid tensor handle
torch::channel_shuffle invalid_handle 2
# Error: Invalid input tensor

# Invalid groups value
torch::channel_shuffle $tensor 0
# Error: Required parameters missing: input tensor and groups (> 0) required

# Unknown parameter
torch::channel_shuffle -input $tensor -invalid_param 2
# Error: Unknown parameter: -invalid_param. Valid parameters are: -input, -tensor, -groups

# Missing parameter value
torch::channel_shuffle -input $tensor -groups
# Error: Missing value for parameter
```

## Performance Notes

- The new named parameter syntax has similar performance to the legacy syntax
- Channel shuffle is implemented using tensor view and transpose operations
- Memory efficient - no data copying required
- CUDA tensors are supported

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Before (legacy - still works)
set result [torch::channel_shuffle $input 2]

# After (recommended)
set result [torch::channel_shuffle -input $input -groups 2]

# CamelCase alternative
set result [torch::channelShuffle -input $input -groups 2]
```

### Parameter Mapping
| Legacy Position | Named Parameter | Alternative |
|----------------|-----------------|-------------|
| 1st argument   | `-input`        | `-tensor`   |
| 2nd argument   | `-groups`       | N/A         |

## Use Cases

1. **ShuffleNet architectures**: Efficient channel communication in grouped convolutions
2. **Mobile neural networks**: Reducing computational cost while maintaining accuracy
3. **Feature mixing**: Enabling information exchange between channel groups
4. **Custom layer implementations**: Building efficient neural network blocks

## Implementation Details

- **Backward Compatible**: Legacy positional syntax fully supported
- **Input Validation**: Comprehensive parameter and tensor validation
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Memory Efficient**: Uses tensor views and transpose operations
- **Thread Safe**: Safe for concurrent execution

## Related Commands

- [`torch::conv2d`](conv2d.md) - 2D convolution operations
- [`torch::group_norm`](group_norm.md) - Group normalization
- [`torch::tensor_reshape`](tensor_reshape.md) - Tensor reshaping
- [`torch::tensor_permute`](tensor_permute.md) - Tensor dimension permutation

## Version History

- **v1.0**: Original implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained

---

**Note**: This command is part of the LibTorch TCL Extension refactoring initiative, providing modern, user-friendly APIs while maintaining full backward compatibility. 