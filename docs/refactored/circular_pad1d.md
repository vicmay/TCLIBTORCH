# torch::circular_pad1d

## Overview

Applies circular padding to tensors along the last dimension. Circular padding wraps values around, so the values from the opposite ends of the tensor are used to pad each side. This is useful for periodic data where the ends naturally connect.

**Status**: ✅ **REFACTORED** - Supports both snake_case and camelCase syntax with named parameters

## Syntax

### Current Syntax (Recommended)
```tcl
# Named parameters (recommended)
torch::circular_pad1d -input tensor_handle -padding {pad_left pad_right}
torch::circularPad1d -input tensor_handle -padding {pad_left pad_right}

# Alternative parameter names
torch::circular_pad1d -tensor tensor_handle -pad {pad_left pad_right}
```

### Legacy Syntax (Backward Compatible)
```tcl
# Positional parameters (still supported)
torch::circular_pad1d tensor_handle {pad_left pad_right}
torch::circularPad1d tensor_handle {pad_left pad_right}
```

## Parameters

### Named Parameters
- **`-input tensor_handle`** (required): Input tensor to pad
  - Alternative: **`-tensor tensor_handle`**
  - Must be a valid tensor handle with at least 2 dimensions
  - For 1D-style padding, use shape `{1, N}` where N is the sequence length
  
- **`-padding {pad_left pad_right}`** (required): Padding specification
  - Alternative: **`-pad {pad_left pad_right}`**
  - Type: List of 2 integers
  - `pad_left`: Number of elements to pad on the left side
  - `pad_right`: Number of elements to pad on the right side
  - Values wrap around from the opposite end (circular behavior)

### Legacy Positional Parameters
1. **`tensor_handle`**: Input tensor to pad
2. **`{pad_left pad_right}`**: List of 2 integers specifying padding amounts

## Return Value

Returns a handle to a new tensor with circular padding applied. The output tensor will have the same shape as the input except for the last dimension, which will be increased by `pad_left + pad_right`.

## Examples

### Basic Usage
```tcl
# Create a 2D tensor (batch_size=1, sequence_length=4)
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set tensor [torch::tensor_reshape $tensor {1 4}]

# Named parameter syntax (recommended)
set padded [torch::circular_pad1d -input $tensor -padding {2 3}]

# Legacy syntax (still works)
set padded [torch::circular_pad1d $tensor {2 3}]

# CamelCase alias
set padded [torch::circularPad1d -input $tensor -padding {2 3}]
```

### Advanced Examples
```tcl
# Symmetric padding
set symmetric [torch::circular_pad1d -input $tensor -padding {2 2}]

# Asymmetric padding
set asymmetric [torch::circular_pad1d -input $tensor -padding {1 3}]

# Zero padding (no padding, but still circular semantics)
set no_pad [torch::circular_pad1d -input $tensor -padding {0 0}]

# Parameter order flexibility
set flexible [torch::circular_pad1d -padding {1 1} -input $tensor]

# Alternative parameter names
set alternative [torch::circular_pad1d -tensor $tensor -pad {1 2}]
```

### Batch Processing
```tcl
# Process multiple sequences simultaneously
set batch_tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
set batch_tensor [torch::tensor_reshape $batch_tensor {2 4}]  # 2 sequences of length 4

set batch_padded [torch::circular_pad1d -input $batch_tensor -padding {1 2}]
# Result shape: {2, 7} (each sequence padded from 4 to 7 elements)
```

### Time Series Padding
```tcl
proc pad_time_series {data pad_amount} {
    # Ensure data is 2D for circular padding
    if {[llength [torch::tensor_shape $data]] == 1} {
        set data [torch::tensor_reshape $data {1 [lindex [torch::tensor_shape $data] 0]}]
    }
    
    # Apply symmetric circular padding
    return [torch::circular_pad1d -input $data -padding [list $pad_amount $pad_amount]]
}

# Example usage
set time_series [torch::tensor_create {10.0 20.0 30.0 40.0 50.0} float32]
set time_series [torch::tensor_reshape $time_series {1 5}]
set padded_series [pad_time_series $time_series 2]
```

## Mathematical Description

Circular padding works by wrapping values from the opposite ends of the tensor:

**For input tensor [a, b, c, d] with padding {2, 3}:**
- Left padding (2): Takes last 2 elements → [c, d]  
- Right padding (3): Takes first 3 elements → [a, b, c]
- **Result: [c, d, a, b, c, d, a, b, c]**

The circular nature means:
- Left padding uses elements from the **right end** of the tensor
- Right padding uses elements from the **left end** of the tensor
- This creates a seamless periodic extension

## Tensor Shape Requirements

- **Input**: Tensor with at least 2 dimensions (PyTorch limitation)
- **For 1D-style data**: Reshape to `{1, N}` format
- **Batch processing**: Use `{batch_size, sequence_length}` format
- **Output**: Same shape as input except last dimension grows by `pad_left + pad_right`

## Error Handling

### Common Errors
```tcl
# Missing required parameters
torch::circular_pad1d
# Error: Usage: torch::circular_pad1d tensor padding | torch::circularPad1d -input tensor -padding {values}

# Invalid tensor handle
torch::circular_pad1d invalid_tensor {1 2}
# Error: Invalid tensor name: invalid_tensor

# Unknown parameter
torch::circular_pad1d -input $tensor -padding {1 2} -invalid_param value
# Error: Unknown parameter: -invalid_param. Valid parameters are: -input, -tensor, -padding, -pad

# Missing parameter value
torch::circular_pad1d -input $tensor -padding
# Error: Missing value for parameter

# Wrong number of padding values
torch::circular_pad1d -input $tensor -padding {1}
# Error: Padding must be a list of 2 values for 1D

torch::circular_pad1d -input $tensor -padding {1 2 3}
# Error: Padding must be a list of 2 values for 1D

# Invalid padding values
torch::circular_pad1d -input $tensor -padding {invalid 2}
# Error: expected integer but got "invalid"
```

## Performance Notes

- **Equivalent Performance**: Named parameter syntax has the same performance as legacy syntax
- **Memory Efficient**: Creates new tensor with minimal memory overhead
- **GPU Support**: Works with CUDA tensors for GPU acceleration
- **Batch Optimized**: Efficient processing of multiple sequences simultaneously

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Before (legacy - still works)
set padded [torch::circular_pad1d $tensor {2 3}]

# After (recommended)
set padded [torch::circular_pad1d -input $tensor -padding {2 3}]

# CamelCase alternative
set padded [torch::circularPad1d -input $tensor -padding {2 3}]
```

### Parameter Mapping
| Legacy Position | Named Parameter | Alternative |
|----------------|-----------------|-------------|
| 1st argument   | `-input`        | `-tensor`   |
| 2nd argument   | `-padding`      | `-pad`      |

## Use Cases

1. **Time Series Processing**: Padding periodic data like seasonal patterns
2. **Signal Processing**: Extending signals while maintaining periodicity
3. **Sequence Modeling**: Preparing sequential data for neural networks
4. **Computer Vision**: Padding feature maps with circular boundary conditions
5. **Audio Processing**: Extending audio signals for convolution operations
6. **Data Augmentation**: Creating synthetic extensions of periodic datasets

## Implementation Details

- **Backward Compatible**: Legacy positional syntax fully supported
- **Input Validation**: Comprehensive parameter and tensor validation
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Flexibility**: Multiple parameter names and order independence
- **Thread Safe**: Safe for concurrent execution
- **Memory Safe**: Proper tensor memory management

## Related Commands

- [`torch::circular_pad2d`](circular_pad2d.md) - 2D circular padding
- [`torch::circular_pad3d`](circular_pad3d.md) - 3D circular padding  
- [`torch::constant_pad1d`](constant_pad1d.md) - Constant value padding
- [`torch::reflection_pad1d`](reflection_pad1d.md) - Reflection padding
- [`torch::replication_pad1d`](replication_pad1d.md) - Replication padding

## Comparison with Other Padding Types

| Padding Type | Behavior | Use Case |
|-------------|----------|----------|
| **Circular** | Wraps from opposite end | Periodic data |
| **Constant** | Fills with constant value | Zero-padding, value padding |
| **Reflection** | Mirrors values at boundary | Smooth boundaries |
| **Replication** | Repeats edge values | Extending boundaries |

## Version History

- **v1.0**: Original implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained

---

**Note**: This command is part of the LibTorch TCL Extension refactoring initiative, providing modern, user-friendly APIs while maintaining full backward compatibility. 