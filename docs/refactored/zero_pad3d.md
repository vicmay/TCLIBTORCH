# torch::zero_pad3d / torch::zeroPad3d

Applies zero padding to a 3D tensor along all three dimensions.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::zero_pad3d tensor padding
```

### Named Parameter Syntax (Recommended)
```tcl
torch::zero_pad3d -input tensor -padding padding_list
torch::zero_pad3d -tensor tensor -pad padding_list
```

### CamelCase Alias
```tcl
torch::zeroPad3d tensor padding
torch::zeroPad3d -input tensor -padding padding_list
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| **tensor** / **-input** / **-tensor** | tensor | Input 3D tensor to be padded |
| **padding** / **-padding** / **-pad** | list | List of 6 integers: `{left right top bottom front back}` |

## Padding Format

The padding list must contain exactly 6 integers in the following order:
- `left`: Padding on the left side of the width dimension
- `right`: Padding on the right side of the width dimension  
- `top`: Padding on the top side of the height dimension
- `bottom`: Padding on the bottom side of the height dimension
- `front`: Padding on the front side of the depth dimension
- `back`: Padding on the back side of the depth dimension

## Return Value

Returns a new tensor with zero padding applied. The output shape is calculated as:
- Output width = input_width + left + right
- Output height = input_height + top + bottom  
- Output depth = input_depth + front + back

## Examples

### Basic Usage
```tcl
# Create a 2x2x2 tensor
set tensor [torch::zeros {2 2 2}]

# Positional syntax - add 1 padding on all sides
set result1 [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
puts [torch::tensor_shape $result1]  ;# Output: 4 4 4

# Named parameter syntax
set result2 [torch::zero_pad3d -input $tensor -padding {1 1 1 1 1 1}]
puts [torch::tensor_shape $result2]  ;# Output: 4 4 4

# CamelCase syntax
set result3 [torch::zeroPad3d $tensor {1 1 1 1 1 1}]
puts [torch::tensor_shape $result3]  ;# Output: 4 4 4
```

### Asymmetric Padding
```tcl
# Apply different padding to each side
set tensor [torch::zeros {2 2 2}]
set result [torch::zero_pad3d $tensor {3 1 2 0 1 2}]
puts [torch::tensor_shape $result]  ;# Output: 5 4 6

# Using named parameters for clarity
set result [torch::zero_pad3d -input $tensor -padding {3 1 2 0 1 2}]
# Width: 2 + 3 + 1 = 6 → ERROR: This should be 2 + 3 + 1 = 6
# Actually: Width: 2 + 3 + 1 = 6, Height: 2 + 2 + 0 = 4, Depth: 2 + 1 + 2 = 5
# But the actual result is {5 4 6}, so the dimensions are ordered differently

# The actual dimension mapping is:
# Depth: 2 + 1 + 2 = 5 (front + back)
# Height: 2 + 2 + 0 = 4 (top + bottom)  
# Width: 2 + 3 + 1 = 6 (left + right)
```

### Zero Padding (Identity Operation)
```tcl
set tensor [torch::zeros {3 3 3}]
set result [torch::zero_pad3d $tensor {0 0 0 0 0 0}]
puts [torch::tensor_shape $result]  ;# Output: 3 3 3
```

### Large Padding Values
```tcl
set tensor [torch::zeros {1 1 1}]
set result [torch::zero_pad3d $tensor {10 10 5 5 3 3}]
puts [torch::tensor_shape $result]  ;# Output: 7 11 21
```

### Single Element Tensors
```tcl
# Single element tensors work perfectly fine
set tensor [torch::zeros {1 1 1}]
set result [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
puts [torch::tensor_shape $result]  ;# Output: 3 3 3
```

## Error Handling

The command validates input parameters and provides clear error messages:

```tcl
# Missing arguments
catch {torch::zero_pad3d} error
puts $error  ;# Usage: torch::zero_pad3d tensor padding | torch::zeroPad3d -input tensor -padding {values}

# Wrong number of padding values
catch {torch::zero_pad3d $tensor {1 2 3}} error
puts $error  ;# Padding must be a list of 6 values for 3D

# Unknown parameter
catch {torch::zero_pad3d -unknown $tensor -padding {1 1 1 1 1 1}} error
puts $error  ;# Unknown parameter: -unknown. Valid parameters are: -input, -tensor, -padding, -pad

# Missing value for parameter
catch {torch::zero_pad3d -input $tensor -padding} error  
puts $error  ;# Missing value for parameter
```

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::zero_pad3d $tensor {2 2 1 1 0 3}]
```

**After (Named Parameters):**
```tcl
set result [torch::zero_pad3d -input $tensor -padding {2 2 1 1 0 3}]
# or
set result [torch::zero_pad3d -tensor $tensor -pad {2 2 1 1 0 3}]
```

### From snake_case to camelCase

**Before:**
```tcl
set result [torch::zero_pad3d $tensor {1 1 1 1 1 1}]
```

**After:**
```tcl
set result [torch::zeroPad3d $tensor {1 1 1 1 1 1}]
```

## Performance Notes

- All three syntax variations have identical performance
- The operation is efficient and implemented using PyTorch's optimized `constant_pad_nd` function
- Memory usage scales with the output tensor size

## See Also

- [torch::zero_pad1d](zero_pad1d.md) - 1D zero padding
- [torch::zero_pad2d](zero_pad2d.md) - 2D zero padding  
- [torch::constant_pad3d](constant_pad3d.md) - 3D constant padding with custom values
- [torch::reflection_pad3d](reflection_pad3d.md) - 3D reflection padding
- [torch::replication_pad3d](replication_pad3d.md) - 3D replication padding 