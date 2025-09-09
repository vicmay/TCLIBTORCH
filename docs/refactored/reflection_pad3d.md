# torch::reflection_pad3d / torch::reflectionPad3d

## Description
Pads a 5D tensor using reflection of the border elements. This operation extends the tensor by reflecting its values across each boundary, which is particularly useful for tasks like 3D image processing, volumetric data augmentation, and 3D convolutional neural networks where edge artifacts need to be minimized.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::reflection_pad3d tensor padding
```

### New Syntax (Named Parameters)
```tcl
torch::reflection_pad3d -input tensor -padding {left right top bottom front back}
torch::reflection_pad3d -tensor tensor -pad {left right top bottom front back}
```

### CamelCase Alias
```tcl
torch::reflectionPad3d tensor padding
torch::reflectionPad3d -input tensor -padding {left right top bottom front back}
```

## Parameters

### Input Tensor Requirements
- Must be a 5D tensor with shape (batch_size, channels, depth, height, width)
- Any numeric data type is supported (float32, float64, etc.)

### Padding List Format
- Must be a list of 6 integers: {left right top bottom front back}
- All values must be non-negative
- Values represent the number of elements to pad on each side
- Order matters: left and right padding for the last dimension, top and bottom for the second-to-last dimension, front and back for the third-to-last dimension

### Parameter Aliases
- `-input` or `-tensor`: The input tensor to pad
- `-padding` or `-pad`: The padding values list

## Return Value
Returns a new tensor with the same number of dimensions as the input tensor, but with sizes increased according to the padding values.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a 5D tensor (1x1x2x2x2)
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {1 1 2 2 2} -dtype float32]

# Pad with 1 on all sides
set result [torch::reflection_pad3d $input {1 1 1 1 1 1}]
;# Result shape: 1x1x4x4x4
```

### Named Parameter Syntax
```tcl
# Using named parameters
set result [torch::reflectionPad3d -input $input -padding {1 1 1 1 1 1}]

# Using parameter aliases
set result [torch::reflectionPad3d -tensor $input -pad {1 1 1 1 1 1}]
```

### Uneven Padding
```tcl
# Pad with different values on each side
set result [torch::reflectionPad3d -input $input -padding {2 1 2 1 2 1}]
;# Result shape: 1x1x6x5x5
```

## Error Conditions

### Invalid Input Tensor
- If the input tensor name doesn't exist
- If the input tensor is not 5D
```tcl
# Wrong dimensions
set tensor2d [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
catch {torch::reflection_pad3d -input $tensor2d -padding {1 1 1 1 1 1}} err
;# Error: Expected 5D tensor for 3D padding, but got 2D tensor
```

### Invalid Padding Values
- If padding list doesn't contain exactly 6 values
- If any padding value is negative
```tcl
# Wrong number of padding values
catch {torch::reflection_pad3d $input {1 1 1}} err
;# Error: Padding must be a list of 6 values for 3D

# Negative padding
catch {torch::reflection_pad3d $input {-1 1 1 1 1 1}} err
;# Error: Invalid padding value: padding cannot be negative
```

### Missing Parameters
- If input tensor is not provided
- If padding values are not provided
```tcl
# Missing input
catch {torch::reflectionPad3d -padding {1 1 1 1 1 1}} err
;# Error: Required parameters missing: input tensor and padding values required

# Missing padding
catch {torch::reflectionPad3d -input $input} err
;# Error: Missing value for parameter
```

## See Also
- [torch::reflection_pad1d](reflection_pad1d.md) - 1D reflection padding
- [torch::reflection_pad2d](reflection_pad2d.md) - 2D reflection padding
- [torch::replication_pad3d](replication_pad3d.md) - Alternative 3D padding mode
- [torch::constant_pad3d](constant_pad3d.md) - 3D padding with constant values 