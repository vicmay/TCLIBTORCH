# torch::constant_pad1d / torch::constantPad1d

## Description
Pads a 1D tensor with a constant value. This operation extends the tensor by adding specified amounts of padding on both sides, filled with a constant value. This is commonly used in signal processing and neural networks for maintaining tensor dimensions through convolutional layers.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::constant_pad1d tensor padding value
```

### New Syntax (Named Parameters)
```tcl
torch::constant_pad1d -input tensor -padding {left right} -value num
torch::constant_pad1d -tensor tensor -pad {left right} -val num
```

### CamelCase Alias
```tcl
torch::constantPad1d tensor padding value
torch::constantPad1d -input tensor -padding {left right} -value num
```

## Parameters

### Positional Format
- **tensor**: Input tensor (required)
- **padding**: List of 2 integers `{left right}` specifying padding amounts (required)
- **value**: Constant value to use for padding (required)

### Named Parameter Format
- **-input/-tensor**: Input tensor (required)
- **-padding/-pad**: List of 2 integers `{left right}` specifying padding amounts (required)
- **-value/-val**: Constant value to use for padding (required)

### Padding Values
- **left**: Number of padding elements to add on the left side
- **right**: Number of padding elements to add on the right side
- Values can be 0 or positive integers
- Negative values may be supported (depending on PyTorch version) for cropping

## Return Value
Returns a new tensor handle with the padded result.

## Examples

### Basic Usage
```tcl
# Create a simple 1D tensor
set input [torch::tensor_create {1.0 2.0 3.0} float32]

# Original syntax - pad with 1 on left, 2 on right, value 0.0
set result1 [torch::constant_pad1d $input {1 2} 0.0]
# Result: [0.0, 1.0, 2.0, 3.0, 0.0, 0.0]

# Named parameter syntax
set result2 [torch::constant_pad1d -input $input -padding {2 1} -value 5.0]
# Result: [5.0, 5.0, 1.0, 2.0, 3.0, 5.0]

# CamelCase alias
set result3 [torch::constantPad1d $input {0 3} -1.0]
# Result: [1.0, 2.0, 3.0, -1.0, -1.0, -1.0]
```

### Signal Processing Example
```tcl
# Audio signal padding for convolution
set audio_signal [torch::tensor_create {0.1 0.5 0.8 0.3 0.2} float32]

# Add padding to prevent boundary effects
set padded_signal [torch::constant_pad1d $audio_signal {2 2} 0.0]
# Result: [0.0, 0.0, 0.1, 0.5, 0.8, 0.3, 0.2, 0.0, 0.0]

# Alternative with named parameters
set padded_signal2 [torch::constant_pad1d -tensor $audio_signal -pad {3 1} -val 0.0]
```

### Different Data Types
```tcl
# Integer tensor
set int_tensor [torch::tensor_create {1 2 3 4} int32]
set padded_int [torch::constant_pad1d $int_tensor {1 1} 0]

# Float tensor with negative padding value
set float_tensor [torch::tensor_create {1.5 2.5 3.5} float32]
set padded_float [torch::constant_pad1d $float_tensor {2 0} -1.0]
```

### Zero Padding (No Change)
```tcl
set input [torch::tensor_create {1.0 2.0 3.0} float32]
set unchanged [torch::constant_pad1d $input {0 0} 999.0]
# Result: [1.0, 2.0, 3.0] - no padding added
```

## Mathematical Background

Constant padding extends a tensor by replicating a constant value:

For input tensor `x = [x₁, x₂, ..., xₙ]` with padding `{left, right}` and constant `c`:

Result = `[c, c, ..., c, x₁, x₂, ..., xₙ, c, c, ..., c]`
         `←─ left ─→              ←─ right ─→`

## Common Use Cases

1. **Signal Processing**: Preventing boundary effects in convolutions
2. **Neural Networks**: Maintaining tensor dimensions through conv1d layers
3. **Time Series**: Adding buffer zones for temporal analysis
4. **Data Preprocessing**: Standardizing sequence lengths

## Error Handling

The command validates:
- Tensor existence and validity
- Padding format (must be list of exactly 2 integers)
- Parameter completeness in named syntax
- Value type compatibility

Common error scenarios:
```tcl
# Wrong number of padding values
torch::constant_pad1d $input {1} 0.0          # Error: needs 2 values
torch::constant_pad1d $input {1 2 3} 0.0      # Error: too many values

# Missing parameters
torch::constant_pad1d -input $input -padding  # Error: missing value

# Invalid tensor
torch::constant_pad1d invalid_tensor {1 1} 0.0  # Error: tensor not found
```

## Performance Notes

- Padding operations are generally fast and memory-efficient
- Large padding values will create proportionally larger output tensors
- The operation is implemented using PyTorch's optimized padding functions

## Migration Guide

### From Original to Named Parameters
```tcl
# Old style
set result [torch::constant_pad1d $tensor {2 3} 1.0]

# New style (equivalent)
set result [torch::constant_pad1d -input $tensor -padding {2 3} -value 1.0]

# CamelCase alias
set result [torch::constantPad1d -tensor $tensor -pad {2 3} -val 1.0]
```

### Parameter Aliases
- `-input` ↔ `-tensor`
- `-padding` ↔ `-pad`
- `-value` ↔ `-val`

## See Also
- `torch::constant_pad2d` - 2D constant padding
- `torch::constant_pad3d` - 3D constant padding
- `torch::circular_pad1d` - 1D circular padding
- `torch::reflection_pad1d` - 1D reflection padding 