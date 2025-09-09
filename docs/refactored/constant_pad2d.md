# torch::constant_pad2d / torch::constantPad2d

## Description
Pads a 2D tensor with a constant value. This operation extends the tensor by adding specified amounts of padding on all four sides (left, right, top, bottom), filled with a constant value. This is extensively used in image processing and 2D convolutional neural networks for maintaining spatial dimensions and preventing boundary artifacts.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::constant_pad2d tensor padding value
```

### New Syntax (Named Parameters)
```tcl
torch::constant_pad2d -input tensor -padding {left right top bottom} -value num
torch::constant_pad2d -tensor tensor -pad {left right top bottom} -val num
```

### CamelCase Alias
```tcl
torch::constantPad2d tensor padding value
torch::constantPad2d -input tensor -padding {left right top bottom} -value num
```

## Parameters

### Positional Format
- **tensor**: Input 2D tensor (required)
- **padding**: List of 4 integers `{left right top bottom}` specifying padding amounts (required)
- **value**: Constant value to use for padding (required)

### Named Parameter Format
- **-input/-tensor**: Input 2D tensor (required)
- **-padding/-pad**: List of 4 integers `{left right top bottom}` specifying padding amounts (required)
- **-value/-val**: Constant value to use for padding (required)

### Padding Values
- **left**: Number of padding columns to add on the left side
- **right**: Number of padding columns to add on the right side  
- **top**: Number of padding rows to add on the top
- **bottom**: Number of padding rows to add on the bottom
- Values can be 0 or positive integers
- Negative values may be supported (depending on PyTorch version) for cropping

## Return Value
Returns a new tensor handle with the padded result.

## Examples

### Basic Usage
```tcl
# Create a simple 2x2 matrix
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set input [torch::tensor_reshape $input {2 2}]

# Original syntax - pad with 1 on all sides, value 0.0
set result1 [torch::constant_pad2d $input {1 1 1 1} 0.0]
# Result: 4x4 matrix with 1-pixel border of zeros

# Named parameter syntax - asymmetric padding
set result2 [torch::constant_pad2d -input $input -padding {2 1 0 3} -value 5.0]
# Result: left=2, right=1, top=0, bottom=3 padding with value 5.0

# CamelCase alias
set result3 [torch::constantPad2d $input {0 2 1 0} -1.0]
# Result: right=2, top=1 padding with value -1.0
```

### Image Processing Example
```tcl
# Create a 3x3 "image"
set image_data {}
for {set i 1} {$i <= 9} {incr i} {
    lappend image_data [expr {$i * 1.0}]
}
set image [torch::tensor_create $image_data float32]
set image [torch::tensor_reshape $image {3 3}]

# Add border for edge detection algorithms
set bordered_image [torch::constant_pad2d $image {1 1 1 1} 0.0]
# Result: 5x5 image with 1-pixel black border

# Asymmetric padding for specific image transformations
set padded_image [torch::constant_pad2d -tensor $image -pad {2 0 1 3} -val 128.0]
# Common in image preprocessing (128 = middle gray value)
```

### Convolutional Neural Network Padding
```tcl
# Feature map padding to maintain spatial dimensions
set feature_map [torch::tensor_create $values float32]
set feature_map [torch::tensor_reshape $feature_map {8 8}]

# SAME padding for 3x3 convolution
set padded_features [torch::constant_pad2d $feature_map {1 1 1 1} 0.0]

# Valid padding (no padding)
set unpadded_features [torch::constant_pad2d $feature_map {0 0 0 0} 0.0]
```

### Different Data Types
```tcl
# Integer matrix
set int_matrix [torch::tensor_create {1 2 3 4} int32]
set int_matrix [torch::tensor_reshape $int_matrix {2 2}]
set padded_int [torch::constant_pad2d $int_matrix {1 1 1 1} 0]

# Float matrix with negative padding value
set float_matrix [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
set float_matrix [torch::tensor_reshape $float_matrix {2 2}]
set padded_float [torch::constant_pad2d $float_matrix {2 0 0 1} -1.0]
```

### Specialized Padding Patterns
```tcl
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set input [torch::tensor_reshape $input {2 2}]

# Horizontal only (left and right)
set horizontal_pad [torch::constant_pad2d $input {2 3 0 0} 0.0]

# Vertical only (top and bottom)  
set vertical_pad [torch::constant_pad2d $input {0 0 1 2} 0.0]

# Symmetric padding
set symmetric_pad [torch::constant_pad2d $input {2 2 2 2} 1.0]
```

## Mathematical Background

For a 2D tensor (matrix) with dimensions H×W, constant padding creates:

```
Original matrix: H×W
Padded matrix: (H + top + bottom) × (W + left + right)
```

Visual representation:
```
    ←left→           ←right→
    ┌─────┬─────────┬─────────┐ ↑
    │  c  │    c    │    c    │ │top
    ├─────┼─────────┼─────────┤ ↓
    │  c  │ original│    c    │
    │  c  │  matrix │    c    │
    ├─────┼─────────┼─────────┤ ↑
    │  c  │    c    │    c    │ │bottom
    └─────┴─────────┴─────────┘ ↓
```

Where `c` represents the constant padding value.

## Common Use Cases

1. **Image Processing**: Adding borders for edge detection, filtering
2. **Computer Vision**: Preprocessing images for neural networks
3. **Convolutional Networks**: Maintaining spatial dimensions through conv2d layers
4. **Medical Imaging**: Adding context borders for patch-based analysis
5. **Data Augmentation**: Creating variations of training images

## Error Handling

The command validates:
- Tensor existence and 2D shape compatibility
- Padding format (must be list of exactly 4 integers)
- Parameter completeness in named syntax
- Value type compatibility

Common error scenarios:
```tcl
# Wrong number of padding values
torch::constant_pad2d $input {1 2} 0.0          # Error: needs 4 values
torch::constant_pad2d $input {1 2 3 4 5} 0.0    # Error: too many values

# Missing parameters
torch::constant_pad2d -input $input -padding    # Error: missing value

# Invalid tensor
torch::constant_pad2d invalid_tensor {1 1 1 1} 0.0  # Error: tensor not found
```

## Performance Notes

- 2D padding operations are optimized for common image sizes
- Memory usage scales with output dimensions: (H+top+bottom)×(W+left+right)
- GPU acceleration available for large tensors
- Symmetric padding patterns may have optimization advantages

## Migration Guide

### From Original to Named Parameters
```tcl
# Old style
set result [torch::constant_pad2d $tensor {1 2 0 3} 1.0]

# New style (equivalent)
set result [torch::constant_pad2d -input $tensor -padding {1 2 0 3} -value 1.0]

# CamelCase alias
set result [torch::constantPad2d -tensor $tensor -pad {1 2 0 3} -val 1.0]
```

### Parameter Aliases
- `-input` ↔ `-tensor`
- `-padding` ↔ `-pad`
- `-value` ↔ `-val`

### Padding Order
Remember the padding order: `{left, right, top, bottom}`
- Horizontal padding: left, right
- Vertical padding: top, bottom

## See Also
- `torch::constant_pad1d` - 1D constant padding
- `torch::constant_pad3d` - 3D constant padding
- `torch::circular_pad2d` - 2D circular padding
- `torch::reflection_pad2d` - 2D reflection padding
- `torch::conv2d` - 2D convolution operations 