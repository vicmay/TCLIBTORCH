# torch::circular_pad2d

## Overview

Applies circular padding to tensors along the last two dimensions (height and width). Circular padding wraps values around, so the values from the opposite edges of the tensor are used to pad each side. This is useful for 2D periodic data where the edges naturally connect.

**Status**: ✅ **REFACTORED** - Supports both snake_case and camelCase syntax with named parameters

## Syntax

### Current Syntax (Recommended)
```tcl
# Named parameters (recommended)
torch::circular_pad2d -input tensor_handle -padding {pad_left pad_right pad_top pad_bottom}
torch::circularPad2d -input tensor_handle -padding {pad_left pad_right pad_top pad_bottom}

# Alternative parameter names
torch::circular_pad2d -tensor tensor_handle -pad {pad_left pad_right pad_top pad_bottom}
```

### Legacy Syntax (Backward Compatible)
```tcl
# Positional parameters (still supported)
torch::circular_pad2d tensor_handle {pad_left pad_right pad_top pad_bottom}
torch::circularPad2d tensor_handle {pad_left pad_right pad_top pad_bottom}
```

## Parameters

### Named Parameters
- **`-input tensor_handle`** (required): Input tensor to pad
  - Alternative: **`-tensor tensor_handle`**
  - Must be a valid tensor handle with at least 3 dimensions
  - Typical format: `{batch_size, height, width}` or `{batch_size, channels, height, width}`
  
- **`-padding {pad_left pad_right pad_top pad_bottom}`** (required): Padding specification
  - Alternative: **`-pad {pad_left pad_right pad_top pad_bottom}`**
  - Type: List of 4 integers
  - `pad_left`: Number of columns to pad on the left side
  - `pad_right`: Number of columns to pad on the right side
  - `pad_top`: Number of rows to pad on the top
  - `pad_bottom`: Number of rows to pad on the bottom
  - **Constraint**: Padding values must be less than the corresponding tensor dimension

### Legacy Positional Parameters
1. **`tensor_handle`**: Input tensor to pad
2. **`{pad_left pad_right pad_top pad_bottom}`**: List of 4 integers specifying padding amounts

## Return Value

Returns a handle to a new tensor with circular padding applied. The output tensor will have the same shape as the input except for the last two dimensions:
- Height dimension increased by `pad_top + pad_bottom`
- Width dimension increased by `pad_left + pad_right`

## Examples

### Basic Usage
```tcl
# Create a 3D tensor (batch_size=1, height=3, width=4)
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0} float32]
set tensor [torch::tensor_reshape $tensor {1 3 4}]

# Named parameter syntax (recommended)
set padded [torch::circular_pad2d -input $tensor -padding {2 1 1 1}]

# Legacy syntax (still works)
set padded [torch::circular_pad2d $tensor {2 1 1 1}]

# CamelCase alias
set padded [torch::circularPad2d -input $tensor -padding {2 1 1 1}]
```

### Advanced Examples
```tcl
# Symmetric padding
set symmetric [torch::circular_pad2d -input $tensor -padding {2 2 1 1}]

# Asymmetric padding
set asymmetric [torch::circular_pad2d -input $tensor -padding {1 3 0 2}]

# Horizontal-only padding
set horizontal [torch::circular_pad2d -input $tensor -padding {2 2 0 0}]

# Vertical-only padding  
set vertical [torch::circular_pad2d -input $tensor -padding {0 0 1 1}]

# Parameter order flexibility
set flexible [torch::circular_pad2d -padding {1 1 1 1} -input $tensor]

# Alternative parameter names
set alternative [torch::circular_pad2d -tensor $tensor -pad {1 2 1 2}]
```

### Image Processing
```tcl
proc pad_image {image_tensor pad_size} {
    # Pad an image with circular boundaries
    # Useful for convolution operations with periodic boundary conditions
    return [torch::circular_pad2d -input $image_tensor -padding [list $pad_size $pad_size $pad_size $pad_size]]
}

# Example usage with RGB image (channels, height, width)
set rgb_image [torch::randn {3 64 64}]
set padded_image [pad_image $rgb_image 8]
# Result shape: {3, 80, 80}
```

### Batch Processing
```tcl
# Process multiple images simultaneously
set batch_images [torch::randn {8 3 32 32}]  # 8 RGB images of 32x32

set batch_padded [torch::circular_pad2d -input $batch_images -padding {4 4 4 4}]
# Result shape: {8, 3, 40, 40} (each image padded from 32x32 to 40x40)
```

### Feature Map Padding
```tcl
proc pad_feature_maps {feature_maps receptive_field} {
    # Pad feature maps for CNN operations
    set pad_size [expr {$receptive_field / 2}]
    return [torch::circular_pad2d -input $feature_maps -padding [list $pad_size $pad_size $pad_size $pad_size]]
}

# Example with convolutional feature maps
set conv_features [torch::randn {1 64 28 28}]  # 64 feature maps of 28x28
set padded_features [pad_feature_maps $conv_features 5]
# Result shape: {1, 64, 33, 33} (padded for 5x5 conv)
```

## Mathematical Description

Circular padding works by wrapping values from the opposite edges:

**For a 2D tensor:**
```
Original:    [a b c]
             [d e f]
             [g h i]

With padding {1,1,1,1}:
[i g h i g]
[c a b c a]  <- Original top row becomes bottom padding
[f d e f d]  <- Original data
[i g h i g]  <- Original bottom row becomes top padding
[c a b c a]
```

The circular nature means:
- **Left/Right padding**: Uses columns from the opposite horizontal edge
- **Top/Bottom padding**: Uses rows from the opposite vertical edge
- Creates seamless periodic extension in both dimensions

## Tensor Shape Requirements

- **Input**: Tensor with at least 3 dimensions (PyTorch limitation)
- **Common formats**: 
  - `{batch_size, height, width}` for grayscale images
  - `{batch_size, channels, height, width}` for color images
  - `{channels, height, width}` for single images
- **Output**: Same rank as input, last two dimensions grow by padding amounts
- **Padding constraint**: Each padding value must be less than the corresponding tensor dimension

## Error Handling

### Common Errors
```tcl
# Missing required parameters
torch::circular_pad2d
# Error: Usage: torch::circular_pad2d tensor padding | torch::circularPad2d -input tensor -padding {values}

# Invalid tensor handle
torch::circular_pad2d invalid_tensor {1 2 1 1}
# Error: Invalid tensor name: invalid_tensor

# Unknown parameter
torch::circular_pad2d -input $tensor -padding {1 2 1 1} -invalid_param value
# Error: Unknown parameter: -invalid_param. Valid parameters are: -input, -tensor, -padding, -pad

# Wrong number of padding values
torch::circular_pad2d -input $tensor -padding {1 2 3}
# Error: Padding must be a list of 4 values for 2D

torch::circular_pad2d -input $tensor -padding {1 2 3 4 5}
# Error: Padding must be a list of 4 values for 2D

# Invalid padding values
torch::circular_pad2d -input $tensor -padding {invalid 2 1 1}
# Error: expected integer but got "invalid"

# Padding too large (causes wrapping more than once)
torch::circular_pad2d -input $small_tensor -padding {10 10 10 10}
# Error: Padding value causes wrapping around more than once
```

## Performance Notes

- **Equivalent Performance**: Named parameter syntax has the same performance as legacy syntax
- **Memory Efficient**: Creates new tensor with minimal memory overhead
- **GPU Support**: Works with CUDA tensors for GPU acceleration
- **Batch Optimized**: Efficient processing of multiple images/feature maps simultaneously
- **Padding Constraint**: Keep padding values smaller than tensor dimensions for optimal performance

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Before (legacy - still works)
set padded [torch::circular_pad2d $tensor {2 3 1 2}]

# After (recommended)
set padded [torch::circular_pad2d -input $tensor -padding {2 3 1 2}]

# CamelCase alternative
set padded [torch::circularPad2d -input $tensor -padding {2 3 1 2}]
```

### Parameter Mapping
| Legacy Position | Named Parameter | Alternative | Description |
|----------------|-----------------|-------------|-------------|
| 1st argument   | `-input`        | `-tensor`   | Input tensor handle |
| 2nd argument   | `-padding`      | `-pad`      | `{left, right, top, bottom}` |

## Use Cases

1. **Computer Vision**: Padding images with periodic boundary conditions
2. **Convolutional Neural Networks**: Preparing feature maps for convolution operations
3. **Signal Processing**: 2D signal extension with circular boundaries
4. **Texture Synthesis**: Creating seamless texture boundaries
5. **Image Filtering**: Extending images for filter operations without edge artifacts
6. **Geographic Data**: Processing periodic spatial data (e.g., longitude wrapping)

## Implementation Details

- **Backward Compatible**: Legacy positional syntax fully supported
- **Input Validation**: Comprehensive parameter and tensor validation
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Flexibility**: Multiple parameter names and order independence
- **Thread Safe**: Safe for concurrent execution
- **Memory Safe**: Proper tensor memory management

## Related Commands

- [`torch::circular_pad1d`](circular_pad1d.md) - 1D circular padding
- [`torch::circular_pad3d`](circular_pad3d.md) - 3D circular padding  
- [`torch::constant_pad2d`](constant_pad2d.md) - Constant value padding
- [`torch::reflection_pad2d`](reflection_pad2d.md) - Reflection padding
- [`torch::replication_pad2d`](replication_pad2d.md) - Replication padding

## Comparison with Other Padding Types

| Padding Type | Behavior | Use Case |
|-------------|----------|----------|
| **Circular** | Wraps from opposite edges | Periodic data, seamless boundaries |
| **Constant** | Fills with constant value | Zero-padding, solid color borders |
| **Reflection** | Mirrors values at edges | Smooth boundaries, natural extension |
| **Replication** | Repeats edge values | Extending boundaries without discontinuity |

## Version History

- **v1.0**: Original implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained with enhanced error handling

---

**Note**: This command is part of the LibTorch TCL Extension refactoring initiative, providing modern, user-friendly APIs while maintaining full backward compatibility. 