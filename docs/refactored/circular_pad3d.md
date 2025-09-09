# torch::circular_pad3d

## Overview

Applies circular padding to tensors along the last three dimensions (depth, height, and width). Circular padding wraps values around, so the values from the opposite faces of the tensor are used to pad each side. This is useful for 3D periodic data where the boundaries naturally connect.

**Status**: ✅ **REFACTORED** - Supports both snake_case and camelCase syntax with named parameters

## Syntax

### Current Syntax (Recommended)
```tcl
# Named parameters (recommended)
torch::circular_pad3d -input tensor_handle -padding {pad_left pad_right pad_top pad_bottom pad_front pad_back}
torch::circularPad3d -input tensor_handle -padding {pad_left pad_right pad_top pad_bottom pad_front pad_back}

# Alternative parameter names
torch::circular_pad3d -tensor tensor_handle -pad {pad_left pad_right pad_top pad_bottom pad_front pad_back}
```

### Legacy Syntax (Backward Compatible)
```tcl
# Positional parameters (still supported)
torch::circular_pad3d tensor_handle {pad_left pad_right pad_top pad_bottom pad_front pad_back}
torch::circularPad3d tensor_handle {pad_left pad_right pad_top pad_bottom pad_front pad_back}
```

## Parameters

### Named Parameters
- **`-input tensor_handle`** (required): Input tensor to pad
  - Alternative: **`-tensor tensor_handle`**
  - Must be a valid tensor handle with at least 4 dimensions
  - Typical format: `{batch_size, depth, height, width}` or `{batch_size, channels, depth, height, width}`
  
- **`-padding {pad_left pad_right pad_top pad_bottom pad_front pad_back}`** (required): Padding specification
  - Alternative: **`-pad {pad_left pad_right pad_top pad_bottom pad_front pad_back}`**
  - Type: List of 6 integers
  - `pad_left`: Number of columns to pad on the left side (width dimension)
  - `pad_right`: Number of columns to pad on the right side (width dimension)
  - `pad_top`: Number of rows to pad on the top (height dimension)
  - `pad_bottom`: Number of rows to pad on the bottom (height dimension)
  - `pad_front`: Number of slices to pad on the front (depth dimension)
  - `pad_back`: Number of slices to pad on the back (depth dimension)
  - **Constraint**: Each padding value must be less than the corresponding tensor dimension

### Legacy Positional Parameters
1. **`tensor_handle`**: Input tensor to pad
2. **`{pad_left pad_right pad_top pad_bottom pad_front pad_back}`**: List of 6 integers specifying padding amounts

## Return Value

Returns a handle to a new tensor with circular padding applied. The output tensor will have the same shape as the input except for the last three dimensions:
- Depth dimension increased by `pad_front + pad_back`
- Height dimension increased by `pad_top + pad_bottom`
- Width dimension increased by `pad_left + pad_right`

## Examples

### Basic Usage
```tcl
# Create a 4D tensor (batch_size=1, depth=2, height=3, width=4)
set values {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0}
set tensor [torch::tensor_create $values float32]
set tensor [torch::tensor_reshape $tensor {1 2 3 4}]

# Named parameter syntax (recommended)
set padded [torch::circular_pad3d -input $tensor -padding {1 1 1 1 1 1}]

# Legacy syntax (still works)
set padded [torch::circular_pad3d $tensor {1 1 1 1 1 1}]

# CamelCase alias
set padded [torch::circularPad3d -input $tensor -padding {1 1 1 1 1 1}]
```

### Advanced Examples
```tcl
# Symmetric padding
set symmetric [torch::circular_pad3d -input $tensor -padding {2 2 1 1 1 1}]

# Asymmetric padding
set asymmetric [torch::circular_pad3d -input $tensor -padding {1 3 0 2 1 0}]

# Width-only padding
set width_only [torch::circular_pad3d -input $tensor -padding {2 2 0 0 0 0}]

# Height-only padding  
set height_only [torch::circular_pad3d -input $tensor -padding {0 0 1 1 0 0}]

# Depth-only padding
set depth_only [torch::circular_pad3d -input $tensor -padding {0 0 0 0 1 1}]

# Parameter order flexibility
set flexible [torch::circular_pad3d -padding {1 1 1 1 1 1} -input $tensor]

# Alternative parameter names
set alternative [torch::circular_pad3d -tensor $tensor -pad {1 1 2 2 1 1}]
```

### 3D Medical Imaging
```tcl
proc pad_volume {volume_tensor pad_size} {
    # Pad a 3D volume with circular boundaries
    # Useful for 3D convolution operations with periodic boundary conditions
    return [torch::circular_pad3d -input $volume_tensor -padding [list $pad_size $pad_size $pad_size $pad_size $pad_size $pad_size]]
}

# Example usage with medical volume data
set medical_volume [torch::randn {1 64 64 64}]  # Single channel 64x64x64 volume
set padded_volume [pad_volume $medical_volume 4]
# Result shape: {1, 72, 72, 72}
```

### Video Processing
```tcl
# Process video data with temporal circular padding
set video_tensor [torch::randn {1 3 16 224 224}]  # Batch=1, RGB, 16 frames, 224x224

# Pad temporally (depth) and spatially
set padded_video [torch::circular_pad3d -input $video_tensor -padding {8 8 8 8 2 2}]
# Result shape: {1, 3, 20, 240, 240} (temporal +4, spatial +16)
```

### 3D Feature Maps
```tcl
proc pad_3d_features {feature_maps kernel_size} {
    # Pad 3D feature maps for 3D CNN operations
    set pad_size [expr {$kernel_size / 2}]
    return [torch::circular_pad3d -input $feature_maps -padding [list $pad_size $pad_size $pad_size $pad_size $pad_size $pad_size]]
}

# Example with 3D convolutional feature maps
set conv3d_features [torch::randn {1 128 8 16 16}]  # 128 feature maps of 8x16x16
set padded_features [pad_3d_features $conv3d_features 3]
# Result shape: {1, 128, 9, 17, 17} (padded for 3x3x3 conv)
```

### Batch Processing
```tcl
# Process multiple 3D samples simultaneously
set batch_volumes [torch::randn {4 1 32 32 32}]  # 4 volumes of 32x32x32

set batch_padded [torch::circular_pad3d -input $batch_volumes -padding {4 4 4 4 4 4}]
# Result shape: {4, 1, 40, 40, 40} (each volume padded from 32³ to 40³)
```

## Mathematical Description

Circular padding works by wrapping values from the opposite faces in 3D space:

**For a 3D tensor, padding wraps around each dimension:**
- **Width (left/right)**: Uses columns from opposite horizontal edges
- **Height (top/bottom)**: Uses rows from opposite vertical edges  
- **Depth (front/back)**: Uses slices from opposite depth faces

The result is a seamless 3D periodic extension where:
- Each face connects to its opposite face
- Corner and edge values are determined by the intersection of wrapped dimensions
- No artificial boundaries or discontinuities are introduced

## Tensor Shape Requirements

- **Input**: Tensor with at least 4 dimensions (PyTorch limitation)
- **Common formats**: 
  - `{batch_size, depth, height, width}` for single-channel 3D data
  - `{batch_size, channels, depth, height, width}` for multi-channel 3D data
  - `{channels, depth, height, width}` for single 3D samples
- **Output**: Same rank as input, last three dimensions grow by padding amounts
- **Padding constraint**: Each padding value must be less than the corresponding tensor dimension

## Error Handling

### Common Errors
```tcl
# Missing required parameters
torch::circular_pad3d
# Error: Usage: torch::circular_pad3d tensor padding | torch::circularPad3d -input tensor -padding {values}

# Invalid tensor handle
torch::circular_pad3d invalid_tensor {1 1 1 1 1 1}
# Error: Invalid tensor name: invalid_tensor

# Unknown parameter
torch::circular_pad3d -input $tensor -padding {1 1 1 1 1 1} -invalid_param value
# Error: Unknown parameter: -invalid_param. Valid parameters are: -input, -tensor, -padding, -pad

# Wrong number of padding values
torch::circular_pad3d -input $tensor -padding {1 2 3 4 5}
# Error: Padding must be a list of 6 values for 3D

torch::circular_pad3d -input $tensor -padding {1 2 3 4 5 6 7}
# Error: Padding must be a list of 6 values for 3D

# Invalid padding values
torch::circular_pad3d -input $tensor -padding {invalid 1 1 1 1 1}
# Error: expected integer but got "invalid"

# Padding too large (causes wrapping more than once)
torch::circular_pad3d -input $small_tensor -padding {10 10 10 10 10 10}
# Error: Padding value causes wrapping around more than once
```

## Performance Notes

- **Equivalent Performance**: Named parameter syntax has the same performance as legacy syntax
- **Memory Efficient**: Creates new tensor with minimal memory overhead
- **GPU Support**: Works with CUDA tensors for GPU acceleration
- **3D Operations**: Optimized for 3D convolution and processing operations
- **Batch Optimized**: Efficient processing of multiple 3D samples simultaneously
- **Padding Constraint**: Keep padding values smaller than tensor dimensions for optimal performance

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Before (legacy - still works)
set padded [torch::circular_pad3d $tensor {1 2 1 2 1 1}]

# After (recommended)
set padded [torch::circular_pad3d -input $tensor -padding {1 2 1 2 1 1}]

# CamelCase alternative
set padded [torch::circularPad3d -input $tensor -padding {1 2 1 2 1 1}]
```

### Parameter Mapping
| Legacy Position | Named Parameter | Alternative | Description |
|----------------|-----------------|-------------|-------------|
| 1st argument   | `-input`        | `-tensor`   | Input tensor handle |
| 2nd argument   | `-padding`      | `-pad`      | `{left, right, top, bottom, front, back}` |

## Use Cases

1. **Medical Imaging**: Padding 3D medical volumes (MRI, CT scans) with periodic boundaries
2. **3D Computer Vision**: Preparing 3D point clouds and voxel data for processing
3. **Video Processing**: Temporal padding for video sequences with periodic content
4. **3D Convolutional Neural Networks**: Preparing 3D feature maps for conv3d operations
5. **Scientific Computing**: 3D simulation data with periodic boundary conditions
6. **Volumetric Rendering**: Extending 3D textures and volumes for rendering operations
7. **Climate Modeling**: Processing atmospheric and oceanic data with periodic boundaries
8. **Crystallography**: Working with periodic crystal structures

## Implementation Details

- **Backward Compatible**: Legacy positional syntax fully supported
- **Input Validation**: Comprehensive parameter and tensor validation
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Flexibility**: Multiple parameter names and order independence
- **3D Optimized**: Efficient handling of 3D tensor operations
- **Thread Safe**: Safe for concurrent execution
- **Memory Safe**: Proper tensor memory management

## Related Commands

- [`torch::circular_pad1d`](circular_pad1d.md) - 1D circular padding
- [`torch::circular_pad2d`](circular_pad2d.md) - 2D circular padding  
- [`torch::constant_pad3d`](constant_pad3d.md) - Constant value padding
- [`torch::reflection_pad3d`](reflection_pad3d.md) - Reflection padding
- [`torch::replication_pad3d`](replication_pad3d.md) - Replication padding

## Comparison with Other Padding Types

| Padding Type | Behavior | Use Case |
|-------------|----------|----------|
| **Circular** | Wraps from opposite faces | Periodic 3D data, seamless boundaries |
| **Constant** | Fills with constant value | Zero-padding, solid color/value borders |
| **Reflection** | Mirrors values at faces | Smooth 3D boundaries, natural extension |
| **Replication** | Repeats edge values | Extending 3D boundaries without discontinuity |

## Version History

- **v1.0**: Original implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained with enhanced error handling

---

**Note**: This command is part of the LibTorch TCL Extension refactoring initiative, providing modern, user-friendly APIs while maintaining full backward compatibility. 