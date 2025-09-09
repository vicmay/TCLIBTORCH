# torch::constant_pad3d / torch::constantPad3d

## Description
Pads a 3D tensor with a constant value. This operation extends the tensor by adding specified amounts of padding on all six faces (left, right, top, bottom, front, back), filled with a constant value. This is crucial for 3D medical imaging, video processing, and 3D convolutional neural networks for maintaining volumetric dimensions and preventing boundary artifacts.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::constant_pad3d tensor padding value
```

### New Syntax (Named Parameters)
```tcl
torch::constant_pad3d -input tensor -padding {left right top bottom front back} -value num
torch::constant_pad3d -tensor tensor -pad {left right top bottom front back} -val num
```

### CamelCase Alias
```tcl
torch::constantPad3d tensor padding value
torch::constantPad3d -input tensor -padding {left right top bottom front back} -value num
```

## Parameters

### Positional Format
- **tensor**: Input 3D tensor (required)
- **padding**: List of 6 integers `{left right top bottom front back}` specifying padding amounts (required)
- **value**: Constant value to use for padding (required)

### Named Parameter Format
- **-input/-tensor**: Input 3D tensor (required)
- **-padding/-pad**: List of 6 integers `{left right top bottom front back}` specifying padding amounts (required)
- **-value/-val**: Constant value to use for padding (required)

### Padding Values
- **left**: Number of padding slices to add on the left side (X-axis negative)
- **right**: Number of padding slices to add on the right side (X-axis positive)
- **top**: Number of padding slices to add on the top (Y-axis negative)
- **bottom**: Number of padding slices to add on the bottom (Y-axis positive)
- **front**: Number of padding slices to add on the front (Z-axis negative)
- **back**: Number of padding slices to add on the back (Z-axis positive)
- Values can be 0 or positive integers
- Negative values may be supported (depending on PyTorch version) for cropping

## Return Value
Returns a new tensor handle with the padded result.

## Examples

### Basic Usage
```tcl
# Create a simple 2x2x2 volume
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
set input [torch::tensor_reshape $input {2 2 2}]

# Original syntax - pad with 1 on all faces, value 0.0
set result1 [torch::constant_pad3d $input {1 1 1 1 1 1} 0.0]
# Result: 4x4x4 volume with 1-voxel border of zeros

# Named parameter syntax - asymmetric padding
set result2 [torch::constant_pad3d -input $input -padding {2 1 0 3 1 0} -value 5.0]
# Result: left=2, right=1, top=0, bottom=3, front=1, back=0 padding

# CamelCase alias
set result3 [torch::constantPad3d $input {0 2 1 0 0 1} -1.0]
# Result: selective padding on specific faces
```

### Medical Imaging Example
```tcl
# Create a 3D medical volume (e.g., CT scan slice)
set volume_data {}
for {set i 1} {$i <= 8} {incr i} {
    lappend volume_data [expr {$i * 10.0}]
}
set medical_volume [torch::tensor_create $volume_data float32]
set medical_volume [torch::tensor_reshape $medical_volume {2 2 2}]

# Add padding for context-aware analysis
set padded_volume [torch::constant_pad3d $medical_volume {1 1 1 1 1 1} 0.0]
# Result: 4x4x4 volume with background padding

# Asymmetric padding for ROI extraction
set roi_volume [torch::constant_pad3d -tensor $medical_volume -pad {2 0 1 3 0 2} -val -1000.0]
# Using -1000 HU (Hounsfield Units) for air/background in CT
```

### Video Processing Example
```tcl
# Video frame sequence processing
set frame_sequence [torch::tensor_create $frame_data float32]
set frame_sequence [torch::tensor_reshape $frame_sequence {8 6 4}]  # Time x Height x Width

# Temporal padding for video analysis
set padded_sequence [torch::constant_pad3d $frame_sequence {1 1 2 2 1 1} 0.0]
# Add spatial and temporal context

# Spatial-only padding (preserve temporal dimension)
set spatial_padded [torch::constant_pad3d -input $frame_sequence -padding {2 2 1 1 0 0} -value 128.0]
```

### 3D Convolutional Neural Network Padding
```tcl
# 3D feature volume padding
set feature_volume [torch::tensor_create $values float32]
set feature_volume [torch::tensor_reshape $feature_volume {8 8 8}]

# SAME padding for 3x3x3 convolution
set padded_features [torch::constant_pad3d $feature_volume {1 1 1 1 1 1} 0.0]

# Anisotropic padding (different for each dimension)
set aniso_padded [torch::constant_pad3d -tensor $feature_volume -pad {2 2 1 1 0 0} -val 0.0]
```

### Different Data Types
```tcl
# Integer volume
set int_volume [torch::tensor_create {1 2 3 4 5 6 7 8} int32]
set int_volume [torch::tensor_reshape $int_volume {2 2 2}]
set padded_int [torch::constant_pad3d $int_volume {1 1 1 1 1 1} 0]

# Float volume with negative padding value
set float_volume [torch::tensor_create {1.5 2.5 3.5 4.5 5.5 6.5 7.5 8.5} float32]
set float_volume [torch::tensor_reshape $float_volume {2 2 2}]
set padded_float [torch::constant_pad3d $float_volume {1 0 0 1 2 0} -1.0]
```

### Specialized 3D Padding Patterns
```tcl
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
set input [torch::tensor_reshape $input {2 2 2}]

# X-Y plane only (spatial padding, no depth)
set spatial_pad [torch::constant_pad3d $input {2 2 1 1 0 0} 0.0]

# Z-axis only (depth/temporal padding)
set depth_pad [torch::constant_pad3d $input {0 0 0 0 2 3} 0.0]

# Symmetric volumetric padding
set symmetric_pad [torch::constant_pad3d $input {2 2 2 2 2 2} 1.0]

# Single-face padding
set single_face [torch::constant_pad3d $input {0 0 0 0 0 3} 0.0]  # Back face only
```

## Mathematical Background

For a 3D tensor with dimensions D×H×W, constant padding creates:

```
Original volume: D×H×W
Padded volume: (D + front + back) × (H + top + bottom) × (W + left + right)
```

3D Padding visualization:
```
                    ←left→     ←right→
              ┌─────────┬───────┬─────────┐ ↑
              │    c    │   c   │    c    │ │top
              ├─────────┼───────┼─────────┤ ↓
              │    c    │original│   c    │
              │    c    │ volume │   c    │
              ├─────────┼───────┼─────────┤ ↑
              │    c    │   c   │    c    │ │bottom
              └─────────┴───────┴─────────┘ ↓
                      ↑front          ↑back
```

Where `c` represents the constant padding value extending in all three dimensions.

## Common Use Cases

1. **Medical Imaging**: 3D CT/MRI volume processing with context borders
2. **Video Analysis**: Temporal and spatial padding for video CNNs
3. **3D Computer Vision**: Volumetric object detection and segmentation
4. **Scientific Computing**: 3D simulation boundary conditions
5. **Geospatial Analysis**: 3D geological and atmospheric data processing
6. **Robotics**: 3D occupancy grid and voxel-based navigation

## Error Handling

The command validates:
- Tensor existence and 3D shape compatibility
- Padding format (must be list of exactly 6 integers)
- Parameter completeness in named syntax
- Value type compatibility

Common error scenarios:
```tcl
# Wrong number of padding values
torch::constant_pad3d $input {1 2 3 4} 0.0         # Error: needs 6 values
torch::constant_pad3d $input {1 2 3 4 5 6 7} 0.0   # Error: too many values

# Missing parameters
torch::constant_pad3d -input $input -padding       # Error: missing value

# Invalid tensor
torch::constant_pad3d invalid_tensor {1 1 1 1 1 1} 0.0  # Error: tensor not found
```

## Performance Notes

- 3D padding operations can be memory-intensive for large volumes
- Memory usage scales with output dimensions: (D+front+back)×(H+top+bottom)×(W+left+right)
- GPU acceleration highly recommended for large 3D tensors
- Consider memory constraints when processing high-resolution medical volumes
- Symmetric padding patterns may have optimization advantages

## Migration Guide

### From Original to Named Parameters
```tcl
# Old style
set result [torch::constant_pad3d $tensor {1 2 0 3 1 0} 1.0]

# New style (equivalent)
set result [torch::constant_pad3d -input $tensor -padding {1 2 0 3 1 0} -value 1.0]

# CamelCase alias
set result [torch::constantPad3d -tensor $tensor -pad {1 2 0 3 1 0} -val 1.0]
```

### Parameter Aliases
- `-input` ↔ `-tensor`
- `-padding` ↔ `-pad`
- `-value` ↔ `-val`

### Padding Order
Remember the 3D padding order: `{left, right, top, bottom, front, back}`
- X-axis (width): left, right
- Y-axis (height): top, bottom
- Z-axis (depth): front, back

## See Also
- `torch::constant_pad1d` - 1D constant padding
- `torch::constant_pad2d` - 2D constant padding
- `torch::circular_pad3d` - 3D circular padding
- `torch::reflection_pad3d` - 3D reflection padding
- `torch::conv3d` - 3D convolution operations
- `torch::max_pool3d` - 3D max pooling 