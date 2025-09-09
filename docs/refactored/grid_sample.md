# torch::grid_sample

## Overview
Applies a spatial transformation to an input tensor using a sampling grid. This function performs grid sampling by interpolating input values at locations specified by the grid coordinates, enabling arbitrary spatial transformations like rotation, scaling, and warping.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::grid_sample -input INPUT -grid GRID ?-mode MODE? ?-padding_mode PADDING? ?-align_corners BOOL?
torch::grid_sample -tensor INPUT -grid GRID ?-mode MODE? ?-paddingMode PADDING? ?-alignCorners BOOL?
```

### Positional Syntax (Legacy)
```tcl
torch::grid_sample INPUT GRID ?MODE? ?PADDING_MODE? ?ALIGN_CORNERS?
```

### camelCase Alias
```tcl
torch::gridSample -input INPUT -grid GRID ?-mode MODE? ?-paddingMode PADDING? ?-alignCorners BOOL?
```

## Parameters

### Required Parameters
- **`-input`** or **`-tensor`**: Input tensor to be sampled
  - Type: Tensor handle
  - Shape: (N, C, H_in, W_in) - 4D tensor with batch, channels, height, width
  - The input tensor from which values will be sampled

- **`-grid`**: Sampling grid coordinates
  - Type: Tensor handle  
  - Shape: (N, H_out, W_out, 2) - normalized coordinates in range [-1, 1]
  - Grid coordinates specify where to sample from the input tensor

### Optional Parameters
- **`-mode`**: Interpolation mode
  - Type: String
  - Values: `"bilinear"` (default), `"nearest"`
  - Default: `"bilinear"`
  - Specifies the interpolation method for sampling

- **`-padding_mode`** or **`-paddingMode`**: Padding mode for out-of-bounds coordinates
  - Type: String
  - Values: `"zeros"` (default), `"border"`, `"reflection"`
  - Default: `"zeros"`
  - Determines how to handle coordinates outside the input tensor

- **`-align_corners`** or **`-alignCorners`**: Corner alignment for coordinate normalization
  - Type: Boolean (0 or 1)
  - Default: `false` (0)
  - Controls how grid coordinates are normalized to input tensor coordinates

## Return Value
Returns a tensor handle containing the sampled output with shape (N, C, H_out, W_out), where H_out and W_out are determined by the grid dimensions.

## Examples

### Basic Grid Sampling
```tcl
# Create input tensor (1 batch, 1 channel, 4x4)
set input_data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0}
set input [torch::tensor_create -data $input_data -shape {1 1 4 4} -dtype float32]

# Create sampling grid (1 batch, 2x2 output, 2 coordinates)
# Grid values in range [-1, 1] correspond to normalized input coordinates
set grid_data {-0.5 -0.5  0.5 -0.5  -0.5  0.5   0.5  0.5}
set grid [torch::tensor_create -data $grid_data -shape {1 2 2 2} -dtype float32]

# Perform grid sampling using named syntax
set result [torch::grid_sample -input $input -grid $grid]
puts "Output shape: [torch::tensor_shape $result]"  ;# Output: 1 1 2 2

# Same operation using positional syntax
set result2 [torch::grid_sample $input $grid]
puts "Output shape: [torch::tensor_shape $result2]"  ;# Output: 1 1 2 2
```

### Different Interpolation Modes
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float32]

# Bilinear interpolation (default)
set bilinear_result [torch::grid_sample -input $input -grid $grid -mode bilinear]

# Nearest neighbor interpolation
set nearest_result [torch::grid_sample -input $input -grid $grid -mode nearest]
```

### Different Padding Modes
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
# Grid with out-of-bounds coordinates
set grid [torch::tensor_create -data {2.0 2.0} -shape {1 1 1 2} -dtype float32]

# Zero padding (default) - out-of-bounds gives 0
set zeros_result [torch::grid_sample -input $input -grid $grid -padding_mode zeros]

# Border padding - clamps to edge values
set border_result [torch::grid_sample -input $input -grid $grid -padding_mode border]

# Reflection padding - mirrors at boundaries
set reflection_result [torch::grid_sample -input $input -grid $grid -padding_mode reflection]
```

### Using camelCase Alias
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float32]

# Using camelCase alias with camelCase parameters
set result [torch::gridSample -input $input -grid $grid -mode bilinear -paddingMode border -alignCorners 1]
```

### Advanced Usage with Align Corners
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
set grid [torch::tensor_create -data {-1.0 -1.0  1.0  1.0} -shape {1 1 2 2} -dtype float32]

# Without align corners (default)
set result1 [torch::grid_sample -input $input -grid $grid -align_corners 0]

# With align corners - different coordinate interpretation
set result2 [torch::grid_sample -input $input -grid $grid -align_corners 1]
```

### Parameter Variations
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]
set grid [torch::tensor_create -data {0.0 0.0} -shape {1 1 1 2} -dtype float32]

# Alternative parameter names
set result1 [torch::grid_sample -tensor $input -grid $grid -padding_mode zeros]
set result2 [torch::grid_sample -input $input -grid $grid -paddingMode zeros]
set result3 [torch::grid_sample -input $input -grid $grid -align_corners 0]
set result4 [torch::grid_sample -input $input -grid $grid -alignCorners 0]
```

## Grid Coordinate System

The grid coordinates follow PyTorch's convention:
- **Range**: [-1, 1] for both x and y coordinates
- **Mapping**: 
  - (-1, -1) → top-left corner of input
  - (1, 1) → bottom-right corner of input
  - (0, 0) → center of input

```tcl
# Example: Identity transformation (no change)
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {1 1 2 2} -dtype float32]

# Create identity grid for 2x2 output
set identity_grid [torch::tensor_create -data {-1.0 -1.0  1.0 -1.0  -1.0  1.0   1.0  1.0} -shape {1 2 2 2} -dtype float32]
set result [torch::grid_sample -input $input -grid $identity_grid]
# Result should be approximately the same as input
```

## Interpolation Modes

### Bilinear Interpolation
- Smoothly interpolates between neighboring pixels
- Best for continuous transformations
- Default mode

### Nearest Neighbor
- Uses the value of the closest pixel
- Preserves sharp edges and discrete values
- Faster computation

## Padding Modes

### Zeros Padding
- Out-of-bounds coordinates return 0
- Default mode
- Good for most applications

### Border Padding  
- Out-of-bounds coordinates clamp to edge values
- Extends edge pixels infinitely
- Good for preventing artifacts at boundaries

### Reflection Padding
- Out-of-bounds coordinates reflect at boundaries
- Creates symmetric extensions
- Good for natural image processing

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::grid_sample} msg
puts $msg  ;# "Required parameters missing: input and grid tensors required"

# Invalid tensor handles
catch {torch::grid_sample "invalid_input" $grid} msg
puts $msg  ;# "Invalid input tensor"

# Invalid interpolation mode
catch {torch::grid_sample -input $input -grid $grid -mode invalid_mode} msg
puts $msg  ;# "Invalid mode: invalid_mode. Valid modes are: bilinear, nearest"

# Invalid padding mode
catch {torch::grid_sample -input $input -grid $grid -padding_mode invalid_padding} msg
puts $msg  ;# "Invalid padding_mode: invalid_padding. Valid modes are: zeros, border, reflection"

# Invalid align_corners parameter
catch {torch::grid_sample -input $input -grid $grid -align_corners "invalid"} msg
puts $msg  ;# "Invalid align_corners parameter: must be integer"
```

## Use Cases

### Image Transformation
```tcl
# Rotate image by sampling from rotated grid coordinates
# (Grid generation would typically be done by torch::affine_grid)
set image [torch::tensor_create -data $image_data -shape {1 3 224 224} -dtype float32]
set rotation_grid [torch::affine_grid $rotation_matrix {1 3 224 224}]
set rotated_image [torch::grid_sample -input $image -grid $rotation_grid -mode bilinear]
```

### Spatial Transformer Networks
```tcl
# Apply learned spatial transformation
set feature_map [torch::tensor_create -data $features -shape {8 256 32 32} -dtype float32]
set learned_grid [some_network_output]  ;# Generated by neural network
set transformed_features [torch::grid_sample -input $feature_map -grid $learned_grid]
```

### Data Augmentation
```tcl
# Apply random geometric transformations for training
set batch_images [torch::tensor_create -data $batch_data -shape {32 3 256 256} -dtype float32]
set augmentation_grid [generate_random_grid]
set augmented_batch [torch::grid_sample -input $batch_images -grid $augmentation_grid -mode bilinear -padding_mode reflection]
```

## Performance Notes

- **Memory Usage**: Output tensor size is determined by grid dimensions
- **Computational Cost**: Bilinear interpolation is more expensive than nearest neighbor
- **Gradient Flow**: Bilinear mode provides smooth gradients for training
- **Batch Processing**: Efficiently processes multiple samples simultaneously

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set result [torch::grid_sample $input $grid]
set result [torch::grid_sample $input $grid bilinear]
set result [torch::grid_sample $input $grid nearest zeros]
set result [torch::grid_sample $input $grid bilinear border 1]

# New named syntax
set result [torch::grid_sample -input $input -grid $grid]
set result [torch::grid_sample -input $input -grid $grid -mode bilinear]
set result [torch::grid_sample -input $input -grid $grid -mode nearest -padding_mode zeros]
set result [torch::grid_sample -input $input -grid $grid -mode bilinear -padding_mode border -align_corners 1]
```

### Parameter Mapping

| Positional | Named | Alternative |
|------------|-------|-------------|
| `input` | `-input` | `-tensor` |
| `grid` | `-grid` | N/A |
| `mode` | `-mode` | N/A |
| `padding_mode` | `-padding_mode` | `-paddingMode` |
| `align_corners` | `-align_corners` | `-alignCorners` |

## See Also

- [torch::affine_grid](affine_grid.md) - Generate transformation grids
- [torch::tensor_create](tensor_create.md) - Create input tensors
- [torch::interpolate](interpolate.md) - Alternative upsampling/downsampling
- [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) - Research paper introducing the concept

## Implementation Notes

- Based on PyTorch's `torch.nn.functional.grid_sample`
- Supports 4D tensors (batch, channel, height, width)
- Grid coordinates are normalized to [-1, 1] range
- Efficient implementation using LibTorch's native grid sampling
- Full gradient support for deep learning applications 