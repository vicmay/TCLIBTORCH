# torch::conv_transpose3d

Applies a 3D transposed convolution operator over an input volume composed of several input planes. Also known as fractionally-strided convolution or deconvolution. This operation is commonly used for upsampling volumetric data in medical imaging, video processing, and 3D generative models.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::conv_transpose3d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?
```

### Named Parameter Syntax  
```tcl
torch::conv_transpose3d -input input_tensor -weight weight_tensor ?-bias bias_tensor? ?-stride stride_value? ?-padding padding_value? ?-output_padding output_padding_value? ?-groups groups_value? ?-dilation dilation_value?
```

### CamelCase Alias
```tcl
torch::convTranspose3d ?-outputPadding output_padding_value? ...
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | tensor | required | Input tensor of shape `(N, C_in, D_in, H_in, W_in)` |
| `weight` | tensor | required | Convolution kernel of shape `(C_in, C_out/groups, kD, kH, kW)` |
| `bias` | tensor | none | Optional bias tensor of shape `(C_out)` |
| `stride` | int or list | 1 | Stride of the transposed convolution. Single int or list of 3 ints `{d, h, w}` |
| `padding` | int or list | 0 | Padding added to all six sides. Single int or list of 3 ints `{d, h, w}` |
| `output_padding` | int or list | 0 | Additional size added to output shape. Single int or list of 3 ints `{d, h, w}` |
| `groups` | int | 1 | Number of blocked connections from input to output channels |
| `dilation` | int or list | 1 | Spacing between kernel elements. Single int or list of 3 ints `{d, h, w}` |

### Parameter Constraints
- `output_padding` must be smaller than either `stride` or `dilation`
- For grouped convolution: `C_in` and `C_out` must be divisible by `groups`
- Weight tensor shape: `(C_in, C_out/groups, kernel_depth, kernel_height, kernel_width)`

## Examples

### Basic 3D Transposed Convolution
```tcl
# Create input volume: batch=1, channels=16, depth=8, height=8, width=8
set input [torch::randn -shape {1 16 8 8 8}]

# Create weight: in_channels=16, out_channels=32, kernel=3x3x3
set weight [torch::randn -shape {16 32 3 3 3}]

# Basic transposed convolution
set output [torch::conv_transpose3d $input $weight]
# Output shape: {1, 32, 10, 10, 10}

# Using named parameters
set output [torch::conv_transpose3d -input $input -weight $weight]
```

### Upsampling with Stride
```tcl
# 2x upsampling in all dimensions
set input [torch::randn -shape {1 8 16 16 16}]
set weight [torch::randn -shape {8 4 4 4 4}]

set upsampled [torch::conv_transpose3d $input $weight none 2]
# Output shape: {1, 4, 35, 35, 35} (approximately 2x larger)

# Named parameter syntax
set upsampled [torch::conv_transpose3d -input $input -weight $weight -stride 2]
```

### Medical Image Super-Resolution
```tcl
# Super-resolution for CT scan volumes
set low_res_ct [torch::randn -shape {1 1 64 64 64}]
set sr_weight [torch::randn -shape {1 1 3 3 3}]

# 2x super-resolution with minimal padding
set high_res_ct [torch::conv_transpose3d $low_res_ct $sr_weight none 2 1]
# Output: approximately 128x128x128 volume

# Using camelCase alias
set high_res_ct [torch::convTranspose3d -input $low_res_ct -weight $sr_weight -stride 2 -padding 1]
```

### Video Upsampling with Different Strides
```tcl
# Video data: batch=2, channels=3, frames=16, height=112, width=112
set video [torch::randn -shape {2 3 16 112 112}]
set weight [torch::randn -shape {3 3 3 3 3}]

# Upsample temporally (depth) more than spatially
set upsampled_video [torch::conv_transpose3d $video $weight none {4 2 2} 1]

# Named parameter with list stride
set upsampled_video [torch::conv_transpose3d -input $video -weight $weight -stride {4 2 2} -padding 1]
```

### Grouped Transposed Convolution
```tcl
# Grouped convolution for efficiency
set input [torch::randn -shape {1 8 4 4 4}]
set weight [torch::randn -shape {8 4 2 2 2}]  # 8 input, 4 output per group

set result [torch::conv_transpose3d $input $weight none 1 0 0 2]
# groups=2: splits 8 input channels into 2 groups of 4

# Named parameter syntax
set result [torch::conv_transpose3d -input $input -weight $weight -groups 2]
```

### Dilated Transposed Convolution
```tcl
# Dilated kernels for larger receptive fields
set input [torch::randn -shape {1 4 8 8 8}]
set weight [torch::randn -shape {4 8 3 3 3}]

set dilated_result [torch::conv_transpose3d $input $weight none 1 0 0 1 2]
# dilation=2: effective kernel size becomes 5x5x5

# Named parameter syntax
set dilated_result [torch::conv_transpose3d -input $input -weight $weight -dilation 2]
```

### Output Padding for Precise Control
```tcl
# Fine-tune output dimensions with output padding
set input [torch::randn -shape {1 2 4 4 4}]
set weight [torch::randn -shape {2 4 2 2 2}]

set result1 [torch::conv_transpose3d $input $weight none 2 0 0]  # Standard
set result2 [torch::conv_transpose3d $input $weight none 2 0 1]  # +1 to each dimension

# output_padding must be < stride (2 in this case)
set result2 [torch::convTranspose3d -input $input -weight $weight -stride 2 -outputPadding 1]
```

## Output Shape Calculation

The output shape for each spatial dimension is calculated as:
```
output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
```

For 3D transposed convolution:
- **Depth**: `D_out = (D_in - 1) * stride_d - 2 * padding_d + kernel_d + output_padding_d`
- **Height**: `H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h` 
- **Width**: `W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w`

## Common Use Cases

### Medical Imaging
- **CT/MRI Super-Resolution**: Enhance low-resolution medical scans
- **3D Segmentation**: Upsample feature maps in U-Net architectures
- **Volume Reconstruction**: Reconstruct high-resolution 3D volumes

### Video Processing
- **Temporal Upsampling**: Increase frame rate in video sequences
- **Video Super-Resolution**: Enhance both spatial and temporal resolution
- **3D Action Recognition**: Upsample spatiotemporal features

### 3D Computer Vision
- **Point Cloud Processing**: Upsample 3D point cloud representations
- **3D Object Generation**: Generate high-resolution 3D objects
- **Volumetric Rendering**: Enhance 3D scene representations

### Scientific Computing
- **Fluid Dynamics**: Upsample computational fluid dynamics simulations
- **Climate Modeling**: Enhance resolution of atmospheric/oceanic data
- **Materials Science**: Process 3D microscopy data

## Error Handling

The command validates inputs and provides clear error messages:

```tcl
# Missing required parameters
torch::conv_transpose3d -weight $weight  
# Error: Required parameters 'input' and 'weight' are missing

# Invalid output padding (must be < stride)
torch::conv_transpose3d $input $weight none 1 0 2
# Error: output padding must be smaller than either stride or dilation

# Invalid tensor references
torch::conv_transpose3d invalid_tensor $weight
# Error: Invalid input tensor name

# Mismatched parameter pairs
torch::conv_transpose3d -input $input -weight
# Error: Named parameters must come in pairs
```

## Performance Considerations

### Memory Usage
3D transposed convolutions are memory-intensive operations:
- **Input**: `N × C_in × D × H × W`
- **Weight**: `C_in × C_out × kD × kH × kW`  
- **Output**: `N × C_out × D_out × H_out × W_out`

### Optimization Tips
1. **Use appropriate stride**: Larger strides reduce memory but may lose detail
2. **Consider groups**: Grouped convolutions reduce computational complexity
3. **Batch processing**: Process multiple volumes together for better GPU utilization
4. **Gradient checkpointing**: For very deep networks, trade compute for memory

## Backward Compatibility

The original positional syntax remains fully supported:
```tcl
# Legacy code continues to work
set result [torch::conv_transpose3d $input $weight $bias 2 1 1 1 1]

# Equivalent modern syntax
set result [torch::conv_transpose3d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -output_padding 1 -groups 1 -dilation 1]
```

## Migration Guide

### From Positional to Named Parameters
```tcl
# Before: Hard to read, parameter order matters
torch::conv_transpose3d $input $weight $bias 2 1 0 1 1

# After: Self-documenting, order-independent  
torch::conv_transpose3d -input $input -weight $weight -bias $bias -stride 2 -padding 1 -groups 1 -dilation 1
```

### From snake_case to camelCase
```tcl
# Old style
torch::conv_transpose3d -output_padding 1

# New style (both supported)
torch::convTranspose3d -outputPadding 1
```

## Related Commands

- [`torch::conv3d`](conv3d.md) - 3D convolution (forward operation)
- [`torch::conv_transpose1d`](conv_transpose1d.md) - 1D transposed convolution
- [`torch::conv_transpose2d`](conv_transpose2d.md) - 2D transposed convolution
- [`torch::tensor_shape`](../tensor_operations.md#tensor-shape) - Get tensor dimensions
- [`torch::randn`](../tensor_creation.md#randn) - Create random tensors

## Technical Notes

### Mathematical Background
Transposed convolution performs the mathematical transpose of the convolution operation. It's equivalent to:
1. Inserting zeros between input elements according to stride
2. Applying regular convolution with the transposed kernel
3. Adjusting output size with padding and output_padding

### Implementation Details
- Uses PyTorch's highly optimized `conv_transpose3d` backend
- Supports automatic differentiation for gradient computation
- Memory layout optimized for both CPU and GPU execution
- Handles edge cases and boundary conditions correctly

### Kernel Memory Layout
Weight tensors use the format `(C_in, C_out/groups, kD, kH, kW)` where:
- `C_in`: Input channels
- `C_out`: Output channels  
- `kD, kH, kW`: Kernel dimensions (depth, height, width)
- For grouped convolution: each group processes `C_in/groups` → `C_out/groups` channels 