# torch::conv3d

## Overview

The `torch::conv3d` command performs 3D convolution operations on input tensors. This is commonly used for processing 3D data such as volumetric medical images, video sequences, or any data with spatial-temporal dimensions. The operation applies 3D filters (kernels) across the depth, height, and width dimensions of the input.

## Syntax

### Legacy Syntax (Backward Compatible)
```tcl
torch::conv3d input weight ?bias? ?stride? ?padding? ?dilation? ?groups?
```

### Modern Syntax (Named Parameters) 
```tcl
torch::conv3d -input input -weight weight ?-bias bias? ?-stride stride? ?-padding padding? ?-dilation dilation? ?-groups groups?
```

### CamelCase (Already camelCase)
The command name `conv3d` is already in camelCase format, so no separate alias is needed.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | - | Handle of the input tensor |
| `weight` | string | Yes | - | Handle of the weight tensor (kernel) |
| `bias` | string | No | none | Handle of the bias tensor (or "none") |
| `stride` | int or list | No | {1,1,1} | Stride for the convolution |
| `padding` | int or list | No | {0,0,0} | Padding applied to input |
| `dilation` | int or list | No | {1,1,1} | Dilation rate for the convolution |
| `groups` | int | No | 1 | Number of blocked connections |

## Parameter Details

### Input Tensor
- **Shape**: `[batch_size, in_channels, depth, height, width]`
- **Data Type**: Any floating-point type supported by LibTorch
- **Description**: The 5D input tensor to be convolved

### Weight Tensor  
- **Shape**: `[out_channels, in_channels/groups, kernel_depth, kernel_height, kernel_width]`
- **Data Type**: Same as input tensor
- **Description**: The 5D convolution kernel

### Bias Tensor (Optional)
- **Shape**: `[out_channels]`
- **Data Type**: Same as input tensor
- **Description**: Bias values added to each output channel
- **Special Value**: Use "none" or omit to skip bias

### Stride, Padding, Dilation
- **Single Integer**: Applied to all three spatial dimensions (depth, height, width)
- **List of 3 Integers**: `{depth_value, height_value, width_value}`
- **Description**: Controls the convolution operation behavior

### Groups
- **Type**: Integer ≥ 1
- **Description**: Controls the connections between inputs and outputs
- **Groups = 1**: Standard convolution (all inputs connected to all outputs)
- **Groups > 1**: Grouped convolution for efficiency

## Return Value

Returns a handle to the output tensor with shape:
`[batch_size, out_channels, out_depth, out_height, out_width]`

Where output spatial dimensions depend on input size, kernel size, stride, padding, and dilation.

## Examples

### Basic 3D Convolution

```tcl
# Create input tensor: batch=1, channels=3, depth=8, height=16, width=16
set input [torch::randn -shape {1 3 8 16 16}]

# Create 3D kernel: out_channels=64, in_channels=3, kernel=3x3x3  
set weight [torch::randn -shape {64 3 3 3 3}]

# Basic convolution using legacy syntax
set output1 [torch::conv3d $input $weight]

# Same operation using modern named syntax
set output2 [torch::conv3d -input $input -weight $weight]

puts "Output shape: [torch::tensor_shape $output1]"
# Output: 1 64 6 14 14 (with default stride=1, padding=0)
```

### With Bias and Custom Parameters

```tcl
# Create tensors
set input [torch::randn -shape {2 32 16 32 32}]
set weight [torch::randn -shape {64 32 5 5 5}]
set bias [torch::randn -shape {64}]

# Using legacy syntax with all parameters
set output1 [torch::conv3d $input $weight $bias 2 {2 1 2} {1 1 1} 1]

# Using modern named syntax
set output2 [torch::conv3d \
    -input $input \
    -weight $weight \
    -bias $bias \
    -stride 2 \
    -padding {2 1 2} \
    -dilation {1 1 1} \
    -groups 1]

puts "Output shape: [torch::tensor_shape $output1]"
```

### Video Processing Example

```tcl
# Process a video sequence: batch=1, channels=3 (RGB), frames=30, height=224, width=224
set video [torch::randn -shape {1 3 30 224 224}]

# Create 3D kernel for temporal-spatial feature extraction
# 8 output channels, 3 input channels, 3 frames, 7x7 spatial kernel
set kernel [torch::randn -shape {8 3 3 7 7}]

# Apply 3D convolution with padding to maintain temporal dimension
set features [torch::conv3d \
    -input $video \
    -weight $kernel \
    -stride {1 2 2} \
    -padding {1 3 3}]

puts "Feature map shape: [torch::tensor_shape $features]"
# Output: 1 8 30 112 112
```

### Medical Volume Analysis

```tcl
# Medical CT scan: batch=1, channels=1, depth=64, height=512, width=512
set ct_scan [torch::randn -shape {1 1 64 512 512}]

# 3D feature detector for medical analysis
set detector [torch::randn -shape {32 1 3 3 3}]

# Apply convolution with padding to preserve spatial information
set features [torch::conv3d \
    -input $ct_scan \
    -weight $detector \
    -padding 1 \
    -stride 1]

puts "Medical features shape: [torch::tensor_shape $features]"
# Output: 1 32 64 512 512
```

### Grouped Convolution

```tcl
# Input with 32 channels
set input [torch::randn -shape {1 32 8 16 16}]

# Grouped convolution: 4 groups, each group processes 8 input channels
set weight [torch::randn -shape {64 8 3 3 3}]

# Apply grouped convolution (groups=4)
set output [torch::conv3d \
    -input $input \
    -weight $weight \
    -groups 4]

puts "Grouped convolution output: [torch::tensor_shape $output]"
# Output: 1 64 6 14 14
```

### Dilated Convolution

```tcl
# Input tensor
set input [torch::randn -shape {1 16 16 32 32}]

# Weight tensor  
set weight [torch::randn -shape {32 16 3 3 3}]

# Dilated convolution increases receptive field without increasing parameters
set output [torch::conv3d \
    -input $input \
    -weight $weight \
    -dilation {2 2 2} \
    -padding {2 2 2}]

puts "Dilated convolution output: [torch::tensor_shape $output]"
```

## Error Handling

### Common Errors

```tcl
# Missing required parameters
catch {torch::conv3d -input $input} error
puts "Error: $error"  ;# Output: Required parameters: input and weight

# Invalid tensor handles
catch {torch::conv3d -input "invalid" -weight $weight} error  
puts "Error: $error"  ;# Output: Invalid input tensor name

# Invalid parameter values
catch {torch::conv3d -input $input -weight $weight -stride {1 2}} error
puts "Error: $error"  ;# Output: Value must be int or list of 3 ints

# Unknown parameters
catch {torch::conv3d -input $input -weight $weight -unknown_param 5} error
puts "Error: $error"  ;# Output: Unknown parameter: -unknown_param
```

### Dimension Validation

```tcl
# Input and weight dimension mismatch will cause LibTorch error
set input [torch::randn -shape {1 3 8 16 16}]      # 3 input channels
set weight [torch::randn -shape {64 5 3 3 3}]      # Expects 5 input channels

catch {torch::conv3d -input $input -weight $weight} error
puts "LibTorch error: $error"  ;# Dimension mismatch error
```

## Mathematical Details

### Output Size Calculation

For each spatial dimension:
```
output_size = (input_size + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1
```

### Example Calculation
```tcl
# Input: [1, 16, 8, 32, 32]
# Kernel: [32, 16, 3, 5, 5] 
# Stride: {1, 2, 2}
# Padding: {1, 2, 2}
# Dilation: {1, 1, 1}

# Output depth:  (8 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 8
# Output height: (32 + 2*2 - 1*(5-1) - 1) / 2 + 1 = 16  
# Output width:  (32 + 2*2 - 1*(5-1) - 1) / 2 + 1 = 16

# Result shape: [1, 32, 8, 16, 16]
```

## Performance Considerations

### Memory Usage
- **Input size**: Large 3D tensors can consume significant memory
- **Kernel size**: Larger kernels require more computation and memory
- **Batch size**: Multiple samples increase memory linearly

### Optimization Tips
```tcl
# 1. Use appropriate data types
set input [torch::randn -shape {1 3 8 16 16} -dtype float32]  # vs float64

# 2. Consider grouped convolutions for efficiency
set output [torch::conv3d -input $input -weight $weight -groups 4]

# 3. Use smaller kernel sizes when possible
set efficient_kernel [torch::randn -shape {64 32 3 3 3}]     # 3x3x3
# vs
set large_kernel [torch::randn -shape {64 32 7 7 7}]         # 7x7x7
```

## Integration with Neural Networks

### Custom 3D CNN Layer

```tcl
proc create_conv3d_layer {in_channels out_channels kernel_size stride padding} {
    # Create weight with proper initialization
    set fan_in [expr {$in_channels * $kernel_size * $kernel_size * $kernel_size}]
    set std [expr {sqrt(2.0 / $fan_in)}]
    
    set weight [torch::randn -shape [list $out_channels $in_channels $kernel_size $kernel_size $kernel_size]]
    set weight [torch::tensor_mul $weight $std]
    
    set bias [torch::zeros -shape [list $out_channels]]
    
    return [list $weight $bias]
}

# Usage
set layer_params [create_conv3d_layer 32 64 3 1 1]
set weight [lindex $layer_params 0]
set bias [lindex $layer_params 1]

set output [torch::conv3d -input $input -weight $weight -bias $bias -stride 1 -padding 1]
```

## Compatibility

- **Backward Compatibility**: ✅ Full support for legacy positional syntax
- **Named Parameters**: ✅ Modern `-parameter value` syntax supported  
- **Parameter Types**: ✅ Supports both integer and list formats for spatial parameters
- **Parameter Order**: ✅ Named parameters can be specified in any order

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# BEFORE (Legacy - still works)
torch::conv3d $input $weight $bias {2 1 2} {1 0 1} {1 1 1} 2

# AFTER (Modern - recommended for new code)  
torch::conv3d -input $input -weight $weight -bias $bias \
              -stride {2 1 2} -padding {1 0 1} -dilation {1 1 1} -groups 2
```

### Benefits of Modern Syntax
- **Parameter Clarity**: Explicit parameter names improve code readability
- **Flexible Ordering**: Parameters can be specified in any order
- **Optional Parameters**: Easy to specify only needed parameters
- **Better Documentation**: Self-documenting code with parameter names
- **Error Prevention**: Less likely to mix up parameter positions

## See Also

- [`torch::conv1d`](conv1d.md) - 1D convolution operations
- [`torch::conv2d`](conv2d.md) - 2D convolution operations  
- [`torch::conv_transpose3d`](conv_transpose3d.md) - 3D transposed convolution
- [`torch::randn`](randn.md) - Generate random tensors
- [`torch::tensor_shape`](tensor_shape.md) - Get tensor dimensions

## Technical Notes

- Uses LibTorch's `torch::conv3d` function internally
- Supports autograd for backpropagation in training
- Compatible with CUDA tensors for GPU acceleration
- Thread-safe operation
- Memory-efficient implementation with proper tensor management 