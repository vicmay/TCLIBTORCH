# torch::fake_quantize_per_channel

Applies fake quantization to a tensor using per-channel quantization parameters.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::fake_quantize_per_channel input scales zero_points axis ?quant_min? ?quant_max?

# Named parameter syntax (recommended)
torch::fake_quantize_per_channel -input input -scales scales -zero_points zero_points -axis axis ?-quant_min min? ?-quant_max max?

# CamelCase alias
torch::fakeQuantizePerChannel -input input -scales scales -zeroPoints zero_points -axis axis ?-quantMin min? ?-quantMax max?
```

## Description

The `torch::fake_quantize_per_channel` command applies fake quantization to the input tensor using per-channel scales and zero points. This operation simulates the effect of quantizing and then immediately dequantizing the tensor, which is useful for training quantization-aware models.

Unlike per-tensor quantization, per-channel quantization applies different quantization parameters (scale and zero point) to each channel of the tensor along the specified axis, allowing for better preservation of the tensor's dynamic range.

## Parameters

### Required Parameters

- **input** (tensor): Input tensor to be fake quantized
- **scales** (tensor): Per-channel scaling factors for quantization
- **zero_points** (tensor): Per-channel zero points for quantization  
- **axis** (integer): Dimension along which to apply per-channel quantization

### Optional Parameters

- **quant_min** (integer, default: -128): Minimum quantized value
- **quant_max** (integer, default: 127): Maximum quantized value

### Parameter Details

#### input
The input tensor can be of any floating-point data type (float32, float64). The tensor can have any number of dimensions.

#### scales
A 1D tensor containing the scaling factors for each channel. The size of this tensor must match the size of the input tensor along the specified axis.

#### zero_points  
A 1D tensor containing the zero point values for each channel. The size must match the scales tensor. Zero points must be integers within the range [quant_min, quant_max].

#### axis
The dimension along which to apply per-channel quantization. Must be a valid dimension index for the input tensor.

#### quant_min/quant_max
Define the quantization range. Common ranges include:
- INT8: -128 to 127 (default)
- UINT8: 0 to 255
- Custom ranges for specific applications

## Return Value

Returns a tensor handle for the fake quantized tensor with the same shape and data type as the input.

## Examples

### Basic Per-Channel Quantization

```tcl
# Create input tensor [2, 3]
set input [torch::tensorCreate -data {1.5 2.5 3.5 4.5 5.5 6.5} -shape {2 3} -dtype float32]

# Create per-channel scales for axis 1 (3 channels)
set scales [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]

# Create per-channel zero points
set zero_points [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]

# Apply fake quantization along axis 1
set quantized [torch::fake_quantize_per_channel $input $scales $zero_points 1]
```

### Named Parameter Syntax

```tcl
# Same operation using named parameters
set quantized [torch::fake_quantize_per_channel \
    -input $input \
    -scales $scales \
    -zero_points $zero_points \
    -axis 1 \
    -quant_min -128 \
    -quant_max 127]
```

### CamelCase Syntax

```tcl
# Using camelCase alias with camelCase parameter names
set quantized [torch::fakeQuantizePerChannel \
    -input $input \
    -scales $scales \
    -zeroPoints $zero_points \
    -axis 1 \
    -quantMin -128 \
    -quantMax 127]
```

### Different Quantization Axis

```tcl
# Create 3D tensor [2, 4, 3]
set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0} -shape {2 4 3} -dtype float32]

# Quantize along axis 1 (4 channels)
set scales [torch::tensorCreate -data {0.1 0.2 0.3 0.4} -shape {4} -dtype float32]
set zero_points [torch::tensorCreate -data {0 1 2 3} -shape {4} -dtype int32]
set quantized [torch::fake_quantize_per_channel $input $scales $zero_points 1]

# Quantize along axis 2 (3 channels)
set scales_axis2 [torch::tensorCreate -data {0.1 0.2 0.3} -shape {3} -dtype float32]
set zero_points_axis2 [torch::tensorCreate -data {0 1 2} -shape {3} -dtype int32]
set quantized_axis2 [torch::fake_quantize_per_channel $input $scales_axis2 $zero_points_axis2 2]
```

### Custom Quantization Range

```tcl
# UINT8 quantization (0 to 255)
set quantized_uint8 [torch::fake_quantize_per_channel \
    -input $input \
    -scales $scales \
    -zero_points $zero_points \
    -axis 1 \
    -quant_min 0 \
    -quant_max 255]

# Custom narrow range
set quantized_narrow [torch::fake_quantize_per_channel \
    -input $input \
    -scales $scales \
    -zero_points [torch::tensorCreate -data {-1 0 1} -shape {3} -dtype int32] \
    -axis 1 \
    -quant_min -1 \
    -quant_max 1]
```

## Mathematical Details

The fake quantization operation for per-channel quantization follows these steps:

1. **Quantize**: For each channel i along the specified axis:
   ```
   quantized_value[i] = round(input[i] / scale[i]) + zero_point[i]
   quantized_value[i] = clamp(quantized_value[i], quant_min, quant_max)
   ```

2. **Dequantize**: 
   ```
   output[i] = (quantized_value[i] - zero_point[i]) * scale[i]
   ```

This process simulates the precision loss that would occur during actual quantization while maintaining floating-point precision for gradient computation.

## Error Handling

The command will return an error in the following cases:

- Invalid tensor handles for input, scales, or zero_points
- Mismatched dimensions between scales/zero_points and the input tensor along the specified axis
- Invalid axis value (must be within tensor dimensions)
- Invalid data types for numeric parameters
- Zero points outside the range [quant_min, quant_max]
- Missing required parameters

## Performance Considerations

- Per-channel quantization is more computationally expensive than per-tensor quantization
- Memory usage scales with the number of channels
- Consider the trade-off between accuracy and performance for your specific use case

## Use Cases

### Quantization-Aware Training
```tcl
# Simulate quantization effects during training
set conv_output [torch::conv2d $input $weight $bias]
set quantized_output [torch::fake_quantize_per_channel $conv_output $scales $zero_points 1]
```

### Model Optimization Research
```tcl
# Compare different quantization strategies
set per_tensor_quantized [torch::fake_quantize_per_tensor $input $scale $zero_point]
set per_channel_quantized [torch::fake_quantize_per_channel $input $scales $zero_points 1]

# Analyze quantization error
set error [torch::tensorSub $input $per_channel_quantized]
set mse [torch::tensorMean [torch::tensorMul $error $error]]
```

## Migration from Legacy Syntax

### Old (Positional) Style
```tcl
set result [torch::fake_quantize_per_channel $input $scales $zero_points 1 -128 127]
```

### New (Named Parameter) Style
```tcl
set result [torch::fake_quantize_per_channel \
    -input $input \
    -scales $scales \
    -zero_points $zero_points \
    -axis 1 \
    -quant_min -128 \
    -quant_max 127]
```

Both syntaxes are fully supported for backward compatibility.

## Related Commands

- `torch::fake_quantize_per_tensor` - Per-tensor fake quantization
- `torch::quantize_per_channel` - Actual per-channel quantization  
- `torch::dequantize` - Dequantization operation
- `torch::q_scale` - Get quantization scale
- `torch::q_zero_point` - Get quantization zero point

## See Also

- [Quantization Operations](../quantization_operations.md)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) 