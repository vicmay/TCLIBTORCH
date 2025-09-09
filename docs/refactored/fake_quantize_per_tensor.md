# torch::fake_quantize_per_tensor

Applies fake quantization to a tensor using per-tensor quantization parameters.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::fake_quantize_per_tensor input scale zero_point ?quant_min? ?quant_max?

# Named parameter syntax (recommended)
torch::fake_quantize_per_tensor -input input -scale scale -zero_point zero_point ?-quant_min min? ?-quant_max max?

# CamelCase alias
torch::fakeQuantizePerTensor -input input -scale scale -zeroPoint zero_point ?-quantMin min? ?-quantMax max?
```

## Description

The `torch::fake_quantize_per_tensor` command applies fake quantization to the input tensor using a single scale and zero point for the entire tensor. This operation simulates the effect of quantizing and then immediately dequantizing the tensor, which is useful for training quantization-aware models.

Unlike per-channel quantization, per-tensor quantization uses the same quantization parameters (scale and zero point) for all elements in the tensor, making it computationally simpler but potentially less precise for tensors with varying dynamic ranges across channels.

## Parameters

### Required Parameters

- **input** (tensor): Input tensor to be fake quantized
- **scale** (double): Scaling factor for quantization
- **zero_point** (integer): Zero point for quantization

### Optional Parameters

- **quant_min** (integer, default: -128): Minimum quantized value
- **quant_max** (integer, default: 127): Maximum quantized value

### Parameter Details

#### input
The input tensor can be of any floating-point data type (float32, float64). The tensor can have any shape and number of dimensions.

#### scale
A floating-point value that determines the quantization resolution. Smaller scales result in finer quantization but smaller representable ranges.

#### zero_point
An integer value that represents the quantized value corresponding to the real value 0.0. Must be within the range [quant_min, quant_max].

#### quant_min/quant_max
Define the quantization range. Common ranges include:
- INT8: -128 to 127 (default)
- UINT8: 0 to 255
- Custom ranges for specific applications

## Return Value

Returns a tensor handle for the fake quantized tensor with the same shape and data type as the input.

## Examples

### Basic Per-Tensor Quantization

```tcl
# Create input tensor [2, 3]
set input [torch::tensorCreate -data {1.5 2.5 3.5 4.5 5.5 6.5} -shape {2 3} -dtype float32]

# Apply fake quantization with scale 0.1 and zero_point 0
set quantized [torch::fake_quantize_per_tensor $input 0.1 0]
```

### Named Parameter Syntax

```tcl
# Same operation using named parameters
set quantized [torch::fake_quantize_per_tensor \
    -input $input \
    -scale 0.1 \
    -zero_point 0 \
    -quant_min -128 \
    -quant_max 127]
```

### CamelCase Syntax

```tcl
# Using camelCase alias with camelCase parameter names
set quantized [torch::fakeQuantizePerTensor \
    -input $input \
    -scale 0.1 \
    -zeroPoint 0 \
    -quantMin -128 \
    -quantMax 127]
```

### Custom Quantization Ranges

```tcl
# UINT8 quantization (0 to 255)
set quantized_uint8 [torch::fake_quantize_per_tensor \
    -input $input \
    -scale 0.1 \
    -zero_point 128 \
    -quant_min 0 \
    -quant_max 255]

# Custom narrow range
set quantized_narrow [torch::fake_quantize_per_tensor \
    -input $input \
    -scale 0.1 \
    -zero_point 0 \
    -quant_min -1 \
    -quant_max 1]

# Wide range for high precision
set quantized_wide [torch::fake_quantize_per_tensor \
    -input $input \
    -scale 0.01 \
    -zero_point 0 \
    -quant_min -32768 \
    -quant_max 32767]
```

### Different Scales and Zero Points

```tcl
# Fine-grained quantization with small scale
set fine_quantized [torch::fake_quantize_per_tensor $input 0.01 0]

# Coarse quantization with large scale
set coarse_quantized [torch::fake_quantize_per_tensor $input 1.0 0]

# Shifted quantization with non-zero zero_point
set shifted_quantized [torch::fake_quantize_per_tensor $input 0.1 50]

# Asymmetric quantization for positive-only data
set positive_quantized [torch::fake_quantize_per_tensor $input 0.05 0 0 255]
```

### Working with Different Tensor Shapes

```tcl
# 1D tensor
set tensor_1d [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -shape {5} -dtype float32]
set quantized_1d [torch::fake_quantize_per_tensor $tensor_1d 0.1 0]

# 3D tensor
set tensor_3d [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]
set quantized_3d [torch::fake_quantize_per_tensor $tensor_3d 0.1 0]

# Large tensor
set large_tensor [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 4} -dtype float32]
set quantized_large [torch::fake_quantize_per_tensor $large_tensor 0.5 0]
```

## Mathematical Details

The fake quantization operation for per-tensor quantization follows these steps:

1. **Quantize**:
   ```
   quantized_value = round(input / scale) + zero_point
   quantized_value = clamp(quantized_value, quant_min, quant_max)
   ```

2. **Dequantize**:
   ```
   output = (quantized_value - zero_point) * scale
   ```

This process simulates the precision loss that would occur during actual quantization while maintaining floating-point precision for gradient computation during training.

### Scale Selection

The scale parameter determines the quantization step size:
- **Small scale**: Higher precision, smaller representable range
- **Large scale**: Lower precision, larger representable range

For a tensor with values in range [min_val, max_val], a suitable scale can be calculated as:
```
scale = (max_val - min_val) / (quant_max - quant_min)
```

### Zero Point Selection

The zero point should ideally map to a commonly occurring value (often 0.0):
```
zero_point = quant_min - round(min_val / scale)
```

## Error Handling

The command will return an error in the following cases:

- Invalid tensor handle for input
- Invalid data types for numeric parameters (scale, zero_point, quant_min, quant_max)
- Zero point outside the range [quant_min, quant_max]
- Missing required parameters
- Unknown parameters in named syntax

## Performance Considerations

- Per-tensor quantization is more computationally efficient than per-channel quantization
- Memory usage is constant regardless of tensor size
- The operation is element-wise and can be efficiently parallelized

## Use Cases

### Quantization-Aware Training

```tcl
# Apply fake quantization to simulate inference quantization
set conv_output [torch::conv2d $input $weight $bias]
set quantized_output [torch::fake_quantize_per_tensor $conv_output $scale $zero_point]
```

### Model Calibration

```tcl
# Find optimal quantization parameters
proc calibrate_quantization {tensor} {
    set min_val [torch::tensorItem [torch::tensorMin $tensor]]
    set max_val [torch::tensorItem [torch::tensorMax $tensor]]
    
    set scale [expr {($max_val - $min_val) / 255.0}]
    set zero_point [expr {int(-$min_val / $scale)}]
    
    return [list $scale $zero_point]
}

set input [torch::tensorCreate -data {-1.0 0.0 1.0 2.0} -shape {4} -dtype float32]
set params [calibrate_quantization $input]
set scale [lindex $params 0]
set zero_point [lindex $params 1]

set quantized [torch::fake_quantize_per_tensor $input $scale $zero_point 0 255]
```

### Comparing Quantization Strategies

```tcl
# Compare different quantization approaches
set original_tensor [torch::tensorCreate -data {1.1 2.2 3.3 4.4} -shape {4} -dtype float32]

# High precision quantization
set high_precision [torch::fake_quantize_per_tensor $original_tensor 0.01 0]

# Low precision quantization
set low_precision [torch::fake_quantize_per_tensor $original_tensor 0.1 0]

# Calculate quantization error
set error_high [torch::tensorSub $original_tensor $high_precision]
set error_low [torch::tensorSub $original_tensor $low_precision]

set mse_high [torch::tensorMean [torch::tensorMul $error_high $error_high]]
set mse_low [torch::tensorMean [torch::tensorMul $error_low $error_low]]

puts "High precision MSE: [torch::tensorItem $mse_high]"
puts "Low precision MSE: [torch::tensorItem $mse_low]"
```

## Migration from Legacy Syntax

### Old (Positional) Style
```tcl
set result [torch::fake_quantize_per_tensor $input 0.1 0 -128 127]
```

### New (Named Parameter) Style
```tcl
set result [torch::fake_quantize_per_tensor \
    -input $input \
    -scale 0.1 \
    -zero_point 0 \
    -quant_min -128 \
    -quant_max 127]
```

Both syntaxes are fully supported for backward compatibility.

## Comparison with Per-Channel Quantization

| Aspect | Per-Tensor | Per-Channel |
|--------|------------|-------------|
| **Precision** | Lower | Higher |
| **Computation** | Faster | Slower |
| **Memory** | Less | More |
| **Use Case** | General | Channel-sensitive data |

## Best Practices

1. **Scale Selection**: Use calibration data to determine optimal scales
2. **Zero Point**: Ensure zero point is within the quantization range
3. **Range Selection**: Choose quantization ranges based on your target hardware
4. **Validation**: Compare quantized and original outputs to assess quality
5. **Gradual Introduction**: Start with higher precision and reduce gradually

## Related Commands

- `torch::fake_quantize_per_channel` - Per-channel fake quantization
- `torch::quantize_per_tensor` - Actual per-tensor quantization
- `torch::dequantize` - Dequantization operation
- `torch::q_scale` - Get quantization scale
- `torch::q_zero_point` - Get quantization zero point

## See Also

- [Per-Channel Fake Quantization](fake_quantize_per_channel.md)
- [Quantization Operations](../quantization_operations.md)
- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html) 