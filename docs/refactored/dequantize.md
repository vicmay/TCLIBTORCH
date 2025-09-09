# torch::dequantize

Dequantizes a quantized tensor, converting it back to a floating-point representation.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::dequantize quantized_tensor
```

### Named Parameter Syntax  
```tcl
torch::dequantize -input quantized_tensor
```

### CamelCase Alias
```tcl
torch::deQuantize quantized_tensor
torch::deQuantize -input quantized_tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `quantized_tensor` / `-input` | string | Yes | Handle to a quantized tensor |

## Return Value

Returns a string handle to the dequantized tensor with floating-point data type.

## Description

The `torch::dequantize` command converts a quantized tensor back to its floating-point representation. This operation undoes the effect of quantization operations like `torch::quantize_per_tensor` or `torch::quantize_per_channel`.

### Key Features:
- **Lossless for supported quantization schemes**: When used with compatible quantization operations
- **Shape preservation**: The output tensor has the same shape as the input quantized tensor
- **Data type conversion**: Converts quantized integer types back to floating-point (typically Float32)
- **Dual syntax support**: Supports both positional and named parameter syntax
- **CamelCase alias**: Available as `torch::deQuantize` for modern naming conventions

### Quantization Context

Dequantization is the reverse operation of quantization. In the quantization workflow:

1. **Quantization**: `float_tensor → quantized_tensor` (lossy compression)
2. **Computation**: Operations on quantized tensors (memory/speed benefits)  
3. **Dequantization**: `quantized_tensor → float_tensor` (back to full precision)

## Examples

### Basic Usage

```tcl
# Assuming you have a quantized tensor from a previous quantization operation
set quantized_tensor $some_quantized_tensor

# Positional syntax
set float_result [torch::dequantize $quantized_tensor]

# Named parameter syntax
set float_result [torch::dequantize -input $quantized_tensor]

# CamelCase alias
set float_result [torch::deQuantize $quantized_tensor]
```

### Complete Quantization/Dequantization Workflow

```tcl
# Create a float tensor
set original [torch::tensor_create -data {1.0 2.5 3.7 4.2} -shape {2 2} -dtype float32]

# Quantize with scale=0.1, zero_point=0, dtype=int8
set quantized [torch::quantize_per_tensor $original 0.1 0 int8]

# Perform operations on quantized tensor (memory efficient)
# ... quantized operations ...

# Dequantize back to float for final result
set dequantized [torch::dequantize $quantized]

# The dequantized tensor should be approximately equal to the original
# (with some quantization error)
```

### Integration with Neural Networks

```tcl
# In a quantized neural network inference pipeline
proc quantized_inference {input_data model_weights} {
    # Load quantized model weights
    set quantized_weights $model_weights
    
    # Quantize input data
    set quantized_input [torch::quantize_per_tensor $input_data 0.05 0 int8]
    
    # Perform quantized operations (fast and memory efficient)
    set quantized_output [quantized_forward_pass $quantized_input $quantized_weights]
    
    # Dequantize final output for interpretation
    set float_output [torch::dequantize $quantized_output]
    
    return $float_output
}
```

### Batch Processing

```tcl
# Process multiple quantized tensors
set quantized_batch [list $q_tensor1 $q_tensor2 $q_tensor3]
set dequantized_batch {}

foreach qtensor $quantized_batch {
    set float_tensor [torch::dequantize $qtensor]
    lappend dequantized_batch $float_tensor
}
```

## Technical Details

### Supported Input Types
- Quantized tensors created by `torch::quantize_per_tensor`
- Quantized tensors created by `torch::quantize_per_channel`
- Any tensor with quantized data type (QInt8, QUInt8, QInt32)

### Output Characteristics
- **Data Type**: Always floating-point (typically Float32)
- **Shape**: Identical to input quantized tensor
- **Values**: Reconstructed floating-point values based on quantization parameters

### Mathematical Operation

For per-tensor quantization, dequantization follows:
```
dequantized_value = scale × (quantized_value - zero_point)
```

Where:
- `scale`: The quantization scale factor
- `zero_point`: The quantization zero point
- `quantized_value`: The integer value in the quantized tensor

### Performance Considerations

- **Memory**: Converts from compact quantized format back to full floating-point
- **Speed**: Fast operation, but results in larger memory footprint
- **Precision**: Some precision loss may have occurred during the original quantization
- **Use Cases**: Typically used at the end of quantized computation pipelines

## Error Handling

The command provides comprehensive error checking:

```tcl
# Invalid tensor handle
catch {torch::dequantize invalid_tensor} error
# Error: "Invalid quantized tensor"

# Missing parameters
catch {torch::dequantize} error  
# Error: Usage message

# Invalid parameter names  
catch {torch::dequantize -wrong_param $tensor} error
# Error: "Unknown parameter: -wrong_param"

# Missing parameter values
catch {torch::dequantize -input} error
# Error: "Missing value for parameter"
```

## Comparison with Related Operations

| Operation | Purpose | Input Type | Output Type |
|-----------|---------|------------|-------------|
| `torch::quantize_per_tensor` | Float → Quantized | Float tensor | Quantized tensor |
| `torch::dequantize` | Quantized → Float | Quantized tensor | Float tensor |
| `torch::int_repr` | Get quantized values | Quantized tensor | Integer tensor |
| `torch::q_scale` | Get quantization scale | Quantized tensor | Scalar value |

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::dequantize $quantized_tensor]

# New named parameter syntax
set result [torch::dequantize -input $quantized_tensor]

# CamelCase alias (recommended for new code)
set result [torch::deQuantize -input $quantized_tensor]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter purpose is clear
- **Extensible**: Future parameters can be added easily
- **Consistent**: Matches modern API design patterns
- **Error-resistant**: Less prone to parameter order mistakes

## See Also

- [torch::quantize_per_tensor](quantize_per_tensor.md) - Quantize tensors per-tensor
- [torch::quantize_per_channel](quantize_per_channel.md) - Quantize tensors per-channel  
- [torch::int_repr](int_repr.md) - Get integer representation of quantized tensors
- [torch::q_scale](q_scale.md) - Get quantization scale
- [torch::q_zero_point](q_zero_point.md) - Get quantization zero point 