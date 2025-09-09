# torch::quantized_relu

Applies the Rectified Linear Unit (ReLU) function to a quantized tensor. ReLU replaces all negative values with zero while keeping positive values unchanged.

## Syntax

### Current Syntax (Positional Parameters)
```tcl
torch::quantized_relu quantized_tensor
```

### New Syntax (Named Parameters)  
```tcl
torch::quantized_relu -input TENSOR
```

### CamelCase Alias
```tcl  
torch::quantizedRelu quantized_tensor
torch::quantizedRelu -input TENSOR
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `quantized_tensor` (positional) | string | Yes | Handle to the input quantized tensor |
| `-input` | string | Yes | Handle to the input quantized tensor |

## Returns

Returns a handle to the new tensor containing the result of applying ReLU to the input quantized tensor.

## Description

The `torch::quantized_relu` function applies the ReLU activation function to a quantized tensor:
- All negative values become 0
- All positive values remain unchanged  
- Zero values remain zero

This function is optimized for quantized tensors and maintains the quantization properties of the input tensor.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create and quantize a tensor
set tensor [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]
set quantized [torch::quantize_per_tensor $tensor 0.1 128 quint8]

# Apply quantized ReLU using positional syntax
set result [torch::quantized_relu $quantized]
puts "Quantized ReLU result: $result"
```

### Named Parameters Syntax
```tcl
# Create and quantize a tensor  
set tensor [torch::tensorCreate -data {-3.0 -1.5 0.0 1.5 3.0} -dtype float32]
set quantized [torch::quantize_per_tensor $tensor 0.1 128 quint8]

# Apply quantized ReLU using named parameters
set result [torch::quantized_relu -input $quantized]
puts "Quantized ReLU result: $result"
```

### CamelCase Alias
```tcl
# Using camelCase alias with positional syntax
set quantized [torch::quantize_per_tensor $input_tensor 0.05 64 quint8]
set result [torch::quantizedRelu $quantized]

# Using camelCase alias with named parameters
set result [torch::quantizedRelu -input $quantized]
```

### Integration with Neural Networks
```tcl
# Quantized neural network layer with ReLU activation
proc quantized_linear_relu {input weights bias scale zero_point} {
    # Quantized linear transformation
    set linear_out [torch::quantized_linear $input $weights $bias]
    
    # Apply quantized ReLU activation
    set relu_out [torch::quantized_relu -input $linear_out]
    
    return $relu_out
}

# Usage
set input [torch::quantize_per_tensor $input_tensor 0.1 128 quint8]
set output [quantized_linear_relu $input $weights $bias 0.1 128]
```

## Mathematical Properties

For a quantized tensor Q with values q:
- If q < 0: result = 0
- If q ≥ 0: result = q

The quantization parameters (scale and zero_point) are preserved in the output tensor.

## Performance Notes

- `torch::quantized_relu` is optimized for quantized tensors and can be faster than regular ReLU for quantized computations
- The function maintains the quantization scheme of the input tensor
- Memory usage is minimal as the operation can be performed in-place in many cases

## Error Handling

The function will raise an error if:
- No input tensor is provided
- The provided tensor handle is invalid
- Unknown parameters are specified (named syntax only)

### Error Examples
```tcl
# Error: Missing tensor
catch {torch::quantized_relu} error
puts "Error: $error"
# Output: Usage: torch::quantized_relu quantized_tensor
#         or: torch::quantized_relu -input TENSOR

# Error: Invalid tensor
catch {torch::quantized_relu "invalid_handle"} error  
puts "Error: $error"
# Output: Invalid quantized tensor

# Error: Unknown parameter  
catch {torch::quantized_relu -invalid_param $tensor} error
puts "Error: $error"  
# Output: Unknown parameter: -invalid_param
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::quantized_relu $quantized_tensor]

# New named parameter syntax  
set result [torch::quantized_relu -input $quantized_tensor]

# Both syntaxes work - choose based on preference
```

### Adopting CamelCase

```tcl
# Traditional snake_case
set result [torch::quantized_relu $quantized_tensor]

# Modern camelCase alias
set result [torch::quantizedRelu $quantized_tensor]

# Both work identically
```

## Compatibility

- ✅ **Backward Compatible**: All existing code using positional syntax continues to work
- ✅ **New Features**: Named parameters provide better readability
- ✅ **Modern Style**: CamelCase aliases follow modern TCL conventions
- ✅ **Performance**: No performance penalty for using either syntax

## See Also

- [`torch::relu`](relu.md) - Regular ReLU function for non-quantized tensors
- [`torch::quantize_per_tensor`](quantize_per_tensor.md) - Quantize tensors
- [`torch::quantized_add`](quantized_add.md) - Quantized tensor addition
- [`torch::quantized_mul`](quantized_mul.md) - Quantized tensor multiplication
- [Quantization Operations Guide](../guides/quantization.md) - Complete quantization workflow

## Version History

- **v2.0**: Added dual syntax support (named parameters + camelCase aliases)
- **v1.0**: Initial implementation with positional parameters 