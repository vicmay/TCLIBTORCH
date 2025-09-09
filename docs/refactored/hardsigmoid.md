# torch::hardsigmoid / torch::hardSigmoid

Applies the hard sigmoid function element-wise to the input tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::hardsigmoid tensor
```

### Named Parameters (New Syntax)
```tcl
torch::hardsigmoid -input tensor
torch::hardSigmoid -input tensor
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |

## Returns

Returns a string handle to the tensor containing the result of applying the hard sigmoid function: `hardsigmoid(x) = max(0, min(1, (x + 3) / 6))`.

## Mathematical Description

The hard sigmoid function is defined as:
```
hardsigmoid(x) = max(0, min(1, (x + 3) / 6))
```

This is a piecewise linear approximation of the sigmoid function:
- For x ≤ -3: hardsigmoid(x) = 0
- For -3 < x < 3: hardsigmoid(x) = (x + 3) / 6
- For x ≥ 3: hardsigmoid(x) = 1

The hard sigmoid function is computationally more efficient than the regular sigmoid function while providing a similar activation pattern.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with various values
set input [torch::tensor_create -data {-4.0 -2.0 0.0 2.0 4.0} -dtype float32 -device cpu]

# Apply hard sigmoid
set result [torch::hardsigmoid $input]

# The result will be approximately {0.0, 0.167, 0.5, 0.833, 1.0}
```

### Named Parameters (New Syntax)
```tcl
# Create input tensor
set input [torch::full {3 3} 1.0]

# Apply hard sigmoid with named parameters
set result [torch::hardsigmoid -input $input]

# Check the result
puts "Shape: [torch::tensor_shape $result]"
puts "Values: [torch::tensor_item $result]"  # For scalar results
```

### CamelCase Alias
```tcl
# Using camelCase syntax
set input [torch::zeros {2 2}]
set result [torch::hardSigmoid $input]

# With named parameters
set result2 [torch::hardSigmoid -input $input]
```

### Mathematical Properties
```tcl
# Hard sigmoid of 0 is always 0.5
set zero_tensor [torch::full {1} 0.0]
set result [torch::hardsigmoid $zero_tensor]
set value [torch::tensor_item $result]
puts "hardsigmoid(0) = $value"  # Should be 0.5

# Test saturation points
set large_neg [torch::full {1} -5.0]
set large_pos [torch::full {1} 5.0]
set neg_result [torch::hardsigmoid $large_neg]
set pos_result [torch::hardsigmoid $large_pos]
puts "hardsigmoid(-5) = [torch::tensor_item $neg_result]"  # Should be ~0.0
puts "hardsigmoid(5) = [torch::tensor_item $pos_result]"   # Should be ~1.0
```

### Neural Network Usage
```tcl
# Hard sigmoid is often used as gate activation in LSTM/GRU
set hidden_state [torch::randn {32 128}]
set gate_weights [torch::randn {128 128}]
set gate_input [torch::matmul $hidden_state $gate_weights]
set gate_activation [torch::hardsigmoid $gate_input]

# The gate activation can now be used to control information flow
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::hardsigmoid invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Arguments
```tcl
catch {torch::hardsigmoid} error
puts $error  # Usage information
```

### Unknown Parameters
```tcl
set input [torch::ones {2}]
catch {torch::hardsigmoid -invalid_param $input} error
puts $error  # "Unknown parameter: -invalid_param"
```

## Data Type Support

Hard sigmoid preserves the input tensor's data type:

```tcl
# Float32 input
set input_f32 [torch::full {2} 1.0]
set result_f32 [torch::hardsigmoid $input_f32]
puts [torch::tensor_dtype $result_f32]  # Float

# Float64 input
set input_f64 [torch::full {2} 1.0 float64]
set result_f64 [torch::hardsigmoid $input_f64]
puts [torch::tensor_dtype $result_f64]  # Double
```

**Note**: Integer tensors are not supported by the hard sigmoid operation in LibTorch and will result in an error.

## Performance Characteristics

- **Computational Efficiency**: Hard sigmoid is significantly faster than regular sigmoid as it uses simple linear operations instead of exponentials
- **Memory Usage**: Same as input tensor
- **Gradient Computation**: Provides non-zero gradients in the range (-3, 3), zero gradients outside

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::hardsigmoid $input_tensor]
```

**After (Named Parameters):**
```tcl
set result [torch::hardsigmoid -input $input_tensor]
```

### Using CamelCase
```tcl
# Snake_case (original)
set result [torch::hardsigmoid $input_tensor]

# CamelCase (modern)
set result [torch::hardSigmoid $input_tensor]

# Both syntaxes work identically
```

## Comparison with Related Functions

| Function | Range | Computational Cost | Smoothness |
|----------|-------|-------------------|------------|
| `hardsigmoid` | [0, 1] | Low (linear) | Piecewise linear |
| `sigmoid` | (0, 1) | High (exponential) | Smooth |
| `tanh` | (-1, 1) | High (exponential) | Smooth |
| `relu` | [0, ∞) | Low (linear) | Piecewise linear |

## Common Use Cases

1. **LSTM/GRU Gates**: Efficient gate activation in recurrent networks
2. **Mobile/Edge Computing**: When computational efficiency is critical
3. **Approximate Sigmoid**: When exact sigmoid shape is not required
4. **Binary Classification**: As an output activation for binary tasks

## Implementation Notes

- The hard sigmoid function is a piecewise linear approximation of the sigmoid function
- It provides computational benefits while maintaining similar activation behavior
- The function saturates at 0 and 1, making it suitable for gate mechanisms
- Gradients are constant (1/6) in the linear region and zero in the saturated regions

## See Also

- [`torch::sigmoid`](sigmoid.md) - Standard sigmoid activation function
- [`torch::hardtanh`](hardtanh.md) - Hard hyperbolic tangent function
- [`torch::relu`](relu.md) - Rectified Linear Unit activation
- [`torch::gelu`](gelu.md) - Gaussian Error Linear Unit activation 