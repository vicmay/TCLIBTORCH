# torch::hardswish / torch::hardSwish

Applies the hard swish function element-wise to the input tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::hardswish tensor
```

### Named Parameters (New Syntax)
```tcl
torch::hardswish -input tensor
torch::hardSwish -input tensor
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |

## Returns

Returns a string handle to the tensor containing the result of applying the hard swish function: `hardswish(x) = x * hardsigmoid(x)`.

## Mathematical Description

The hard swish function is defined as:
```
hardswish(x) = x * hardsigmoid(x) = x * max(0, min(1, (x + 3) / 6))
```

This can be written in piecewise form as:
- For x ≤ -3: hardswish(x) = 0
- For -3 < x < 3: hardswish(x) = x * (x + 3) / 6
- For x ≥ 3: hardswish(x) = x

The hard swish function is a computationally efficient alternative to the swish activation function (x * sigmoid(x)) while providing similar activation patterns. It's particularly popular in mobile and edge computing applications.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with various values
set input [torch::tensor_create -data {-4.0 -2.0 0.0 2.0 4.0} -dtype float32 -device cpu]

# Apply hard swish
set result [torch::hardswish $input]

# The result will be approximately {0.0, -0.33, 0.0, 1.67, 4.0}
```

### Named Parameters (New Syntax)
```tcl
# Create input tensor
set input [torch::full {3 3} 1.0]

# Apply hard swish with named parameters
set result [torch::hardswish -input $input]

# Check the result
puts "Shape: [torch::tensor_shape $result]"
puts "Values: [torch::tensor_item $result]"  # For scalar results
```

### CamelCase Alias
```tcl
# Using camelCase syntax
set input [torch::zeros {2 2}]
set result [torch::hardSwish $input]

# With named parameters
set result2 [torch::hardSwish -input $input]
```

### Mathematical Properties
```tcl
# Hard swish of 0 is always 0
set zero_tensor [torch::full {1} 0.0]
set result [torch::hardswish $zero_tensor]
set value [torch::tensor_item $result]
puts "hardswish(0) = $value"  # Should be 0.0

# Test saturation points
set large_neg [torch::full {1} -5.0]
set large_pos [torch::full {1} 5.0]
set neg_result [torch::hardswish $large_neg]
set pos_result [torch::hardswish $large_pos]
puts "hardswish(-5) = [torch::tensor_item $neg_result]"  # Should be ~0.0
puts "hardswish(5) = [torch::tensor_item $pos_result]"   # Should be ~5.0
```

### Neural Network Usage
```tcl
# Hard swish is commonly used as activation in modern architectures
set hidden_layer [torch::randn {32 128}]
set weights [torch::randn {128 256}]
set linear_output [torch::matmul $hidden_layer $weights]
set activated_output [torch::hardswish $linear_output]

# The activation provides smooth non-linearity with computational efficiency
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::hardswish invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Arguments
```tcl
catch {torch::hardswish} error
puts $error  # Usage information
```

### Unknown Parameters
```tcl
set input [torch::ones {2}]
catch {torch::hardswish -invalid_param $input} error
puts $error  # "Unknown parameter: -invalid_param"
```

## Data Type Support

Hard swish preserves the input tensor's data type:

```tcl
# Float32 input
set input_f32 [torch::full {2} 1.0]
set result_f32 [torch::hardswish $input_f32]
puts [torch::tensor_dtype $result_f32]  # Float

# Float64 input
set input_f64 [torch::full {2} 1.0 float64]
set result_f64 [torch::hardswish $input_f64]
puts [torch::tensor_dtype $result_f64]  # Double
```

**Note**: Integer tensors are not supported by the hard swish operation in LibTorch and will result in an error.

## Performance Characteristics

- **Computational Efficiency**: Hard swish is significantly faster than swish (x * sigmoid(x)) as it uses simple linear operations instead of exponentials
- **Memory Usage**: Same as input tensor
- **Gradient Computation**: Provides smooth gradients in the range (-3, 3), linear gradients outside

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::hardswish $input_tensor]
```

**After (Named Parameters):**
```tcl
set result [torch::hardswish -input $input_tensor]
```

### Using CamelCase
```tcl
# Snake_case (original)
set result [torch::hardswish $input_tensor]

# CamelCase (modern)
set result [torch::hardSwish $input_tensor]

# Both syntaxes work identically
```

## Comparison with Related Functions

| Function | Range | Computational Cost | Smoothness |
|----------|-------|-------------------|------------|
| `hardswish` | [0, ∞) for x≥0, [0, x] for x<0 | Low (linear) | Piecewise smooth |
| `swish` | (-∞, ∞) | High (exponential) | Smooth |
| `relu` | [0, ∞) | Low (linear) | Piecewise linear |
| `gelu` | (-∞, ∞) | High (exponential) | Smooth |

## Common Use Cases

1. **Mobile Networks**: Efficient activation in MobileNet architectures
2. **Edge Computing**: When computational efficiency is critical
3. **CNN Architectures**: Modern convolutional neural networks
4. **Approximate Swish**: When exact swish shape is not required
5. **Object Detection**: Used in efficient detection models like EfficientNet

## Implementation Notes

- The hard swish function is a piecewise linear approximation of the swish function
- It provides computational benefits while maintaining similar activation behavior to swish
- The function has a smooth transition around zero, making it suitable for gradient-based optimization
- Unlike ReLU, it doesn't have a hard cutoff at zero, allowing for better gradient flow

## Mathematical Properties

- **Monotonicity**: Hard swish is monotonically increasing
- **Boundedness**: For negative inputs, output is bounded between 0 and the input value
- **Smoothness**: Continuous and differentiable everywhere except at x = -3 and x = 3
- **Zero Point**: hardswish(0) = 0, maintaining the zero-preserving property

## See Also

- [`torch::hardSigmoid`](hardsigmoid.md) - Hard sigmoid activation function
- [`torch::swish`](swish.md) - Swish activation function (if available)
- [`torch::relu`](relu.md) - ReLU activation function
- [`torch::gelu`](gelu.md) - GELU activation function 