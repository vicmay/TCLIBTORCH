# torch::tensor_tanh

Element-wise hyperbolic tangent (tanh) activation function for tensors.

## Syntax

### Positional Syntax (Original)
```tcl
torch::tensor_tanh tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_tanh -input tensor
```

### CamelCase Alias
```tcl
torch::tensorTanh tensor
torch::tensorTanh -input tensor
```

## Parameters

### Required Parameters
- **input** (`string`): Handle to the input tensor

## Return Value

Returns a string handle to a new tensor containing the element-wise hyperbolic tangent (tanh) activation of the input tensor.

## Description

The `torch::tensor_tanh` command computes the element-wise hyperbolic tangent (tanh) activation function of a tensor. The tanh function is defined as:

**tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))**

For each element `x` in the input tensor, the corresponding element in the output tensor will be the hyperbolic tangent of `x`.

**Mathematical Properties:**
- **Range**: Output is bounded between -1 and 1
- **Odd function**: tanh(-x) = -tanh(x)
- **Zero at origin**: tanh(0) = 0
- **Asymptotic behavior**: As x approaches ±∞, tanh(x) approaches ±1
- **Smooth and differentiable**: Unlike ReLU, tanh is smooth everywhere
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- Preserves tensor shape and data type

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with a single value
set t1 [torch::full {1} 1.0]

# Apply tanh activation
set result [torch::tensor_tanh $t1]

# Result: approximately 0.7616
```

### Named Parameter Syntax
```tcl
# Create a tensor with negative values
set input_tensor [torch::full {3} -0.5]

# Apply tanh using named parameters
set activated [torch::tensor_tanh -input $input_tensor]

# All elements will be approximately -0.4621
```

### CamelCase Syntax
```tcl
# Using camelCase alias with positional syntax
set t1 [torch::full {1} 2.0]
set result [torch::tensorTanh $t1]  # Result: approximately 0.9640

# Using camelCase alias with named parameters
set result2 [torch::tensorTanh -input $t1]  # Result: approximately 0.9640
```

### Neural Network Activation
```tcl
# Common usage in neural networks - activate hidden layer outputs
set linear_output [torch::linear $input $weights $bias]
set activated_hidden [torch::tensor_tanh $linear_output]

# Chain with other operations
set output_layer [torch::linear $activated_hidden $output_weights $output_bias]
```

### Mixed Positive and Negative Values
```tcl
# Create tensor with mixed values
set mixed [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0}]
set result [torch::tensor_tanh $mixed]

# Result will be approximately [-0.9640, -0.7616, 0.0, 0.7616, 0.9640]
```

### Neural Network Pattern
```tcl
# Linear layer followed by tanh activation
set input [torch::full {1} 1.0]
set weight [torch::full {1} 0.5]
set bias [torch::full {1} 0.0]

# Linear transformation: input * weight + bias = 1.0 * 0.5 + 0 = 0.5
set linear_output [torch::tensor_mul $input $weight]
set with_bias [torch::tensor_add $linear_output $bias]

# Apply tanh activation
set activated [torch::tensor_tanh $with_bias]
# Result: approximately 0.4621
```

## Error Handling

### Invalid Tensor Handle
```tcl
catch {torch::tensor_tanh invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Parameters
```tcl
catch {torch::tensor_tanh} error
# Error: "Usage: torch::tensor_tanh tensor" or "Required parameter missing: input"
```

### Unknown Parameters
```tcl
set t1 [torch::ones {1}]
catch {torch::tensor_tanh -invalid $t1} error
# Error: "Unknown parameter: -invalid"
```

### Missing Parameter Values
```tcl
catch {torch::tensor_tanh -input} error
# Error: "Missing value for parameter"
```

## Implementation Notes

### Backward Compatibility
The original positional syntax remains fully supported. Existing code using `torch::tensor_tanh tensor` will continue to work without modification.

### Performance
- All three syntax variants (positional, named, camelCase) produce identical results and have the same performance characteristics
- Uses PyTorch's optimized tanh implementation
- Supports GPU acceleration when input tensors are on CUDA devices

### Numerical Stability
- Numerically stable (no overflow/underflow issues)
- For very large positive or negative inputs, tanh saturates at exactly ±1.0 due to floating-point precision
- For very small inputs close to zero, tanh(x) ≈ x

## Use Cases

### Neural Network Activations
```tcl
# Hidden layer activation (common in RNNs, LSTMs, GRUs)
set hidden_layer [torch::linear $input $hidden_weights $hidden_bias]
set activated_hidden [torch::tensor_tanh $hidden_layer]
```

### Feature Normalization
```tcl
# Normalize features to [-1, 1] range
set normalized_features [torch::tensor_tanh $features]
```

### Signal Processing
```tcl
# Soft clipping of signals
set clipped_signal [torch::tensor_tanh $signal]
```

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old style
set result [torch::tensor_tanh $my_tensor]

# New style (equivalent)
set result [torch::tensor_tanh -input $my_tensor]
```

### Adopting CamelCase
```tcl
# Snake case
set result [torch::tensor_tanh -input $tensor]

# CamelCase (equivalent)  
set result [torch::tensorTanh -input $tensor]
```

## Mathematical Background

The hyperbolic tangent (tanh) is a widely used activation function in neural networks:

### Definition
**tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x)) = 2 / (1 + e^(-2x)) - 1**

### Properties
- **Bounded output**: Output range is (-1, 1)
- **Zero-centered**: Unlike sigmoid, tanh is centered at zero
- **Odd function**: tanh(-x) = -tanh(x)
- **Smooth and differentiable**: Continuous with continuous derivatives
- **Derivative**: tanh'(x) = 1 - tanh²(x)
- **Saturation**: For large |x|, tanh approaches ±1 and gradients become very small

### Advantages
- **Zero-centered output**: Mean activation is close to zero, which helps with optimization
- **Bounded gradients**: Helps prevent exploding gradient problems
- **Smooth non-linearity**: No "kinks" like in ReLU
- **Symmetric around origin**: Treats positive and negative inputs symmetrically

### Limitations
- **Vanishing gradient**: For large inputs, gradients become very small
- **Computationally more expensive** than ReLU
- **Saturation**: Neurons can saturate, leading to slow learning

### Common Applications
- **Recurrent Neural Networks (RNNs)**: Often used as activation in hidden layers
- **LSTM and GRU cells**: Used in gates and output activations
- **Feature normalization**: When features need to be bounded in [-1, 1]
- **Signal processing**: For soft clipping and normalization

## See Also

- [`torch::tensor_sigmoid`](tensor_sigmoid.md) - Sigmoid activation function (0 to 1 range)
- [`torch::tensor_relu`](tensor_relu.md) - Rectified Linear Unit activation
- [`torch::linear`](linear.md) - Linear layer (often used before tanh)
- [`torch::tensor_softmax`](tensor_softmax.md) - Softmax normalization function 