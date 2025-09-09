# torch::softplus

Applies the Softplus activation function to the input tensor. Softplus is a smooth approximation to the ReLU function.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::softplus -input tensor
torch::softplus -tensor tensor
torch::softPlus -input tensor
```

### Positional Parameters (Legacy)
```tcl
torch::softplus tensor
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input`/`tensor` | tensor | required | Input tensor |

## Description

The Softplus function is computed as:
```
softplus(x) = log(1 + exp(x))
```

Softplus is a smooth approximation to the ReLU activation function. Unlike ReLU, which has a sharp corner at zero, Softplus is differentiable everywhere and produces strictly positive outputs.

## Examples

### Basic Usage
```tcl
# Create input tensor
set input [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -shape {5} -dtype float32]

# Apply softplus activation
set result [torch::softplus -input $input]
puts "Softplus result: [torch::tensorToList $result]"
# Output: Softplus result: {0.127 0.313 0.693 1.313 2.127} (approximately)
```

### Multi-dimensional Tensor
```tcl
# Create 2D tensor
set input [torch::tensorCreate -data {-1.0 0.0 1.0 -2.0 2.0 3.0} -shape {2 3} -dtype float32]

# Apply softplus
set result [torch::softplus -input $input]
puts "Shape: [torch::tensorShape $result]"
# Output: Shape: 2 3
```

### Mathematical Properties
```tcl
# Demonstrate key mathematical properties

# 1. Always positive output
set input [torch::tensorCreate -data {-100.0 0.0 100.0} -shape {3} -dtype float32]
set result [torch::softplus -input $input]
set values [torch::tensorToList $result]
puts "All positive: $values"
# Output: All positive: {3.72e-44 0.693 100.0} (approximately)

# 2. softplus(0) = log(2) ≈ 0.693
set zero_input [torch::tensorCreate -data {0.0} -shape {1} -dtype float32]
set zero_result [torch::softplus -input $zero_input]
puts "softplus(0): [torch::tensorItem $zero_result]"
# Output: softplus(0): 0.6931471824646

# 3. For large positive x, softplus(x) ≈ x
set large_input [torch::tensorCreate -data {10.0} -shape {1} -dtype float32]
set large_result [torch::softplus -input $large_input]
puts "softplus(10): [torch::tensorItem $large_result]"
# Output: softplus(10): 10.000045
```

## Legacy Syntax

```tcl
# Positional syntax (backward compatible)
set input [torch::tensorCreate -data {-1.0 0.0 1.0} -shape {3} -dtype float32]
set result [torch::softplus $input]
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::softplus $tensor]
```

**New (Named):**
```tcl
set result [torch::softplus -input $tensor]
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Consistency**: Follows modern API patterns
- **Maintainability**: Code is self-documenting
- **Flexibility**: Compatible with future parameter additions

## Mathematical Background

### Softplus vs ReLU
- **ReLU**: `max(0, x)` - Not differentiable at x=0
- **Softplus**: `log(1 + exp(x))` - Smooth and differentiable everywhere

### Key Properties
1. **Always Positive**: `softplus(x) > 0` for all x
2. **Smooth**: Differentiable everywhere (unlike ReLU)
3. **Approximation**: For large positive x, `softplus(x) ≈ x`
4. **Bounded Below**: `softplus(x) ≥ 0`
5. **Monotonic**: Strictly increasing function

### Derivative
The derivative of Softplus is the sigmoid function:
```
d/dx softplus(x) = sigmoid(x) = 1 / (1 + exp(-x))
```

## Common Use Cases

1. **Neural Network Activations**: Smooth alternative to ReLU
2. **Variational Autoencoders**: Ensuring positive variance parameters
3. **Bayesian Neural Networks**: Positive scaling factors
4. **Optimization**: When gradients must be well-defined everywhere
5. **Probabilistic Models**: Ensuring positive parameters

## Performance Considerations

- **Computational Cost**: More expensive than ReLU due to exponential computation
- **Numerical Stability**: Well-behaved for reasonable input ranges
- **Gradient Flow**: Better gradient flow than ReLU for negative inputs

## Comparison with Other Activations

| Function | Formula | Differentiable | Always Positive | Computational Cost |
|----------|---------|----------------|-----------------|-------------------|
| ReLU | max(0, x) | No (at x=0) | No | Low |
| Softplus | log(1 + exp(x)) | Yes | Yes | Medium |
| ELU | x if x>0, α(exp(x)-1) | Yes | No | Medium |
| Swish/SiLU | x * sigmoid(x) | Yes | No | Medium |

## Return Value

Returns a new tensor with the same shape as the input, containing the Softplus-activated values. All output values are strictly positive.

## See Also

- `torch::relu` - ReLU activation function
- `torch::elu` - Exponential Linear Unit
- `torch::silu` - SiLU/Swish activation function
- `torch::sigmoid` - Sigmoid activation function 