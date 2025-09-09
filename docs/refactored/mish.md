# torch::mish

## Overview
The Mish activation function is a smooth, non-monotonic activation function that has been shown to improve performance in deep neural networks. It provides a smooth alternative to ReLU and Swish/SiLU activations.

## Mathematical Definition
```
mish(x) = x * tanh(softplus(x))
mish(x) = x * tanh(ln(1 + exp(x)))
```

Where:
- `x` is the input value
- `tanh` is the hyperbolic tangent function
- `softplus(x) = ln(1 + exp(x))`

## Key Properties
- **Smooth**: Infinitely differentiable everywhere
- **Self-gated**: Uses its own values to gate the output
- **Unbounded above**: Can output values greater than the input
- **Bounded below**: Output approaches 0 as input approaches negative infinity
- **Non-monotonic**: Has a small negative region for negative inputs

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::mish -input tensor_name
```

### Positional Syntax (Legacy)
```tcl
torch::mish tensor_name
```

## Parameters

### Named Parameters
- **`-input`** (required): Input tensor name

### Positional Parameters
1. **tensor_name** (required): Input tensor name

## Return Value
Returns a new tensor handle containing the Mish activation applied element-wise to the input tensor.

## Examples

### Basic Usage with Named Parameters
```tcl
# Create a tensor with various values
set input [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

# Apply Mish activation
set output [torch::mish -input $input]

# Print the result
torch::tensor_print $output
# Output shows smooth activation values
```

### Basic Usage with Positional Parameters
```tcl
# Create a tensor
set input [torch::tensor_create {-1.0 0.0 1.0} float32 cpu false]

# Apply Mish activation (legacy syntax)
set output [torch::mish $input]

torch::tensor_print $output
```

### Neural Network Integration
```tcl
# In a neural network layer
set hidden [torch::linear -input $x -weight $w1 -bias $b1]
set activated [torch::mish -input $hidden]
set output [torch::linear -input $activated -weight $w2 -bias $b2]
```

### Comparison with Other Activations
```tcl
set x [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

# Compare different activations
set relu_out [torch::relu -input $x]
set silu_out [torch::silu -input $x]
set mish_out [torch::mish -input $x]

puts "ReLU output:"
torch::tensor_print $relu_out

puts "SiLU output:"
torch::tensor_print $silu_out

puts "Mish output:"
torch::tensor_print $mish_out
```

### Gradient Flow Analysis
```tcl
# Create input tensor with gradient tracking
set x [torch::tensor_create {-1.0 0.0 1.0} float32 cpu true]

# Apply Mish
set y [torch::mish -input $x]

# Compute some loss
set loss [torch::tensor_mean $y]

# Backpropagate to see gradients
torch::tensor_backward $loss

# Check gradients
set grad [torch::tensor_grad $x]
torch::tensor_print $grad
```

## Mathematical Behavior

### Range Analysis
- **Input range**: All real numbers (-∞, +∞)
- **Output range**: Approximately (-0.31, +∞)
- **Zero point**: mish(0) = 0
- **Positive values**: mish(x) ≈ x for large positive x
- **Negative values**: mish(x) approaches 0 as x → -∞

### Derivative Properties
```tcl
# The derivative of Mish is smooth everywhere
# d/dx mish(x) = sech²(softplus(x)) * x * sigmoid(x) + mish(x)/x
```

### Key Values
```tcl
# mish(0) = 0
# mish(1) ≈ 0.865
# mish(-1) ≈ -0.303
# mish(large positive) ≈ input value
# mish(large negative) ≈ 0
```

## Performance Characteristics

### Computational Cost
- **Forward pass**: Moderate (requires softplus and tanh computation)
- **Backward pass**: Smooth gradient computation
- **Memory**: Standard activation function memory usage

### Numerical Stability
```tcl
# Mish is numerically stable across typical input ranges
set extreme_values [torch::tensor_create {-100.0 -10.0 0.0 10.0 100.0} float32 cpu false]
set stable_output [torch::mish -input $extreme_values]
torch::tensor_print $stable_output
```

## Use Cases

### When to Use Mish
1. **Deep neural networks**: Particularly effective in deep architectures
2. **Computer vision**: Good performance in CNN architectures
3. **Smooth activation needed**: When you need differentiable activation
4. **Alternative to ReLU**: When ReLU causes dying neuron problems

### Performance Benefits
- Often outperforms ReLU in deep networks
- Better gradient flow than ReLU
- Self-regularizing properties
- Good empirical results in many domains

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old positional syntax
set result [torch::mish $input_tensor]

# New named parameter syntax
set result [torch::mish -input $input_tensor]
```

### From Other Activations
```tcl
# Replacing ReLU
# OLD: set activated [torch::relu $hidden]
set activated [torch::mish -input $hidden]

# Replacing Swish/SiLU
# OLD: set activated [torch::silu $hidden]
set activated [torch::mish -input $hidden]
```

## Error Handling

### Common Errors
```tcl
# Invalid tensor name
catch {torch::mish -input invalid_tensor} error
puts $error  # "Invalid tensor name"

# Missing required parameter
catch {torch::mish} error
puts $error  # "wrong # args: should be..."

# Unknown parameter
catch {torch::mish -unknown_param value} error
puts $error  # "unknown option -unknown_param"
```

### Input Validation
- Input must be a valid tensor handle
- Supports all floating-point data types
- Works with any tensor shape

## Implementation Notes

### Tensor Data Types
- **float32**: Standard precision (recommended)
- **float64**: Double precision for high accuracy
- **Other types**: Automatically promoted to appropriate floating-point type

### Memory Management
- Creates new tensor for output
- Input tensor remains unchanged
- Automatic memory cleanup

### Broadcasting
```tcl
# Mish works element-wise, so standard broadcasting rules apply
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
set reshaped [torch::tensor_reshape $input {2 3}]
set output [torch::mish -input $reshaped]
# Output maintains the 2x3 shape
```

## Related Functions
- `torch::relu` - ReLU activation
- `torch::silu` - SiLU/Swish activation  
- `torch::gelu` - GELU activation
- `torch::tanh` - Hyperbolic tangent
- `torch::sigmoid` - Sigmoid activation

## References
- [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
- Original paper by Diganta Misra

## Version History
- **v1.0**: Initial implementation with positional syntax
- **v2.0**: Added named parameter support and comprehensive error handling 