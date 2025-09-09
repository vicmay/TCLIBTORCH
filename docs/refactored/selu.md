# torch::selu

## Overview
The Scaled Exponential Linear Unit (SELU) is a self-normalizing activation function that enables deep neural networks to converge towards a normalized state. It has been proven to induce strong regularization effects and maintain stable activations even in very deep networks.

## Mathematical Definition
```
selu(x) = scale * (max(0, x) + min(0, α * (exp(x) - 1)))
```

Where:
- `α ≈ 1.6732632423543772` (derived theoretically)
- `scale ≈ 1.0507009873554805` (derived theoretically)
- `x` is the input value

## Key Properties
- **Self-normalizing**: Automatically normalizes activations in deep networks
- **Strong regularization**: Built-in regularization effects
- **Theoretical foundation**: Parameters derived from mathematical principles
- **Continuous**: Smooth function everywhere
- **Zero-preserving**: selu(0) = 0

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::selu -input tensor_name
```

### Positional Syntax (Legacy)
```tcl
torch::selu tensor_name
```

## Parameters

### Named Parameters
- **`-input`** (required): Input tensor name

### Positional Parameters
1. **tensor_name** (required): Input tensor name

## Return Value
Returns a new tensor handle containing the SELU activation applied element-wise to the input tensor.

## Examples

### Basic Usage with Named Parameters
```tcl
# Create a tensor with various values
set input [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

# Apply SELU activation
set output [torch::selu -input $input]

# Print the result
torch::tensor_print $output
# Output shows self-normalizing activation values
```

### Basic Usage with Positional Parameters
```tcl
# Create a tensor
set input [torch::tensor_create {-1.0 0.0 1.0} float32 cpu false]

# Apply SELU activation (legacy syntax)
set output [torch::selu $input]

torch::tensor_print $output
```

### Deep Neural Network Example
```tcl
# SELU is particularly effective in deep networks
set layer1 [torch::linear -input $x -weight $w1 -bias $b1]
set selu1 [torch::selu -input $layer1]

set layer2 [torch::linear -input $selu1 -weight $w2 -bias $b2]
set selu2 [torch::selu -input $layer2]

set layer3 [torch::linear -input $selu2 -weight $w3 -bias $b3]
set selu3 [torch::selu -input $layer3]

# Can continue for many layers without normalization
```

### Self-Normalizing Properties Demo
```tcl
# Test self-normalizing behavior
set x [torch::tensor_create {-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0} float32 cpu false]

# Apply multiple SELU layers
set layer1 [torch::selu -input $x]
set layer2 [torch::selu -input $layer1]
set layer3 [torch::selu -input $layer2]

puts "Layer 1 mean:"
set mean1 [torch::tensor_mean $layer1]
torch::tensor_print $mean1

puts "Layer 3 mean:"
set mean3 [torch::tensor_mean $layer3]
torch::tensor_print $mean3
# Means should remain close to 0
```

### Comparison with Other Activations
```tcl
set x [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

# Compare different activations
set relu_out [torch::relu -input $x]
set elu_out [torch::elu -input $x]
set selu_out [torch::selu -input $x]

puts "ReLU output:"
torch::tensor_print $relu_out

puts "ELU output:"
torch::tensor_print $elu_out

puts "SELU output:"
torch::tensor_print $selu_out
```

### Gradient Analysis
```tcl
# Create input tensor with gradient tracking
set x [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu true]

# Apply SELU
set y [torch::selu -input $x]

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
- **Output range**: (-α × scale, +∞) ≈ (-1.758, +∞)
- **Zero point**: selu(0) = 0
- **Positive values**: selu(x) = scale × x ≈ 1.0507 × x
- **Negative values**: selu(x) = scale × α × (exp(x) - 1)

### Self-Normalizing Property
The SELU activation function has the unique property that it maintains mean activations close to 0 and variance close to 1 in deep networks when:
1. Weights are initialized with He initialization
2. Dropout is replaced with Alpha Dropout
3. Network architecture is feedforward

### Key Values
```tcl
# selu(0) = 0
# selu(1) ≈ 1.0507
# selu(-1) ≈ -1.1113
# For large positive x: selu(x) ≈ 1.0507 × x
# For large negative x: selu(x) → -1.758
```

## Performance Characteristics

### Computational Cost
- **Forward pass**: Moderate (requires exponential computation for negative values)
- **Backward pass**: Smooth gradient computation
- **Memory**: Standard activation function memory usage

### Numerical Stability
```tcl
# SELU is numerically stable across typical input ranges
set extreme_values [torch::tensor_create {-100.0 -10.0 0.0 10.0 100.0} float32 cpu false]
set stable_output [torch::selu -input $extreme_values]
torch::tensor_print $stable_output
```

## Use Cases

### When to Use SELU
1. **Deep feedforward networks**: Primary use case for self-normalization
2. **Networks without batch normalization**: SELU can replace normalization layers
3. **Regression tasks**: Often works well for continuous output prediction
4. **When avoiding batch dependencies**: SELU doesn't require batch statistics

### Advantages
- Theoretical guarantees for self-normalization
- Built-in regularization effects
- No need for batch normalization in feedforward networks
- Maintains stable gradients in deep networks

### Limitations
- Primarily effective in feedforward architectures
- Requires specific initialization and dropout strategies
- Less effective in recurrent or convolutional architectures
- Performance depends on proper hyperparameter choices

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old positional syntax
set result [torch::selu $input_tensor]

# New named parameter syntax
set result [torch::selu -input $input_tensor]
```

### From Other Activations
```tcl
# Replacing ReLU in deep networks
# OLD: set activated [torch::relu $hidden]
set activated [torch::selu -input $hidden]

# Replacing ELU
# OLD: set activated [torch::elu $hidden]
set activated [torch::selu -input $hidden]
```

### Converting to Self-Normalizing Networks
```tcl
# Traditional approach with batch normalization
# set hidden [torch::linear -input $x -weight $w -bias $b]
# set normalized [torch::batch_norm $hidden]
# set activated [torch::relu $normalized]

# Self-normalizing approach with SELU
set hidden [torch::linear -input $x -weight $w -bias $b]
set activated [torch::selu -input $hidden]  # No normalization needed
```

## Error Handling

### Common Errors
```tcl
# Invalid tensor name
catch {torch::selu -input invalid_tensor} error
puts $error  # "Invalid tensor name"

# Missing required parameter
catch {torch::selu} error
puts $error  # "wrong # args: should be..."

# Unknown parameter
catch {torch::selu -unknown_param value} error
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
# SELU works element-wise, so standard broadcasting rules apply
set input [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32 cpu false]
set reshaped [torch::tensor_reshape $input {2 3}]
set output [torch::selu -input $reshaped]
# Output maintains the 2x3 shape
```

## Integration with Self-Normalizing Networks

### Proper Initialization
```tcl
# Use He initialization for weights
# set weights [torch::tensor_randn {input_size output_size} float32 cpu false]
# Scale by sqrt(2/input_size) for proper SELU initialization
```

### Alpha Dropout
```tcl
# Replace standard dropout with alpha dropout for SELU networks
# This maintains the self-normalizing property during training
```

### Network Architecture
```tcl
# Example of a proper self-normalizing network
proc create_snn_layer {input weight bias} {
    set linear [torch::linear -input $input -weight $weight -bias $bias]
    set activated [torch::selu -input $linear]
    return $activated
}
```

## Related Functions
- `torch::elu` - Exponential Linear Unit (ELU)
- `torch::relu` - Rectified Linear Unit
- `torch::alpha_dropout` - Alpha Dropout (for SELU networks)
- `torch::batch_norm` - Batch Normalization (alternative approach)

## References
- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
- Original paper by Günter Klambauer et al.
- Mathematical derivation of α and scale parameters

## Best Practices
1. **Use with feedforward architectures**: SELU works best in feedforward networks
2. **Proper initialization**: Use He initialization scaled appropriately
3. **Alpha dropout**: Use alpha dropout instead of standard dropout
4. **Avoid with batch norm**: Don't combine SELU with batch normalization
5. **Deep networks**: SELU shines in networks with many layers

## Version History
- **v1.0**: Initial implementation with positional syntax
- **v2.0**: Added named parameter support and comprehensive error handling 