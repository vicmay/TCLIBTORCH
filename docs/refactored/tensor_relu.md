# torch::tensor_relu

Element-wise Rectified Linear Unit (ReLU) activation function for tensors.

## Syntax

### Positional Syntax (Original)
```tcl
torch::tensor_relu tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_relu -input tensor
```

### CamelCase Alias
```tcl
torch::tensorRelu tensor
torch::tensorRelu -input tensor
```

## Parameters

### Required Parameters
- **input** (`string`): Handle to the input tensor

## Return Value

Returns a string handle to a new tensor containing the element-wise ReLU activation of the input tensor.

## Description

The `torch::tensor_relu` command computes the element-wise Rectified Linear Unit (ReLU) activation function of a tensor. The ReLU function is defined as:

**ReLU(x) = max(0, x)**

For each element `x` in the input tensor, the corresponding element in the output tensor will be `x` if `x > 0`, and `0` otherwise.

**Mathematical Properties:**
- **Non-negativity**: Output is always greater than or equal to 0
- **Identity for positive**: For x > 0, ReLU(x) = x
- **Zero for negative**: For x ≤ 0, ReLU(x) = 0
- **Non-differentiable at x = 0**: Has a "kink" at zero
- **Sparsity**: Creates sparse activations (many zeros for negative inputs)
- Preserves tensor shape and data type

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with mixed positive and negative values
set t1 [torch::full {3} -2.0]
set t2 [torch::full {3} 3.0]
set mixed [torch::tensor_add $t1 $t2]  # [-2+3, -2+3, -2+3] = [1, 1, 1]

# Apply ReLU activation
set result [torch::tensor_relu $mixed]

# Result: [1.0, 1.0, 1.0] (unchanged since all values are positive)
```

### Named Parameter Syntax
```tcl
# Create a tensor with negative values
set input_tensor [torch::full {3} -5.0]

# Apply ReLU using named parameters
set activated [torch::tensor_relu -input $input_tensor]

# All elements will be 0.0 (since all inputs are negative)
```

### CamelCase Syntax
```tcl
# Using camelCase alias with positional syntax
set t1 [torch::full {1} -2.0]
set result [torch::tensorRelu $t1]  # Result: 0.0

# Using camelCase alias with named parameters
set result2 [torch::tensorRelu -input $t1]  # Result: 0.0
```

### Neural Network Activation
```tcl
# Common usage in neural networks - activate hidden layer outputs
set linear_output [torch::linear $input $weights $bias]
set activated_hidden [torch::tensor_relu $linear_output]

# Chain with other operations
set output_layer [torch::linear $activated_hidden $output_weights $output_bias]
```

### Mixed Positive and Negative Values
```tcl
# Create tensor with mixed values
set mixed [torch::tensor_create {-5.0 -2.0 0.0 3.0 7.0}]
set result [torch::tensor_relu $mixed]

# Result will be [0.0, 0.0, 0.0, 3.0, 7.0]
```

### Neural Network Pattern
```tcl
# Linear layer followed by ReLU activation
set input [torch::full {1} 2.0]
set weight [torch::full {1} -1.5]  # Negative weight
set bias [torch::full {1} 1.0]

# Linear transformation: input * weight + bias = 2 * (-1.5) + 1 = -3 + 1 = -2
set linear_output [torch::tensor_mul $input $weight]
set with_bias [torch::tensor_add $linear_output $bias]

# Apply ReLU activation
set activated [torch::tensor_relu $with_bias]
# Result: 0.0 (since -2 < 0)
```

## Error Handling

### Invalid Tensor Handle
```tcl
catch {torch::tensor_relu invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Parameters
```tcl
catch {torch::tensor_relu} error
# Error: "Usage: torch::tensor_relu tensor" or "Required parameter missing: input"
```

### Unknown Parameters
```tcl
set t1 [torch::ones {1}]
catch {torch::tensor_relu -invalid $t1} error
# Error: "Unknown parameter: -invalid"
```

### Missing Parameter Values
```tcl
catch {torch::tensor_relu -input} error
# Error: "Missing value for parameter"
```

## Implementation Notes

### Backward Compatibility
The original positional syntax remains fully supported. Existing code using `torch::tensor_relu tensor` will continue to work without modification.

### Performance
- All three syntax variants (positional, named, camelCase) produce identical results and have the same performance characteristics
- Uses PyTorch's optimized ReLU implementation
- Very computationally efficient (simple max operation)
- Supports GPU acceleration when input tensors are on CUDA devices

### Numerical Stability
- Numerically stable (no overflow/underflow issues)
- No saturation for large positive values
- Creates exact zeros for negative inputs (not approximate)

## Use Cases

### Neural Network Activations
```tcl
# Hidden layer activation
set hidden_layer [torch::linear $input $hidden_weights $hidden_bias]
set activated_hidden [torch::tensor_relu $hidden_layer]
```

### Feature Extraction
```tcl
# Extract only positive components of a signal
set signal [torch::tensor_create $raw_signal_data]
set positive_components [torch::tensor_relu $signal]
```

### Sparse Representation
```tcl
# Create sparse representation by zeroing out negative values
set sparse_features [torch::tensor_relu $features]
```

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old style
set result [torch::tensor_relu $my_tensor]

# New style (equivalent)
set result [torch::tensor_relu -input $my_tensor]
```

### Adopting CamelCase
```tcl
# Snake case
set result [torch::tensor_relu -input $tensor]

# CamelCase (equivalent)  
set result [torch::tensorRelu -input $tensor]
```

## Mathematical Background

The ReLU function is one of the most widely used activation functions in deep learning:

### Definition
**ReLU(x) = max(0, x)**

### Properties
- **Piecewise linear**: Two linear segments joined at x=0
- **Non-saturating for positive values**: No upper bound for x > 0
- **Sparse activation**: Creates exact zeros for negative inputs
- **Efficient computation**: Simple max operation
- **Derivative**: 0 for x < 0, 1 for x > 0, undefined at x = 0 (but typically set to 0 or 1)

### Advantages
- **Mitigates vanishing gradient**: No saturation for positive values
- **Computational efficiency**: Simple max operation
- **Sparse activations**: Many neurons output exactly zero
- **Biological plausibility**: Similar to neuronal firing patterns

### Limitations
- **Dying ReLU problem**: Neurons can "die" if they only receive negative inputs
- **Mean shift**: Introduces positive bias in activations
- **Non-zero centered**: All outputs are ≥ 0
- **Non-differentiable at x = 0**: Requires special handling at this point

### Variants
- **Leaky ReLU**: f(x) = x if x > 0, αx otherwise (where α is a small constant)
- **Parametric ReLU**: Learnable α parameter
- **ELU**: Exponential Linear Unit (smooth approximation)
- **GELU**: Gaussian Error Linear Unit (smooth approximation)

## See Also

- [`torch::tensor_sigmoid`](tensor_sigmoid.md) - Sigmoid activation function
- [`torch::tensor_tanh`](tensor_tanh.md) - Hyperbolic tangent activation
- [`torch::linear`](linear.md) - Linear layer (often used before ReLU)
- [`torch::tensor_leaky_relu`](tensor_leaky_relu.md) - Leaky ReLU variant 