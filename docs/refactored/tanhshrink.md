# torch::tanhshrink

Applies the Tanh Shrink activation function element-wise to a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tanhshrink tensor
```

### Named Parameter Syntax (Recommended)
```tcl
torch::tanhshrink -input tensor
```

### camelCase Alias
```tcl
torch::tanhShrink tensor
torch::tanhShrink -input tensor
```

## Parameters

### Required Parameters
- **input** (tensor): Input tensor

## Return Value

Returns a tensor handle containing the result of applying the Tanh Shrink function to the input tensor.

## Mathematical Definition

The Tanh Shrink (tanhshrink) function is defined as:
```
tanhshrink(x) = x - tanh(x)
```

### Properties
- **Continuous and differentiable** everywhere
- **Range**: (-∞, +∞)
- **Non-monotonic**: Has both increasing and decreasing regions
- **Zero point**: tanhshrink(0) = 0
- **Asymptotic behavior**: 
  - For large positive x: tanhshrink(x) ≈ x - 1
  - For large negative x: tanhshrink(x) ≈ x + 1

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

# Positional syntax
set result1 [torch::tanhshrink $input]

# Named parameter syntax  
set result2 [torch::tanhshrink -input $input]

# camelCase alias
set result3 [torch::tanhShrink -input $input]
```

### Neural Network Integration

```tcl
# Custom activation in a neural network layer
proc create_tanhshrink_layer {input_tensor} {
    # Linear transformation
    set linear_out [torch::linear $input_tensor $weights $bias]
    
    # Apply tanhshrink activation
    set activated [torch::tanhshrink -input $linear_out]
    
    return $activated
}

# Usage in training loop
set features [torch::tensor_create {/* feature data */} float32 cpu false]
set layer_output [create_tanhshrink_layer $features]
```

### Activation Function Comparison

```tcl
# Compare different activation functions
set x [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0} float32 cpu false]

set tanh_result [torch::tanh $x]
set tanhshrink_result [torch::tanhshrink -input $x]

# Verify mathematical relationship: tanhshrink(x) = x - tanh(x)
set manual_calc [torch::sub $x $tanh_result]
```

### Gradient Flow Analysis

```tcl
# Enable gradient computation
set x [torch::tensor_create {-1.0 0.0 1.0} float32 cpu true]
torch::requires_grad_ -tensor $x -grad true

# Apply tanhshrink
set y [torch::tanhshrink -input $x]

# Compute gradients (if loss function is applied)
# The derivative is: d/dx tanhshrink(x) = 1 - sech²(x) = tanh²(x)
```

## Applications

### 1. **Custom Activation Functions**
```tcl
# Alternative to ReLU in hidden layers
set hidden1 [torch::linear $input $weights1 $bias1]
set activated1 [torch::tanhshrink -input $hidden1]
```

### 2. **Residual Connections**
```tcl
# Modified residual block with tanhshrink
set residual [torch::tanhshrink -input $conv_output]
set output [torch::add $input $residual]
```

### 3. **Signal Processing**
```tcl
# Non-linear signal transformation
set signal [torch::tensor_create {/* audio samples */} float32 cpu false]
set processed [torch::tanhshrink -input $signal]
```

### 4. **Feature Engineering**
```tcl
# Non-linear feature transformation
set features [torch::tensor_create {/* raw features */} float32 cpu false]
set transformed [torch::tanhshrink -input $features]
```

## Performance Characteristics

- **Computational complexity**: O(n) where n is the number of elements
- **Memory usage**: Creates new tensor, original tensor unchanged
- **Gradient computation**: Supports automatic differentiation
- **Numerical stability**: Stable for all input ranges

## Data Type Support

Supports all floating-point tensor types:
- `float32` (single precision)
- `float64` (double precision)

## Device Support

Works on both CPU and GPU tensors (if CUDA is available).

## Error Handling

### Common Errors

```tcl
# Error: Missing required parameter
catch {torch::tanhshrink} msg
# Returns: "wrong # args" error

# Error: Invalid tensor name
catch {torch::tanhshrink invalid_tensor} msg  
# Returns: "Invalid tensor name" error

# Error: Unknown parameter
catch {torch::tanhshrink -input $tensor -unknown value} msg
# Returns: "unknown option" error
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::tanhshrink $input]

# New named parameter syntax
set result [torch::tanhshrink -input $input]

# Both produce identical results
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Consistency**: Matches modern API design patterns
- **Extensibility**: Easier to add optional parameters in future

## Mathematical Background

The tanhshrink function combines linear and hyperbolic tangent components:

1. **Linear component**: x (identity function)
2. **Hyperbolic component**: -tanh(x) (bounded between -1 and 1)
3. **Result**: Unbounded function with non-linear characteristics

### Derivative
```
d/dx tanhshrink(x) = d/dx (x - tanh(x)) = 1 - sech²(x) = tanh²(x)
```

### Use Cases
- **Alternative activation**: When you need unbounded but non-linear activation
- **Residual learning**: Natural for skip connections
- **Signal processing**: Non-linear transformations with identity component

## See Also

- [`torch::tanh`](tanh.md) - Hyperbolic tangent activation
- [`torch::shrink`](shrink.md) - Soft shrinkage function  
- [`torch::relu`](relu.md) - Rectified Linear Unit
- [`torch::selu`](selu.md) - Scaled Exponential Linear Unit
- [`torch::softshrink`](softshrink.md) - Soft shrinkage activation

## Version History

- **v1.0**: Initial implementation with positional syntax
- **v2.0**: Added dual syntax support and camelCase alias 