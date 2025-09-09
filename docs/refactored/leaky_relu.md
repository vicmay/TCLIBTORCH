# torch::leaky_relu / torch::leakyRelu

Applies the Leaky ReLU (Rectified Linear Unit) function element-wise to the input tensor with configurable negative slope.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::leaky_relu tensor ?negative_slope?
```

### Named Parameters (New Syntax)
```tcl
torch::leaky_relu -input tensor ?-negativeSlope slope?
torch::leaky_relu -input tensor ?-negative_slope slope?
torch::leaky_relu -input tensor ?-slope slope?
torch::leakyRelu -input tensor ?-negativeSlope slope?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |
| `negative_slope` / `-negativeSlope` / `-negative_slope` / `-slope` | double | Slope for negative values | 0.01 |

## Returns

Returns a string handle to the tensor containing the result of applying the Leaky ReLU function: `leaky_relu(x) = max(x, negative_slope * x)`.

## Mathematical Description

The Leaky ReLU function is defined as:
```
leaky_relu(x) = max(x, negative_slope * x)
```

This can be written piecewise as:
- For x ≥ 0: leaky_relu(x) = x (identity)
- For x < 0: leaky_relu(x) = negative_slope * x

The default negative slope is 0.01, which allows a small, non-zero gradient for negative inputs, addressing the "dying ReLU" problem where neurons can become permanently inactive.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with various values
set input [torch::tensor_create -data {-3.0 -1.0 0.0 1.0 3.0} -dtype float32 -device cpu]

# Apply Leaky ReLU with default negative slope (0.01)
set result [torch::leaky_relu $input]
# Result will be {-0.03, -0.01, 0.0, 1.0, 3.0}

# Apply Leaky ReLU with custom negative slope
set result [torch::leaky_relu $input 0.1]
# Result will be {-0.3, -0.1, 0.0, 1.0, 3.0}
```

### Named Parameters (New Syntax)
```tcl
# Create input tensor
set input [torch::full {3 3} -2.0]

# Apply Leaky ReLU with named parameters (default slope)
set result [torch::leaky_relu -input $input]

# Apply Leaky ReLU with custom slope using named parameters
set result [torch::leaky_relu -input $input -negativeSlope 0.2]

# Alternative parameter names
set result [torch::leaky_relu -input $input -negative_slope 0.1]
set result [torch::leaky_relu -input $input -slope 0.05]

# Check the result
puts "Shape: [torch::tensor_shape $result]"
```

### CamelCase Alias
```tcl
# Using camelCase syntax
set input [torch::full {2 2} -1.5]
set result [torch::leakyRelu $input]

# With named parameters and custom slope
set result [torch::leakyRelu -input $input -negativeSlope 0.1]

# With positional parameters
set result [torch::leakyRelu $input 0.05]
```

### Mathematical Properties
```tcl
# Leaky ReLU preserves positive values
set positive_input [torch::full {1} 2.0]
set result [torch::leaky_relu $positive_input]
set value [torch::tensor_item $result]
puts "leaky_relu(2.0) = $value"  # Should be 2.0

# Leaky ReLU scales negative values by the slope
set negative_input [torch::full {1} -2.0]
set result [torch::leaky_relu $negative_input 0.1]
set value [torch::tensor_item $result]
puts "leaky_relu(-2.0, slope=0.1) = $value"  # Should be -0.2

# Zero input remains zero
set zero_input [torch::full {1} 0.0]
set result [torch::leaky_relu $zero_input]
set value [torch::tensor_item $result]
puts "leaky_relu(0.0) = $value"  # Should be 0.0
```

### Neural Network Usage
```tcl
# Leaky ReLU is commonly used as an activation function
set hidden_output [torch::randn {32 64}]
set activated [torch::leaky_relu $hidden_output 0.01]

# For networks requiring stronger gradients for negative values
set layer_output [torch::randn {128 256}]
set activated [torch::leaky_relu $layer_output 0.1]

# Comparison with different slopes
set features [torch::randn {10 20}]
set conservative [torch::leaky_relu $features 0.01]   # Small slope
set aggressive [torch::leaky_relu $features 0.2]     # Larger slope
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::leaky_relu invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Arguments
```tcl
catch {torch::leaky_relu} error
puts $error  # Usage information
```

### Invalid Negative Slope
```tcl
set input [torch::ones {2}]
catch {torch::leaky_relu -input $input -negativeSlope -0.1} error
puts $error  # "negative_slope must be >= 0"

catch {torch::leaky_relu $input invalid_slope} error
puts $error  # "Invalid negative_slope"
```

### Unknown Parameters
```tcl
set input [torch::ones {2}]
catch {torch::leaky_relu -invalid_param $input} error
puts $error  # "Unknown parameter: -invalid_param"
```

## Data Type Support

Leaky ReLU preserves the input tensor's data type:

```tcl
# Float32 input
set input_f32 [torch::full {2} -1.0]
set result_f32 [torch::leaky_relu $input_f32]
puts [torch::tensor_dtype $result_f32]  # Float

# Float64 input
set input_f64 [torch::full {2} -1.0 float64]
set result_f64 [torch::leaky_relu $input_f64]
puts [torch::tensor_dtype $result_f64]  # Double
```

## Performance Characteristics

- **Computational Efficiency**: Fast element-wise max operation
- **Memory Usage**: Same as input tensor
- **Gradient Computation**: Provides gradients of 1 for positive values, negative_slope for negative values

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::leaky_relu $input_tensor]
set result [torch::leaky_relu $input_tensor 0.1]
```

**After (Named Parameters):**
```tcl
set result [torch::leaky_relu -input $input_tensor]
set result [torch::leaky_relu -input $input_tensor -negativeSlope 0.1]
```

### Using CamelCase
```tcl
# Snake_case (original)
set result [torch::leaky_relu $input_tensor 0.1]

# CamelCase (modern)
set result [torch::leakyRelu $input_tensor 0.1]

# Both syntaxes work identically
```

### Parameter Name Variations
```tcl
# All of these are equivalent:
set result [torch::leaky_relu -input $tensor -negativeSlope 0.1]
set result [torch::leaky_relu -input $tensor -negative_slope 0.1]
set result [torch::leaky_relu -input $tensor -slope 0.1]

# Parameter order doesn't matter in named syntax:
set result [torch::leaky_relu -negativeSlope 0.1 -input $tensor]
```

## Comparison with Related Functions

| Function | Positive Values | Negative Values | Use Case |
|----------|----------------|-----------------|----------|
| `leaky_relu` | x | negative_slope * x | Addressing dying ReLU problem |
| `relu` | x | 0 | Standard activation, risk of dying neurons |
| `elu` | x | α * (exp(x) - 1) | Smooth negative part, zero-centered |
| `selu` | λ * x | λ * α * (exp(x) - 1) | Self-normalizing networks |

## Common Use Cases

1. **Addressing Dying ReLU**: Prevents neurons from becoming permanently inactive
2. **Improved Gradient Flow**: Maintains gradients for negative values during backpropagation
3. **Better Learning Dynamics**: Can lead to faster convergence in some cases
4. **Robust Training**: Reduces the risk of "dead" neurons in deep networks
5. **Alternative to ReLU**: When standard ReLU causes training issues

## Advanced Examples

### Adaptive Slope Based on Training Phase
```tcl
# Start with larger slope, reduce over time
proc adaptive_leaky_relu {tensor epoch} {
    set slope [expr {0.1 / (1.0 + $epoch * 0.01)}]
    return [torch::leaky_relu $tensor $slope]
}

set activations [torch::randn {32 128}]
set result [adaptive_leaky_relu $activations 10]
```

### Comparing Different Slopes
```tcl
# Test different slopes to find optimal value
set input [torch::randn {64 128}]
set slopes [list 0.01 0.05 0.1 0.2]

foreach slope $slopes {
    set result [torch::leaky_relu $input $slope]
    puts "Slope $slope: shape = [torch::tensor_shape $result]"
}
```

### Gradient Analysis
```tcl
# Demonstrate gradient preservation for negative values
set negative_tensor [torch::full {1} -1.0]

# Standard ReLU would give 0 gradient
# Leaky ReLU maintains gradient
set small_slope [torch::leaky_relu $negative_tensor 0.01]   # Small gradient
set large_slope [torch::leaky_relu $negative_tensor 0.1]    # Larger gradient

puts "Small slope result: [torch::tensor_item $small_slope]"
puts "Large slope result: [torch::tensor_item $large_slope]"
```

## Implementation Notes

- Leaky ReLU was introduced to address the "dying ReLU" problem where neurons output zero for all inputs
- The negative slope parameter should be small (typically 0.01) to maintain the benefits of sparsity
- Unlike standard ReLU, Leaky ReLU is not zero-centered, but it allows gradient flow for negative inputs
- The function is differentiable everywhere, with derivatives of 1 for positive inputs and negative_slope for negative inputs
- Parameter validation ensures negative_slope ≥ 0 to maintain mathematical consistency

## Choosing the Negative Slope

- **0.01** (default): Conservative approach, minimal gradient for negative values
- **0.1**: More aggressive, stronger gradients for negative values
- **0.2**: Maximum commonly used value, approaches linear behavior
- **0.0**: Equivalent to standard ReLU
- **1.0**: Linear function (not recommended for activation)

## See Also

- [`torch::relu`](relu.md) - Standard Rectified Linear Unit
- [`torch::elu`](elu.md) - Exponential Linear Unit
- [`torch::selu`](selu.md) - Scaled Exponential Linear Unit
- [`torch::prelu`](prelu.md) - Parametric ReLU (learnable slope)
- [`torch::relu6`](relu6.md) - ReLU clamped to [0, 6] range 