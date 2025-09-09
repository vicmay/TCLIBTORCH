# torch::hardtanh / torch::hardTanh

Applies the hard tanh function element-wise to the input tensor with configurable clipping range.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::hardtanh tensor ?min_val? ?max_val?
```

### Named Parameters (New Syntax)
```tcl
torch::hardtanh -input tensor ?-min min_val? ?-max max_val?
torch::hardtanh -input tensor ?-minVal min_val? ?-maxVal max_val?
torch::hardTanh -input tensor ?-min min_val? ?-max max_val?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |
| `min_val` / `-min` / `-minVal` | double | Minimum value for clipping | -1.0 |
| `max_val` / `-max` / `-maxVal` | double | Maximum value for clipping | 1.0 |

## Returns

Returns a string handle to the tensor containing the result of applying the hard tanh function: `hardtanh(x) = max(min_val, min(max_val, x))`.

## Mathematical Description

The hard tanh function is defined as:
```
hardtanh(x) = max(min_val, min(max_val, x))
```

This is a piecewise linear function that clips values to a specified range:
- For x ≤ min_val: hardtanh(x) = min_val
- For min_val < x < max_val: hardtanh(x) = x (identity)
- For x ≥ max_val: hardtanh(x) = max_val

By default, min_val = -1.0 and max_val = 1.0, making it equivalent to `max(-1, min(1, x))`.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with various values
set input [torch::tensor_create -data {-3.0 -1.0 0.0 1.0 3.0} -dtype float32 -device cpu]

# Apply hard tanh with default range [-1, 1]
set result [torch::hardtanh $input]
# Result will be {-1.0, -1.0, 0.0, 1.0, 1.0}

# Apply hard tanh with custom range
set result [torch::hardtanh $input -0.5 0.5]
# Result will be {-0.5, -0.5, 0.0, 0.5, 0.5}
```

### Named Parameters (New Syntax)
```tcl
# Create input tensor
set input [torch::full {3 3} 2.0]

# Apply hard tanh with named parameters (default range)
set result [torch::hardtanh -input $input]

# Apply hard tanh with custom range using named parameters
set result [torch::hardtanh -input $input -min -2.0 -max 2.0]

# Alternative parameter names
set result [torch::hardtanh -input $input -minVal -0.5 -maxVal 0.5]

# Check the result
puts "Shape: [torch::tensor_shape $result]"
```

### CamelCase Alias
```tcl
# Using camelCase syntax
set input [torch::full {2 2} 1.5]
set result [torch::hardTanh $input]

# With named parameters and custom range
set result [torch::hardTanh -input $input -min -0.8 -max 0.8]

# With positional parameters
set result [torch::hardTanh $input -2.0 2.0]
```

### Mathematical Properties
```tcl
# Hard tanh preserves values within the range
set in_range [torch::full {1} 0.5]
set result [torch::hardtanh $in_range]
set value [torch::tensor_item $result]
puts "hardtanh(0.5) = $value"  # Should be 0.5

# Hard tanh clips values outside the range
set above_range [torch::full {1} 2.0]
set below_range [torch::full {1} -2.0]
set result_above [torch::hardtanh $above_range]
set result_below [torch::hardtanh $below_range]
puts "hardtanh(2.0) = [torch::tensor_item $result_above]"   # Should be 1.0
puts "hardtanh(-2.0) = [torch::tensor_item $result_below]"  # Should be -1.0

# Custom clipping range
set tensor [torch::full {1} 3.0]
set result [torch::hardtanh $tensor -0.5 0.5]
puts "hardtanh(3.0, min=-0.5, max=0.5) = [torch::tensor_item $result]"  # Should be 0.5
```

### Neural Network Usage
```tcl
# Hard tanh is often used as an activation function with custom ranges
set hidden_output [torch::randn {32 64}]
set activated [torch::hardtanh $hidden_output -0.1 0.1]

# For specific applications requiring bounded outputs
set logits [torch::randn {10 5}]
set bounded_logits [torch::hardtanh $logits -5.0 5.0]
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::hardtanh invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Arguments
```tcl
catch {torch::hardtanh} error
puts $error  # Usage information
```

### Invalid Parameter Order
```tcl
set input [torch::ones {2}]
catch {torch::hardtanh -input $input -min 1.0 -max -1.0} error
puts $error  # "min_val must be <= max_val"
```

### Invalid Parameter Values
```tcl
set input [torch::ones {2}]
catch {torch::hardtanh $input invalid_min} error
puts $error  # "Invalid min_val"

catch {torch::hardtanh $input -1.0 invalid_max} error
puts $error  # "Invalid max_val"
```

### Unknown Parameters
```tcl
set input [torch::ones {2}]
catch {torch::hardtanh -invalid_param $input} error
puts $error  # "Unknown parameter: -invalid_param"
```

## Data Type Support

Hard tanh preserves the input tensor's data type:

```tcl
# Float32 input
set input_f32 [torch::full {2} 2.0]
set result_f32 [torch::hardtanh $input_f32]
puts [torch::tensor_dtype $result_f32]  # Float

# Float64 input
set input_f64 [torch::full {2} 2.0 float64]
set result_f64 [torch::hardtanh $input_f64]
puts [torch::tensor_dtype $result_f64]  # Double
```

## Performance Characteristics

- **Computational Efficiency**: Very fast - simple min/max operations
- **Memory Usage**: Same as input tensor
- **Gradient Computation**: Provides gradients of 1 within the range, 0 outside the range

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::hardtanh $input_tensor]
set result [torch::hardtanh $input_tensor -0.5 0.5]
```

**After (Named Parameters):**
```tcl
set result [torch::hardtanh -input $input_tensor]
set result [torch::hardtanh -input $input_tensor -min -0.5 -max 0.5]
```

### Using CamelCase
```tcl
# Snake_case (original)
set result [torch::hardtanh $input_tensor -0.5 0.5]

# CamelCase (modern)
set result [torch::hardTanh $input_tensor -0.5 0.5]

# Both syntaxes work identically
```

### Parameter Name Variations
```tcl
# All of these are equivalent:
set result [torch::hardtanh -input $tensor -min -0.5 -max 0.5]
set result [torch::hardtanh -input $tensor -minVal -0.5 -maxVal 0.5]

# Parameter order doesn't matter in named syntax:
set result [torch::hardtanh -min -0.5 -input $tensor -max 0.5]
```

## Comparison with Related Functions

| Function | Range | Gradient | Use Case |
|----------|-------|----------|----------|
| `hardtanh` | [min_val, max_val] | 1 inside, 0 outside | Bounded activation, gradient clipping |
| `tanh` | (-1, 1) | Smooth, decreasing | Smooth bounded activation |
| `sigmoid` | (0, 1) | Smooth, bell-shaped | Probability-like outputs |
| `relu` | [0, ∞) | 1 for x>0, 0 for x≤0 | Standard ReLU activation |

## Common Use Cases

1. **Bounded Activations**: When you need outputs constrained to a specific range
2. **Gradient Clipping**: Preventing gradient explosion by clipping intermediate values
3. **Custom Output Ranges**: When domain knowledge suggests specific value bounds
4. **Robust Training**: Preventing extreme activations that could destabilize training
5. **Mobile/Edge Deployment**: Simple operations for efficient inference

## Advanced Examples

### Dynamic Range Adjustment
```tcl
# Adjust clipping range based on training phase
proc adaptive_hardtanh {tensor epoch} {
    set range [expr {1.0 + 0.1 * $epoch}]
    return [torch::hardtanh $tensor -$range $range]
}

set activations [torch::randn {32 128}]
set clipped [adaptive_hardtanh $activations 5]
```

### Symmetric Clipping
```tcl
# Ensure symmetric clipping around zero
proc symmetric_clip {tensor magnitude} {
    return [torch::hardtanh $tensor -$magnitude $magnitude]
}

set outputs [torch::randn {64}]
set symmetric_outputs [symmetric_clip $outputs 0.5]
```

### Batch Processing with Different Ranges
```tcl
# Process multiple tensors with different clipping ranges
set tensors [list $tensor1 $tensor2 $tensor3]
set ranges [list {-1.0 1.0} {-0.5 0.5} {-2.0 2.0}]

foreach tensor $tensors range $ranges {
    lassign $range min_val max_val
    set result [torch::hardtanh $tensor $min_val $max_val]
    # Process result...
}
```

## Implementation Notes

- The hard tanh function is a piecewise linear approximation of the tanh function
- It provides computational benefits while maintaining bounded output ranges
- The function is not differentiable at the clipping boundaries (min_val and max_val)
- Gradients are exactly 1 within the range and 0 outside, making it suitable for gradient flow control
- Parameter validation ensures min_val ≤ max_val to prevent invalid ranges

## See Also

- [`torch::tanh`](tanh.md) - Standard hyperbolic tangent function
- [`torch::hardsigmoid`](hardsigmoid.md) - Hard sigmoid activation function
- [`torch::relu`](relu.md) - Rectified Linear Unit activation
- [`torch::relu6`](relu6.md) - ReLU clamped to [0, 6] range
- [`torch::clip`](clip.md) - General tensor clipping function 