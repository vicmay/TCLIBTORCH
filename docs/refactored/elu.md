# torch::elu

## Description

The **Exponential Linear Unit (ELU)** activation function applies element-wise ELU transformation to the input tensor. ELU provides smoothly saturating nonlinearity for negative inputs while maintaining identity function for positive inputs.

## Mathematical Formula

```
elu(x) = max(0, x) + min(0, α * (exp(x) - 1))
```

For practical implementation:
- If x ≥ 0: `elu(x) = x`
- If x < 0: `elu(x) = α * (exp(x) - 1)`

Where:
- `x` is the input value
- `α` (alpha) is a positive scalar parameter that controls the saturation value for negative inputs (default: 1.0)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::elu tensor ?alpha?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::elu -input tensor ?-alpha value?
torch::elu -tensor tensor ?-alpha value?
```

### CamelCase Alias
```tcl
torch::Elu tensor ?alpha?
torch::Elu -input tensor ?-alpha value?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` / `-input` / `-tensor` | string | **required** | Input tensor handle |
| `alpha` / `-alpha` | double | 1.0 | Positive scaling factor for negative values |

## Returns

Returns a tensor handle containing the element-wise ELU activation of the input tensor.

## Examples

### Basic Usage
```tcl
# Create input tensor
set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32]

# Apply ELU with default alpha (1.0)
set result [torch::elu $input]
# Result: approximately {-0.8647 -0.6321 0.0 1.0 2.0}

# Apply ELU with custom alpha
set result [torch::elu $input 2.0]
# Negative values will be scaled by alpha=2.0
```

### Named Parameter Syntax
```tcl
# Using -input parameter
set result [torch::elu -input $input]

# Using -input with custom alpha
set result [torch::elu -input $input -alpha 0.5]

# Using -tensor alias
set result [torch::elu -tensor $input -alpha 2.0]
```

### CamelCase Syntax
```tcl
# CamelCase positional
set result [torch::Elu $input]

# CamelCase with named parameters
set result [torch::Elu -input $input -alpha 1.5]
```

### Multi-dimensional Tensors
```tcl
# 2D tensor
set input [torch::tensor_create -data {-1.0 0.0 1.0 -2.0} -dtype float32]
set input_2d [torch::tensor_reshape $input {2 2}]
set result [torch::elu $input_2d]

# 3D tensor with custom alpha
set input_3d [torch::zeros {2 3 4} float32 cpu false]
set result [torch::elu -input $input_3d -alpha 0.8]
```

### Integration with Neural Networks
```tcl
# ELU activation in a neural network layer
proc create_elu_layer {input_size output_size alpha} {
    # Create linear transformation
    set weights [torch::tensor_randn {$output_size $input_size} cpu float32]
    set bias [torch::zeros {$output_size} float32 cpu false]
    
    return [list $weights $bias $alpha]
}

proc forward_elu_layer {layer_params input} {
    lassign $layer_params weights bias alpha
    
    # Linear transformation: y = Wx + b
    set linear_out [torch::addmm $bias $weights $input]
    
    # Apply ELU activation
    set activated [torch::elu -input $linear_out -alpha $alpha]
    
    return $activated
}

# Usage
set layer [create_elu_layer 10 5 1.2]
set input [torch::tensor_randn {1 10} cpu float32]
set output [forward_elu_layer $layer $input]
```

## Mathematical Properties

### Continuity and Differentiability
- **Continuous**: ELU is continuous everywhere, including at x = 0
- **Differentiable**: ELU has defined derivatives everywhere
- **Smooth**: Unlike ReLU, ELU doesn't have sharp corners

### Derivative
```
d/dx elu(x) = {
    1           if x > 0
    α * exp(x)  if x ≤ 0
}
```

### Advantages over ReLU
1. **No dying neurons**: ELU has non-zero gradients for negative inputs
2. **Smooth activation**: Continuous derivatives help with optimization
3. **Centered around zero**: Mean activations closer to zero, reducing internal covariate shift
4. **Noise robustness**: Saturating negative values provide robustness to noise

## Error Handling

### Invalid Parameters
```tcl
# Missing required parameter
catch {torch::elu} error
# Error: wrong # args: should be "torch::elu tensor ?alpha?"

# Invalid tensor handle
catch {torch::elu "invalid_tensor"} error
# Error: Invalid tensor name

# Invalid alpha value
catch {torch::elu $tensor "not_a_number"} error
# Error: invalid alpha value

# Zero or negative alpha
catch {torch::elu $tensor 0.0} error
# Error: alpha must be > 0

catch {torch::elu $tensor -1.0} error
# Error: alpha must be > 0
```

### Named Parameter Errors
```tcl
# Missing required input parameter
catch {torch::elu -alpha 1.0} error
# Error: required parameter -input missing

# Unknown parameter
catch {torch::elu -input $tensor -unknown_param 1.0} error
# Error: unknown option -unknown_param

# Missing value for parameter
catch {torch::elu -input $tensor -alpha} error
# Error: wrong # args: should be "torch::elu -input tensor ?-alpha value?"
```

## Performance Considerations

### Memory Usage
- **In-place operation**: No additional memory overhead
- **Same shape**: Output tensor has identical shape to input tensor
- **Type preservation**: Output maintains input tensor's data type and device

### Computational Complexity
- **O(n)** time complexity where n is the number of elements
- **Vectorized**: Efficiently computed using vectorized operations
- **GPU optimized**: When using CUDA tensors, computation runs on GPU

### Numerical Stability
```tcl
# Handle very large negative values (exp(-large) ≈ 0)
set large_negative [torch::tensor_create -data {-100.0 -50.0 -10.0} -dtype float32]
set result [torch::elu $large_negative]
# Result approaches [-α, -α, -α] for very negative values

# Small positive values remain unchanged
set small_positive [torch::tensor_create -data {0.001 0.01 0.1} -dtype float32]
set result [torch::elu $small_positive]
# Result: {0.001 0.01 0.1} (unchanged)
```

## Comparison with Other Activations

| Activation | Formula | Advantages | Disadvantages |
|------------|---------|------------|---------------|
| **ELU** | `x if x≥0 else α*(e^x-1)` | Smooth, no dying neurons, mean≈0 | Computationally expensive (exp) |
| **ReLU** | `max(0, x)` | Fast, simple | Dying neurons, not smooth |
| **LeakyReLU** | `x if x≥0 else αx` | No dying neurons | Not smooth, linear negative |
| **SELU** | `scale*elu(x)` | Self-normalizing properties | Requires specific initialization |

### When to Use ELU
- **Deep networks**: Where gradient flow and smoothness matter
- **Computer vision**: CNNs benefit from smooth activations
- **When avoiding dying neurons**: Alternative to ReLU in problematic layers
- **Regression tasks**: Where negative outputs are meaningful

## Migration Guide

### From Simple ELU Usage
```tcl
# OLD: Basic ELU (if using custom implementation)
set activated [custom_elu $input]

# NEW: Using torch::elu
set activated [torch::elu $input]
```

### From ReLU to ELU
```tcl
# OLD: ReLU activation
set activated [torch::relu $input]

# NEW: ELU activation (smoother alternative)
set activated [torch::elu $input]

# NEW: ELU with custom alpha for stronger negative saturation
set activated [torch::elu -input $input -alpha 2.0]
```

### From Positional to Named Parameters
```tcl
# OLD: Positional syntax
set result [torch::elu $tensor 1.5]

# NEW: Named parameter syntax (recommended)
set result [torch::elu -input $tensor -alpha 1.5]

# ALTERNATIVE: Using -tensor alias
set result [torch::elu -tensor $tensor -alpha 1.5]
```

### Batch Processing Migration
```tcl
# OLD: Processing individual tensors
foreach tensor $tensor_list {
    set result [torch::elu $tensor]
    lappend results $result
}

# NEW: Using vectorized operations (more efficient)
set batched_input [torch::stack $tensor_list 0]
set batched_result [torch::elu $batched_input]
# Split back if needed: torch::unbind $batched_result 0
```

## Related Functions

- [`torch::relu`](relu.md) - Rectified Linear Unit activation
- [`torch::selu`](selu.md) - Scaled Exponential Linear Unit
- [`torch::leaky_relu`](leaky_relu.md) - Leaky ReLU activation  
- [`torch::gelu`](gelu.md) - Gaussian Error Linear Unit
- [`torch::swish`](swish.md) - Swish activation function
- [`torch::mish`](mish.md) - Mish activation function

## Technical Notes

### Alpha Parameter Selection
- **Default (α=1.0)**: Good general-purpose choice
- **Small α (0.1-0.5)**: Gentler negative saturation
- **Large α (2.0-5.0)**: Stronger negative response
- **Network-specific**: May require hyperparameter tuning

### Gradient Properties
- **Positive inputs**: Gradient = 1 (identity)
- **Negative inputs**: Gradient = α * exp(x), always positive
- **At zero**: Gradient = α, ensuring smooth transition

### Implementation Details
- Uses efficient vectorized operations from LibTorch
- Supports all standard tensor data types (float32, float64)
- Compatible with autograd for gradient computation
- Thread-safe for concurrent operations

---

*This documentation covers torch::elu implementation in LibTorch TCL extension. For additional examples and advanced usage patterns, refer to the test suite in `tests/refactored/elu_test.tcl`.* 