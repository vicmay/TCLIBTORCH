# torch::asinh

Computes the inverse hyperbolic sine (area hyperbolic sine) of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::asinh tensor
```

### Named Parameter Syntax  
```tcl
torch::asinh -input tensor
torch::asinh -tensor tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Alias for `-input` parameter

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::asinh` function computes the inverse hyperbolic sine (area hyperbolic sine) of each element in the input tensor. Unlike the regular inverse sine function (`asin`), the hyperbolic inverse sine function has no domain restrictions and can accept any real number. The output values are real numbers in the range (-∞, ∞).

## Mathematical Details

For any tensor element x (where x ∈ ℝ):
- `asinh(x)` returns the value y such that `sinh(y) = x`
- The output range is (-∞, ∞) - all real numbers
- Mathematical formula: `asinh(x) = ln(x + √(x² + 1))`
- Special values:
  - `asinh(0) = 0`
  - `asinh(1) ≈ 0.8814`
  - `asinh(-1) ≈ -0.8814`
  - `asinh(x)` is an odd function: `asinh(-x) = -asinh(x)`

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with various values (no domain restrictions)
set input [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0}]
set result [torch::asinh $input]
# Result contains asinh values for each element
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set input [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0}]
set result [torch::asinh -input $input]

# Alternative parameter name
set result [torch::asinh -tensor $input]
```

### Mathematical Properties

```tcl
# Verify that asinh is the inverse of sinh
set values [torch::tensor_create {-2.0 -1.0 0.0 1.0 2.0}]
set sinh_values [torch::sinh $values]
set recovered_values [torch::asinh $sinh_values]
# recovered_values should be approximately equal to values
```

### Working with Different Tensor Shapes

```tcl
# 2D tensor
set matrix [torch::ones {3 3}]
set result [torch::asinh -input $matrix]  # All elements become asinh(1) ≈ 0.8814

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::asinh -input $tensor3d]  # All elements become 0
```

### Large Value Handling

```tcl
# asinh can handle very large values (no domain restrictions)
set large_values [torch::tensor_create {100.0 1000.0 10000.0}]
set result [torch::asinh -input $large_values]

# Negative large values
set negative_large [torch::tensor_create {-100.0 -1000.0 -10000.0}]
set result [torch::asinh -input $negative_large]
```

## Input Requirements

- **Domain**: All real numbers (no restrictions, unlike `asin`)
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported

## Output

Returns a new tensor with the same shape as the input, containing the inverse hyperbolic sine values.

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Too many positional arguments are provided

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input
- Computational complexity: O(n) where n is the number of elements
- More computationally intensive than basic arithmetic operations

## Common Use Cases

1. **Hyperbolic Geometry**: Calculations in hyperbolic space
2. **Physics Simulations**: Relativistic calculations and transformations
3. **Signal Processing**: Non-linear transformations and analysis
4. **Machine Learning**: Activation functions and mathematical transformations
5. **Numerical Analysis**: Inverse transformations and mathematical modeling
6. **Statistics**: Transformations for data analysis

## Related Functions

- `torch::sinh` - Hyperbolic sine function (inverse of asinh)
- `torch::asin` - Inverse sine function (domain-restricted)
- `torch::acosh` - Inverse hyperbolic cosine function
- `torch::atanh` - Inverse hyperbolic tangent function
- `torch::cosh` - Hyperbolic cosine function
- `torch::tanh` - Hyperbolic tangent function

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::asinh $input_tensor]

# New named parameter syntax
set result [torch::asinh -input $input_tensor]

# Alternative parameter name
set result [torch::asinh -tensor $input_tensor]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions
5. **Multiple Aliases**: Both `-input` and `-tensor` are supported

## Technical Notes

- Implements PyTorch's `torch.asinh()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Parameter aliases provide flexibility in naming conventions
- No domain restrictions unlike `asin` function

## Comparison with Other Inverse Functions

| Function | Domain | Range | Use Case |
|----------|--------|-------|----------|
| `asin` | [-1, 1] | [-π/2, π/2] | Angle from sine value |
| `asinh` | (-∞, ∞) | (-∞, ∞) | Hyperbolic angle from sinh value |
| `acos` | [-1, 1] | [0, π] | Angle from cosine value |
| `acosh` | [1, ∞) | [0, ∞) | Hyperbolic angle from cosh value |
| `atan` | (-∞, ∞) | (-π/2, π/2) | Angle from tangent value |
| `atanh` | (-1, 1) | (-∞, ∞) | Hyperbolic angle from tanh value |

## Mathematical Background

The inverse hyperbolic sine function is the inverse of the hyperbolic sine function:
- If `y = sinh(x)`, then `x = asinh(y)`
- Formula: `asinh(x) = ln(x + √(x² + 1))`
- Derivative: `d/dx asinh(x) = 1/√(x² + 1)`
- It's an odd function: `asinh(-x) = -asinh(x)`
- It's strictly increasing and continuous

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added `-tensor` parameter alias for flexibility
- Enhanced documentation with mathematical background 