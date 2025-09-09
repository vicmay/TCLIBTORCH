# torch::cosh

Computes the hyperbolic cosine of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::cosh tensor
```

### Named Parameter Syntax  
```tcl
torch::cosh -input tensor
torch::cosh -tensor tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Input tensor name (alias for `-input`)

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::cosh` function computes the hyperbolic cosine of each element in the input tensor. The input tensor can contain values in any range, as the hyperbolic cosine function is defined for all real numbers. The output values will always be in the range [1, ∞).

## Mathematical Details

For a tensor element x:
- `cosh(x) = (e^x + e^(-x)) / 2` returns the hyperbolic cosine of x
- The output range is [1, ∞) (always greater than or equal to 1)
- The function is even: `cosh(-x) = cosh(x)`
- Special values:
  - `cosh(0) = 1`
  - `cosh(x)` grows exponentially for large |x|
  - `cosh(x)` approaches e^|x|/2 for large |x|

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with values
set values [torch::tensor_create {0.0 1.0 -1.0 2.0 -2.0}]
set result [torch::cosh $values]
# Result contains approximately [1, 1.543, 1.543, 3.762, 3.762]
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set values [torch::tensor_create {0.0 1.0 -1.0 2.0 -2.0}]
set result [torch::cosh -input $values]
```

#### Using -tensor alias
```tcl
# Alternative named parameter
set values [torch::tensor_create {0.0 1.0 2.0}]
set result [torch::cosh -tensor $values]
```

### Mathematical Properties

```tcl
# Hyperbolic cosine of zero is always 1
set zeros [torch::zeros {3 3}]
set result [torch::cosh $zeros]
# All elements in result will be 1.0

# Verify even function property: cosh(-x) = cosh(x)
set x [torch::tensor_create {1.0 2.0 3.0}]
set neg_x [torch::tensor_create {-1.0 -2.0 -3.0}]  # Negative values
set cosh_x [torch::cosh $x]
set cosh_neg_x [torch::cosh $neg_x]
# cosh_x and cosh_neg_x should be identical (even function)
```

### Working with Different Tensor Shapes

```tcl
# 1D tensor
set vector [torch::tensor_create {0.0 1.0 2.0 3.0}]
set result [torch::cosh -input $vector]

# 2D tensor
set matrix [torch::ones {3 3}]
set result [torch::cosh -input $matrix]  # All elements become cosh(1) ≈ 1.543

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::cosh -input $tensor3d]  # All elements become 1.0
```

### Hyperbolic Calculations

```tcl
# Calculate hyperbolic cosine for common values
set common_values [torch::tensor_create {0.0 0.5 1.0 1.5 2.0}]
set cosh_values [torch::cosh -input $common_values]
# Results: [1, 1.128, 1.543, 2.352, 3.762] approximately

# Demonstrate exponential growth for large values
set large_values [torch::tensor_create {3.0 4.0 5.0}]
set cosh_large [torch::cosh -input $large_values]
# Results: [10.068, 27.308, 74.210] approximately
```

## Input Requirements

- **Domain**: All real numbers (no restrictions)
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported

## Output

Returns a new tensor with the same shape as the input, containing the hyperbolic cosine values in the range [1, ∞).

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input
- Computational complexity: O(n) where n is the number of elements
- Highly optimized using PyTorch's native hyperbolic cosine implementation

## Common Use Cases

1. **Hyperbolic Geometry**: Computing distances and angles in hyperbolic space
2. **Neural Networks**: Hyperbolic activation functions and special layers
3. **Signal Processing**: Processing exponentially growing or decaying signals
4. **Physics Simulations**: Modeling catenary curves, hanging chains, and cables
5. **Machine Learning**: Hyperbolic embeddings and representation learning
6. **Mathematical Analysis**: Solving differential equations with hyperbolic solutions

## Related Functions

- `torch::sinh` - Hyperbolic sine function (complementary hyperbolic function)
- `torch::tanh` - Hyperbolic tangent function
- `torch::acosh` - Inverse hyperbolic cosine function (inverse of cosh)
- `torch::cos` - Regular cosine function
- `torch::asinh` - Inverse hyperbolic sine function
- `torch::atanh` - Inverse hyperbolic tangent function

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::cosh $input_tensor]

# New named parameter syntax
set result [torch::cosh -input $input_tensor]

# Alternative with -tensor alias
set result [torch::cosh -tensor $input_tensor]

# All three produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions
5. **Self-documenting**: Code is more readable and maintainable

## Mathematical Properties Reference

### Fundamental Identities
- `cosh²(x) - sinh²(x) = 1` (Fundamental hyperbolic identity)
- `cosh(-x) = cosh(x)` (even function)
- `cosh(x) = (e^x + e^(-x)) / 2` (exponential definition)

### Addition Formulas
- `cosh(x + y) = cosh(x)cosh(y) + sinh(x)sinh(y)`
- `cosh(x - y) = cosh(x)cosh(y) - sinh(x)sinh(y)`

### Range and Domain
- **Domain**: All real numbers (-∞, ∞)
- **Range**: [1, ∞)
- **Minimum value**: 1 (achieved at x = 0)

### Asymptotic Behavior
- For large positive x: `cosh(x) ≈ e^x / 2`
- For large negative x: `cosh(x) ≈ e^|x| / 2`

## Relationship to Trigonometric Functions

### Connection to Regular Cosine
- `cosh(ix) = cos(x)` where i is the imaginary unit
- `cos(ix) = cosh(x)` (Euler's formula connection)

### Inverse Relationship
- `acosh(cosh(x)) = |x|` for all real x
- `cosh(acosh(x)) = x` for x ≥ 1

## Technical Notes

- Implements PyTorch's `torch.cosh()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Uses highly optimized mathematical libraries (Intel MKL, cuBLAS)
- Numerically stable for large input values

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added `-tensor` parameter alias for enhanced flexibility

## Physical Interpretation

The hyperbolic cosine function describes:
- **Catenary curves**: The shape of hanging chains or cables
- **Hyperbolic geometry**: Distances in hyperbolic space
- **Wave equations**: Solutions to certain partial differential equations
- **Relativistic physics**: Time dilation and Lorentz transformations 