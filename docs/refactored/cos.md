# torch::cos

Computes the cosine of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::cos tensor
```

### Named Parameter Syntax  
```tcl
torch::cos -input tensor
torch::cos -tensor tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Input tensor name (alias for `-input`)

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::cos` function computes the cosine of each element in the input tensor. The input tensor can contain values in any range, as the cosine function is defined for all real numbers. The output values will always be in the range [-1, 1].

## Mathematical Details

For a tensor element x:
- `cos(x)` returns the cosine of angle x (measured in radians)
- The output range is [-1, 1]
- The function is periodic with period 2π
- Special values:
  - `cos(0) = 1`
  - `cos(π/2) ≈ 0`
  - `cos(π) = -1`
  - `cos(3π/2) ≈ 0`
  - `cos(2π) = 1`

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with angle values in radians
set angles [torch::tensor_create {0.0 1.5708 3.14159 4.71239 6.28318}]
set result [torch::cos $angles]
# Result contains approximately [1, 0, -1, 0, 1]
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set angles [torch::tensor_create {0.0 1.5708 3.14159 4.71239 6.28318}]
set result [torch::cos -input $angles]
```

#### Using -tensor alias
```tcl
# Alternative named parameter
set angles [torch::tensor_create {0.0 1.5708 3.14159}]
set result [torch::cos -tensor $angles]
```

### Mathematical Properties

```tcl
# Cosine of zero is always 1
set zeros [torch::zeros {3 3}]
set result [torch::cos $zeros]
# All elements in result will be 1.0

# Verify periodic property: cos(x) = cos(x + 2π)
set x [torch::tensor_create {1.0 2.0 3.0}]
set x_plus_2pi [torch::tensor_create {7.28318 8.28318 9.28318}]  # Added 2π ≈ 6.28318
set cos_x [torch::cos $x]
set cos_x_plus_2pi [torch::cos $x_plus_2pi]
# cos_x and cos_x_plus_2pi should be approximately equal
```

### Working with Different Tensor Shapes

```tcl
# 1D tensor
set vector [torch::tensor_create {0.0 1.0 2.0 3.0}]
set result [torch::cos -input $vector]

# 2D tensor
set matrix [torch::ones {3 3}]
set result [torch::cos -input $matrix]  # All elements become cos(1) ≈ 0.540

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::cos -input $tensor3d]  # All elements become 1.0
```

### Trigonometric Calculations

```tcl
# Calculate cosine for common angles
set common_angles [torch::tensor_create {0.0 0.7854 1.5708 2.3562 3.14159}]
# Angles: 0°, 45°, 90°, 135°, 180° in radians
set cosines [torch::cos -input $common_angles]
# Results: [1, 0.707, 0, -0.707, -1] approximately
```

## Input Requirements

- **Domain**: All real numbers (no restrictions)
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported

## Output

Returns a new tensor with the same shape as the input, containing the cosine values in the range [-1, 1].

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
- Highly optimized using PyTorch's native cosine implementation

## Common Use Cases

1. **Trigonometric Calculations**: Computing cosine values for angle measurements
2. **Signal Processing**: Generating cosine waves and analyzing periodic signals
3. **Computer Graphics**: Calculating rotations, transformations, and lighting
4. **Machine Learning**: Positional encoding in transformers, periodic activations
5. **Physics Simulations**: Wave equations, oscillatory motion, and periodic phenomena
6. **Fourier Analysis**: Decomposing signals into frequency components

## Related Functions

- `torch::acos` - Arccosine function (inverse of cos)
- `torch::sin` - Sine function (complementary trigonometric function)
- `torch::tan` - Tangent function
- `torch::cosh` - Hyperbolic cosine function
- `torch::asin` - Arcsine function
- `torch::atan` - Arctangent function

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::cos $input_tensor]

# New named parameter syntax
set result [torch::cos -input $input_tensor]

# Alternative with -tensor alias
set result [torch::cos -tensor $input_tensor]

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
- `cos²(x) + sin²(x) = 1` (Pythagorean identity)
- `cos(-x) = cos(x)` (even function)
- `cos(x + 2π) = cos(x)` (periodic with period 2π)

### Addition Formulas
- `cos(a + b) = cos(a)cos(b) - sin(a)sin(b)`
- `cos(a - b) = cos(a)cos(b) + sin(a)sin(b)`

### Range and Domain
- **Domain**: All real numbers (-∞, ∞)
- **Range**: [-1, 1]
- **Period**: 2π

## Technical Notes

- Implements PyTorch's `torch.cos()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Uses highly optimized mathematical libraries (Intel MKL, cuBLAS)

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added `-tensor` parameter alias for enhanced flexibility 