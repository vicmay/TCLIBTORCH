# torch::asin

Computes the inverse sine (arcsine) of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::asin tensor
```

### Named Parameter Syntax  
```tcl
torch::asin -input tensor
torch::asin -tensor tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name
- `-tensor` (required): Alias for `-input` parameter

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::asin` function computes the inverse sine (arcsine) of each element in the input tensor. The input values should be in the range [-1, 1], as the arcsine function is only defined for this domain. The output values will be in the range [-π/2, π/2] radians.

## Mathematical Details

For a tensor element x where -1 ≤ x ≤ 1:
- `asin(x)` returns the angle θ such that `sin(θ) = x`
- The output range is [-π/2, π/2] radians
- Special values:
  - `asin(-1) = -π/2 ≈ -1.5708`
  - `asin(0) = 0`
  - `asin(1) = π/2 ≈ 1.5708`

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with values in valid range
set input [torch::tensor_create {0.0 0.5 1.0 -0.5 -1.0}]
set result [torch::asin $input]
# Result contains [0, π/6, π/2, -π/6, -π/2] approximately
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set input [torch::tensor_create {0.0 0.5 1.0 -0.5 -1.0}]
set result [torch::asin -input $input]

# Alternative parameter name
set result [torch::asin -tensor $input]
```

### Mathematical Properties

```tcl
# Verify that asin is the inverse of sin
set angles [torch::tensor_create {-1.0 -0.5 0.0 0.5 1.0}]
set sines [torch::sin $angles]
set recovered_angles [torch::asin $sines]
# recovered_angles should be approximately equal to angles
```

### Working with Different Tensor Shapes

```tcl
# 2D tensor
set matrix [torch::ones {3 3}]
set result [torch::asin -input $matrix]  # All elements become π/2

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::asin -input $tensor3d]  # All elements become 0
```

## Input Requirements

- **Domain**: Input values must be in the range [-1, 1]
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported

## Output

Returns a new tensor with the same shape as the input, containing the arcsine values in radians.

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Too many positional arguments are provided

For input values outside [-1, 1], the behavior follows PyTorch conventions (typically returns NaN).

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input
- Computational complexity: O(n) where n is the number of elements

## Common Use Cases

1. **Inverse Trigonometry**: Converting sine values back to angles
2. **Computer Graphics**: Angle calculations for rotations and transformations
3. **Signal Processing**: Phase recovery and angle extraction
4. **Machine Learning**: Activation functions and mathematical transformations
5. **Physics Simulations**: Angular calculations and wave analysis

## Related Functions

- `torch::sin` - Sine function (inverse of asin)
- `torch::acos` - Arccosine function
- `torch::atan` - Arctangent function
- `torch::cos` - Cosine function
- `torch::tan` - Tangent function

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::asin $input_tensor]

# New named parameter syntax
set result [torch::asin -input $input_tensor]

# Alternative parameter name
set result [torch::asin -tensor $input_tensor]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions
5. **Multiple Aliases**: Both `-input` and `-tensor` are supported

## Technical Notes

- Implements PyTorch's `torch.asin()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Parameter aliases provide flexibility in naming conventions

## Comparison with Other Inverse Trigonometric Functions

| Function | Domain | Range | Use Case |
|----------|--------|-------|----------|
| `asin` | [-1, 1] | [-π/2, π/2] | Angle from sine value |
| `acos` | [-1, 1] | [0, π] | Angle from cosine value |
| `atan` | (-∞, ∞) | (-π/2, π/2) | Angle from tangent value |

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added `-tensor` parameter alias for flexibility 