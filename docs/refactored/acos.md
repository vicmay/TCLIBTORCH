# torch::acos

Computes the inverse cosine (arccosine) of each element in the input tensor.

## Syntax

### Current Syntax
```tcl
torch::acos tensor
```

### Named Parameter Syntax  
```tcl
torch::acos -input tensor
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input` (required): Input tensor name

### Positional Parameters
1. `tensor` (required): Input tensor name

## Description

The `torch::acos` function computes the inverse cosine (arccosine) of each element in the input tensor. The input values should be in the range [-1, 1], as the arccosine function is only defined for this domain. The output values will be in the range [0, π] radians.

## Mathematical Details

For a tensor element x where -1 ≤ x ≤ 1:
- `acos(x)` returns the angle θ such that `cos(θ) = x`
- The output range is [0, π] radians
- Special values:
  - `acos(-1) = π`
  - `acos(0) = π/2 ≈ 1.5708`
  - `acos(1) = 0`

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with values in valid range
set input [torch::tensor_create {0.0 0.5 1.0 -0.5 -1.0}]
set result [torch::acos $input]
# Result contains [π/2, π/3, 0, 2π/3, π] approximately
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set input [torch::tensor_create {0.0 0.5 1.0 -0.5 -1.0}]
set result [torch::acos -input $input]
```

### Mathematical Properties

```tcl
# Verify that acos is the inverse of cos
set angles [torch::tensor_create {0.0 1.0 2.0 3.0}]
set cosines [torch::cos $angles]
set recovered_angles [torch::acos $cosines]
# recovered_angles should be approximately equal to angles
```

### Working with Different Tensor Shapes

```tcl
# 2D tensor
set matrix [torch::ones {3 3}]
set result [torch::acos -input $matrix]  # All elements become 0

# 3D tensor  
set tensor3d [torch::zeros {2 2 2}]
set result [torch::acos -input $tensor3d]  # All elements become π/2
```

## Input Requirements

- **Domain**: Input values must be in the range [-1, 1]
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Any tensor shape is supported
- **Device**: CPU and CUDA tensors supported

## Output

Returns a new tensor with the same shape as the input, containing the arccosine values in radians.

## Error Handling

The function will raise an error if:
- Input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided

For input values outside [-1, 1], the behavior follows PyTorch conventions (typically returns NaN).

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify input
- Computational complexity: O(n) where n is the number of elements

## Common Use Cases

1. **Inverse Trigonometry**: Converting cosine values back to angles
2. **Computer Graphics**: Angle calculations for rotations and transformations
3. **Signal Processing**: Phase recovery in frequency domain analysis
4. **Machine Learning**: Activation functions and mathematical transformations

## Related Functions

- `torch::cos` - Cosine function (inverse of acos)
- `torch::asin` - Arcsine function
- `torch::atan` - Arctangent function
- `torch::sin` - Sine function
- `torch::tan` - Tangent function

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::acos $input_tensor]

# New named parameter syntax
set result [torch::acos -input $input_tensor]

# Both produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be provided in any order  
3. **Extensibility**: Easy to add optional parameters in the future
4. **Consistency**: Matches modern TCL conventions

## Technical Notes

- Implements PyTorch's `torch.acos()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions 