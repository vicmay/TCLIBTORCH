# torch::atan2

Computes the element-wise arctangent of y/x given the signs of both y and x, returning the angle in radians.

## Syntax

### Current Syntax
```tcl
torch::atan2 y x
```

### Named Parameter Syntax  
```tcl
torch::atan2 -y y -x x
torch::atan2 -input1 y -input2 x
```

Both syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-y` (required): Y-coordinate tensor (first input)
- `-x` (required): X-coordinate tensor (second input)
- `-input1` (required): Alias for `-y` parameter  
- `-input2` (required): Alias for `-x` parameter

### Positional Parameters
1. `y` (required): Y-coordinate tensor (first input)
2. `x` (required): X-coordinate tensor (second input)

## Description

The `torch::atan2` function computes the element-wise arctangent of `y/x` given the signs of both `y` and `x`. This is the two-argument arctangent function that returns the angle θ in radians such that `x = r·cos(θ)` and `y = r·sin(θ)`, where `r = √(x² + y²)`. The function considers the signs of both inputs to determine which quadrant the angle should be in.

Unlike the regular `atan` function, `atan2` can handle cases where `x = 0` and provides the full range of angles from -π to π.

## Mathematical Details

For tensor elements y and x:
- `atan2(y, x)` returns the angle θ in radians where -π ≤ θ ≤ π
- The output represents the angle from the positive x-axis to the point (x, y)
- Special cases:
  - `atan2(0, x)` = 0 for x > 0
  - `atan2(0, x)` = π for x < 0  
  - `atan2(y, 0)` = π/2 for y > 0
  - `atan2(y, 0)` = -π/2 for y < 0
  - `atan2(0, 0)` = 0 (by convention)
- The function considers the signs of both arguments to determine the correct quadrant

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create coordinate tensors
set y [torch::tensor_create {1.0 0.0 -1.0}]
set x [torch::tensor_create {1.0 1.0 1.0}]
set angles [torch::atan2 $y $x]
# Results: π/4, 0, -π/4 (45°, 0°, -45°)
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set y [torch::tensor_create {1.0 0.0 -1.0}]
set x [torch::tensor_create {1.0 1.0 1.0}]
set angles [torch::atan2 -y $y -x $x]

# Alternative parameter names
set angles [torch::atan2 -input1 $y -input2 $x]

# Parameters can be in any order
set angles [torch::atan2 -x $x -y $y]
```

### Quadrant Analysis

```tcl
# Points in different quadrants
set y_vals [torch::tensor_create {1.0 -1.0 -1.0 1.0}]   # Quadrants: I, IV, III, II
set x_vals [torch::tensor_create {1.0 1.0 -1.0 -1.0}]   # Quadrants: I, IV, III, II
set angles [torch::atan2 -y $y_vals -x $x_vals]
# Results: π/4, -π/4, -3π/4, 3π/4
```

### Working with Different Tensor Shapes

```tcl
# 2D coordinate grids
set y_grid [torch::ones {3 3}]
set x_grid [torch::ones {3 3}]
set angle_grid [torch::atan2 -y $y_grid -x $x_grid]  # All elements = π/4

# 3D tensors
set y_tensor [torch::zeros {2 2 2}]
set x_tensor [torch::ones {2 2 2}]
set result [torch::atan2 -y $y_tensor -x $x_tensor]  # All elements = 0
```

### Special Cases

```tcl
# Handling division by zero cases
set y [torch::tensor_create {1.0 -1.0 0.0}]
set x [torch::zeros {3}]
set result [torch::atan2 -y $y -x $x]
# Results: π/2, -π/2, 0

# Origin case
set zeros_y [torch::zeros {2 2}]
set zeros_x [torch::zeros {2 2}]
set result [torch::atan2 -y $zeros_y -x $zeros_x]  # All elements = 0
```

## Input Requirements

- **Domain**: All real numbers for both inputs (no restrictions)
- **Data Type**: Floating-point tensors (float32, float64)
- **Shape**: Both tensors must have compatible shapes (broadcastable)
- **Device**: CPU and CUDA tensors supported
- **Broadcasting**: Standard PyTorch broadcasting rules apply

## Output

Returns a new tensor with the same shape as the broadcasted inputs, containing the arctangent values in radians (range: -π to π).

## Error Handling

The function will raise an error if:
- Either input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Incompatible tensor shapes (non-broadcastable)
- Too many or too few positional arguments

## Performance Considerations

- The operation is element-wise and parallelizable
- GPU acceleration available for CUDA tensors
- Memory usage: Creates a new tensor, doesn't modify inputs
- Computational complexity: O(n) where n is the number of elements
- More computationally intensive than basic arithmetic operations
- Broadcasting may affect memory usage and performance

## Common Use Cases

1. **Computer Vision**: Converting Cartesian to polar coordinates
2. **Robotics**: Calculating joint angles and orientations
3. **Physics Simulations**: Computing direction angles in 2D space
4. **Signal Processing**: Phase calculations in complex analysis
5. **Machine Learning**: Attention mechanisms and angular features
6. **Navigation**: Bearing and heading calculations

## Related Functions

- `torch::atan` - Single-argument arctangent function (limited range)
- `torch::sin` - Sine function (inverse of arctangent operations)
- `torch::cos` - Cosine function (complementary to arctangent)
- `torch::tan` - Tangent function (ratio y/x without quadrant info)
- `torch::hypot` - Hypotenuse calculation (√(x² + y²))
- `torch::angle` - Complex number phase calculation

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set result [torch::atan2 $y_tensor $x_tensor]

# New named parameter syntax
set result [torch::atan2 -y $y_tensor -x $x_tensor]

# Alternative parameter names
set result [torch::atan2 -input1 $y_tensor -input2 $x_tensor]

# Parameters can be in any order
set result [torch::atan2 -x $x_tensor -y $y_tensor]

# All produce identical results
```

### Benefits of Named Parameters

1. **Clarity**: Parameter roles are explicit (y vs x coordinates)
2. **Flexibility**: Parameters can be provided in any order
3. **Consistency**: Matches modern TCL conventions
4. **Error Prevention**: Reduces confusion about parameter order
5. **Multiple Aliases**: Both mathematical (-y, -x) and generic (-input1, -input2) names

## Technical Notes

- Implements PyTorch's `torch.atan2()` function
- Preserves tensor properties (device, requires_grad, etc.)
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Follows standard mathematical conventions for quadrant determination
- Handles edge cases consistently with IEEE floating-point standards

## Mathematical Background

The two-argument arctangent function `atan2(y, x)` is defined as:
- The angle θ such that `tan(θ) = y/x` and θ is in the correct quadrant
- Range: -π ≤ θ ≤ π (full circle coverage)
- Quadrant I (x > 0, y > 0): 0 < θ < π/2
- Quadrant II (x < 0, y > 0): π/2 < θ < π  
- Quadrant III (x < 0, y < 0): -π < θ < -π/2
- Quadrant IV (x > 0, y < 0): -π/2 < θ < 0

## Comparison with Related Functions

| Function | Inputs | Range | Use Case |
|----------|--------|-------|----------|
| `atan(z)` | 1 (z = y/x) | (-π/2, π/2) | Simple angle from ratio |
| `atan2(y, x)` | 2 (y, x) | [-π, π] | Full angle with quadrant |
| `asin(y/r)` | 1 (normalized) | [-π/2, π/2] | Angle from sine value |
| `acos(x/r)` | 1 (normalized) | [0, π] | Angle from cosine value |

## Broadcasting Examples

```tcl
# Scalar and tensor
set y_scalar [torch::tensor_create {1.0}]
set x_matrix [torch::ones {3 3}]
set result [torch::atan2 -y $y_scalar -x $x_matrix]  # Broadcasts to 3x3

# Different but compatible shapes
set y_row [torch::ones {1 4}]
set x_col [torch::ones {3 1}]
set result [torch::atan2 -y $y_row -x $x_col]  # Broadcasts to 3x4
```

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- Added both mathematical (-y, -x) and generic (-input1, -input2) parameter names
- Enhanced documentation with mathematical background and quadrant analysis 