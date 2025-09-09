# torch::pow

## Description
Computes the power operation element-wise: `base^exponent`. This function supports both positional and named parameter syntax for maximum flexibility.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::pow base exponent
```

### Named Parameter Syntax (Recommended)
```tcl
torch::pow -base base -exponent exponent
torch::pow -input1 base -input2 exponent
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `base` / `input1` | tensor | The base tensor | Yes |
| `exponent` / `input2` | tensor | The exponent tensor | Yes |

## Returns
Returns a new tensor containing the element-wise power operation results.

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create tensors
set base [torch::tensor_create {2.0 3.0 4.0}]
set exponent [torch::tensor_create {2.0 2.0 2.0}]

# Compute power: [4.0, 9.0, 16.0]
set result [torch::pow $base $exponent]
```

### Basic Usage - Named Parameters
```tcl
# Create tensors
set base [torch::tensor_create {2.0 3.0 4.0}]
set exponent [torch::tensor_create {3.0 2.0 1.0}]

# Compute power: [8.0, 9.0, 4.0]
set result [torch::pow -base $base -exponent $exponent]
```

### Alternative Parameter Names
```tcl
# Using input1/input2 parameter names
set base [torch::tensor_create {5.0}]
set exponent [torch::tensor_create {3.0}]

# Compute 5^3 = 125.0
set result [torch::pow -input1 $base -input2 $exponent]
```

### Mathematical Operations
```tcl
# Square operation: x^2
set x [torch::tensor_create {1.0 2.0 3.0 4.0}]
set two [torch::tensor_create {2.0 2.0 2.0 2.0}]
set squares [torch::pow $x $two]  # [1.0, 4.0, 9.0, 16.0]

# Square root: x^0.5
set numbers [torch::tensor_create {4.0 9.0 16.0 25.0}]
set half [torch::tensor_create {0.5 0.5 0.5 0.5}]
set roots [torch::pow -base $numbers -exponent $half]  # [2.0, 3.0, 4.0, 5.0]

# Power of zero: x^0 = 1
set base [torch::tensor_create {5.0 10.0 100.0}]
set zero [torch::tensor_create {0.0 0.0 0.0}]
set ones [torch::pow $base $zero]  # [1.0, 1.0, 1.0]
```

### Broadcasting Example
```tcl
# Base: 2x2 matrix, Exponent: scalar (broadcasts automatically)
set base [torch::tensor_create {2.0 3.0 4.0 5.0} -shape {2 2}]
set exponent [torch::tensor_create {2.0}]

# Each element raised to power 2
set result [torch::pow -base $base -exponent $exponent]
```

### Multi-dimensional Tensors
```tcl
# 3D tensors
set base [torch::ones {2 3 4}]
set exponent [torch::full {2 3 4} 3.0]

# All elements: 1^3 = 1
set result [torch::pow $base $exponent]
```

## Mathematical Properties

### Identity Operations
- `x^0 = 1` for any x ≠ 0
- `x^1 = x` for any x
- `1^y = 1` for any y

### Special Cases
- `0^0` is implementation-defined (typically 1)
- `x^(-y) = 1/(x^y)` for x ≠ 0
- Negative bases with fractional exponents may produce complex results

## Error Handling

### Invalid Parameters
```tcl
# Missing arguments
catch {torch::pow} error
# Returns: "Usage: torch::pow base exponent | torch::pow -base base -exponent exponent"

# Missing second argument
catch {torch::pow $base} error
# Returns: "Usage: torch::pow base exponent"

# Missing parameter value
catch {torch::pow -base} error
# Returns: "Missing value for parameter"

# Unknown parameter
catch {torch::pow -base $base -exponent $exp -invalid param} error
# Returns: "Unknown parameter: -invalid. Valid parameters are: -base, -exponent"
```

### Invalid Tensor Names
```tcl
catch {torch::pow nonexistent_tensor $exponent} error
# Returns: "Invalid base tensor name"

catch {torch::pow $base nonexistent_tensor} error
# Returns: "Invalid exponent tensor name"
```

## Technical Notes

### Implementation Details
- Uses PyTorch's `tensor.pow(exponent)` operation
- Supports automatic broadcasting between base and exponent tensors
- Handles different tensor shapes and sizes efficiently
- Element-wise operation preserves tensor structure

### Performance Considerations
- Very fast for same-sized tensors (no broadcasting overhead)
- Broadcasting adds minimal computational cost
- GPU acceleration available when using CUDA tensors
- Memory usage is O(max(base_size, exponent_size))

### Data Type Support
- Supports all floating-point tensor types (float32, float64)
- Integer tensors supported for positive integer exponents
- Mixed precision operations handled automatically

## Compatibility

### Backward Compatibility
The positional syntax (`torch::pow base exponent`) is fully supported and will continue to work with existing code.

### Migration Guide
```tcl
# Old code (still works)
set result [torch::pow $base $exponent]

# New code (recommended for clarity)
set result [torch::pow -base $base -exponent $exponent]
```

## Related Commands
- `torch::sqrt` - Square root (equivalent to pow with exponent 0.5)
- `torch::square` - Square operation (equivalent to pow with exponent 2)
- `torch::exp` - Exponential function (e^x)
- `torch::log` - Natural logarithm (inverse of exp)

## See Also
- [tensor_create](tensor_create.md) - Creating tensors
- [Mathematical Operations](../mathematical_operations.md) - Overview of math functions
- [Broadcasting](../broadcasting.md) - Understanding tensor broadcasting 