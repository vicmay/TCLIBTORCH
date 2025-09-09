# torch::floor

## Overview

The `torch::floor` command computes the element-wise floor (largest integer ≤ x) of the input tensor. This operation rounds each element down to the nearest integer.

## Syntax

### Modern Syntax (Named Parameters)
```tcl
torch::floor -input <tensor_name>
torch::floor -tensor <tensor_name>

# CamelCase alias
torch::Floor -input <tensor_name>
torch::Floor -tensor <tensor_name>
```

### Legacy Syntax (Positional Parameters)
```tcl
torch::floor <tensor_name>

# CamelCase alias
torch::Floor <tensor_name>
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` or `-tensor` | string | Yes | Name of the input tensor |

For positional syntax, the tensor name is the first and only argument.

## Return Value

Returns a tensor handle containing the element-wise floor of the input tensor. The output tensor has the same shape and device as the input tensor.

## Mathematical Foundation

The floor function is defined as:
```
floor(x) = ⌊x⌋ = max{n ∈ ℤ : n ≤ x}
```

**Key Properties:**
- For positive numbers: `floor(2.7) = 2`, `floor(3.0) = 3`
- For negative numbers: `floor(-2.3) = -3`, `floor(-3.0) = -3`
- For zero: `floor(0.0) = 0`, `floor(-0.0) = 0`
- For integers: `floor(n) = n` (identity)

**Important Note:** The floor of a negative non-integer is always one less than the truncation. For example:
- `floor(-2.3) = -3` (not -2)
- `floor(-0.1) = -1` (not 0)

## Examples

### Basic Usage

#### Positive Values
```tcl
# Create tensor with positive fractional values
set input [torch::tensor_create -data {1.2 2.7 3.9 4.1} -dtype float32]

# Modern syntax
set result [torch::floor -input $input]

# Legacy syntax  
set result [torch::floor $input]

# CamelCase alias
set result [torch::Floor -input $input]
# Result: tensor with values [1.0, 2.0, 3.0, 4.0]
```

#### Negative Values
```tcl
# Create tensor with negative values
set input [torch::tensor_create -data {-1.2 -2.7 -3.9 -4.1} -dtype float32]

set result [torch::floor -input $input]
# Result: tensor with values [-2.0, -3.0, -4.0, -5.0]
```

#### Mixed Values
```tcl
# Create tensor with mixed positive and negative values
set input [torch::tensor_create -data {-2.3 -0.5 0.0 1.7 2.8} -dtype float32]

set result [torch::floor -input $input]
# Result: tensor with values [-3.0, -1.0, 0.0, 1.0, 2.0]
```

### Multidimensional Tensors

```tcl
# 2D tensor
set input [torch::tensor_create -data {1.7 2.3 3.9 4.1} -shape {2 2} -dtype float32]

set result [torch::floor -input $input]
# Result: 2x2 tensor with floor values

# 3D tensor
set input [torch::zeros {2 3 4}]
set result [torch::floor -input $input]
# Result: 2x3x4 tensor of zeros
```

### Edge Cases

#### Integer Values
```tcl
# Floor of integers remains unchanged
set input [torch::tensor_create -data {1.0 2.0 -3.0 -4.0} -dtype float32]

set result [torch::floor -input $input]
# Result: [1.0, 2.0, -3.0, -4.0] (unchanged)
```

#### Boundary Values
```tcl
# Fractional boundary cases
set input [torch::tensor_create -data {0.5 -0.5 1.5 -1.5} -dtype float32]

set result [torch::floor -input $input]
# Result: [0.0, -1.0, 1.0, -2.0]
```

#### Very Large Numbers
```tcl
# Large magnitude values
set input [torch::tensor_create -data {1000000.7 -1000000.3} -dtype float32]

set result [torch::floor -input $input]
# Result: [1000000.0, -1000001.0]
```

## Data Type Support

The `torch::floor` function supports various floating-point data types:

```tcl
# Float32
set input [torch::tensor_create -data {2.7 3.2} -dtype float32]
set result [torch::floor -input $input]

# Float64
set input [torch::tensor_create -data {2.7 3.2} -dtype float64]
set result [torch::floor -input $input]
```

**Note:** Integer tensors will also work, but the floor operation is essentially a no-op for integers.

## Error Handling

### Common Errors

```tcl
# Missing tensor argument
catch {torch::floor} error
# Error: "Usage: torch::floor tensor | torch::floor -input tensor"

# Invalid tensor name
catch {torch::floor invalid_tensor} error  
# Error: "Invalid tensor name"

# Missing parameter value
catch {torch::floor -input} error
# Error: "Named parameter requires a value"

# Unknown parameter
catch {torch::floor -unknown param} error
# Error: "Unknown parameter: -unknown"

# Too many positional arguments
set tensor [torch::zeros {3}]
catch {torch::floor $tensor extra_arg} error
# Error: "Wrong number of positional arguments. Expected: torch::floor tensor"
```

### Error Messages

| Error Condition | Error Message |
|-----------------|---------------|
| No arguments | "Usage: torch::floor tensor \| torch::floor -input tensor" |
| Invalid tensor name | "Invalid tensor name" |
| Missing parameter value | "Named parameter requires a value" |
| Unknown parameter | "Unknown parameter: \<param\>" |
| Too many positional args | "Wrong number of positional arguments. Expected: torch::floor tensor" |

## Performance Considerations

- **Element-wise operation**: The floor function is applied to each tensor element independently
- **In-place capability**: Creates a new tensor; does not modify the input tensor
- **Memory usage**: Requires memory for the output tensor equal to the input tensor size
- **Computational complexity**: O(n) where n is the number of elements in the tensor

## Mathematical Comparison with Related Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `torch::floor` | Round down | `floor(2.7) = 2`, `floor(-2.3) = -3` |
| `torch::ceil` | Round up | `ceil(2.3) = 3`, `ceil(-2.7) = -2` |
| `torch::round` | Round to nearest | `round(2.3) = 2`, `round(2.7) = 3` |
| `torch::trunc` | Truncate toward zero | `trunc(2.7) = 2`, `trunc(-2.7) = -2` |

## Migration Guide

### From Legacy to Modern Syntax

**Before (Legacy):**
```tcl
set result [torch::floor $input_tensor]
```

**After (Modern):**
```tcl
set result [torch::floor -input $input_tensor]
# or
set result [torch::floor -tensor $input_tensor]
```

### Benefits of Modern Syntax

1. **Explicit parameter naming**: `-input` makes the code self-documenting
2. **Extensibility**: Easy to add new parameters in future versions
3. **Consistency**: Matches the pattern used across other refactored commands
4. **IDE support**: Better autocomplete and parameter validation

## Integration Examples

### With Tensor Creation
```tcl
# Create and floor in sequence
set data {1.7 2.3 -0.5 -1.8}
set input [torch::tensor_create -data $data -dtype float32]
set result [torch::floor -input $input]
```

### With Mathematical Operations
```tcl
# Mathematical pipeline
set input [torch::tensor_create -data {1.7 2.3 3.9} -dtype float32]
set floored [torch::floor -input $input]
set squared [torch::tensor_mul $floored $floored]
# Result: [1.0, 4.0, 9.0]
```

### With Different Tensor Types
```tcl
# Process different tensor shapes
set tensor_1d [torch::tensor_create -data {1.7 2.3} -dtype float32]
set tensor_2d [torch::tensor_create -data {1.7 2.3 3.9 4.1} -shape {2 2} -dtype float32]

set result_1d [torch::floor -input $tensor_1d]
set result_2d [torch::floor -input $tensor_2d]
```

## Best Practices

1. **Use modern syntax** for new code to ensure future compatibility
2. **Validate tensor names** before calling floor to avoid runtime errors  
3. **Consider data types** - ensure input tensors are floating-point for meaningful results
4. **Handle edge cases** - be aware of floor behavior with negative numbers
5. **Check tensor shapes** - verify the output tensor shape matches expectations

## Related Commands

- [`torch::ceil`](ceil.md) - Ceiling (round up) function
- [`torch::round`](round.md) - Round to nearest integer function  
- [`torch::trunc`](trunc.md) - Truncate toward zero function
- [`torch::frac`](frac.md) - Fractional part function
- [`torch::tensor_create`](tensor_create.md) - Create tensors
- [`torch::tensor_shape`](tensor_shape.md) - Get tensor shape

## Version History

- **Latest**: Added dual syntax support with named parameters
- **Legacy**: Original positional parameter syntax (still supported)

---

**Note**: Both legacy and modern syntaxes are fully supported. The legacy syntax is maintained for backward compatibility, while the modern syntax is recommended for new development. 