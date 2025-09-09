# torch::atan

Computes the arctangent (inverse tangent) of input tensor elements.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::atan input_tensor
```

### Named Parameter Syntax
```tcl
torch::atan -input input_tensor
torch::atan -tensor input_tensor
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` | string | Input tensor handle | Yes |
| `-tensor` | string | Alias for `-input` | Yes (alternative) |

## Description

The `torch::atan` command computes the arctangent (inverse tangent) of each element in the input tensor. The function returns values in the range (-π/2, π/2).

**Mathematical Formula:**
```
output = atan(input)
```

**Key Properties:**
- **Domain:** All real numbers (-∞, ∞)
- **Range:** (-π/2, π/2) ≈ (-1.5708, 1.5708)
- **Special Values:**
  - atan(0) = 0
  - atan(1) = π/4 ≈ 0.7854
  - atan(-1) = -π/4 ≈ -0.7854
  - atan(∞) = π/2 ≈ 1.5708
  - atan(-∞) = -π/2 ≈ -1.5708

## Return Value

Returns a new tensor handle containing the arctangent of the input tensor elements.

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create a tensor with sample values
set input_tensor [torch::tensorCreate -data {0.0 1.0 -1.0} -dtype float32]

# Compute arctangent (positional syntax)
set result [torch::atan $input_tensor]

# Expected values: [0.0, 0.7854, -0.7854] (approximately)
puts "Shape: [torch::tensor_shape $result]"
```

#### Named Parameter Syntax
```tcl
# Create a tensor
set input_tensor [torch::zeros {3 3}]

# Compute arctangent using named parameters
set result1 [torch::atan -input $input_tensor]
set result2 [torch::atan -tensor $input_tensor]  # Alternative parameter name

puts "Shape: [torch::tensor_shape $result1]"
```

### Mathematical Examples

#### Computing Arctangent of Common Values
```tcl
# Test mathematical properties
set zero_tensor [torch::tensorCreate -data {0.0} -dtype float32]
set one_tensor [torch::tensorCreate -data {1.0} -dtype float32]

set atan_zero [torch::atan $zero_tensor]    # Should be 0
set atan_one [torch::atan $one_tensor]      # Should be π/4 ≈ 0.7854
```

#### Working with Different Tensor Shapes
```tcl
# 1D tensor
set tensor_1d [torch::tensorCreate -data {0.5 -0.5 2.0 -2.0} -dtype float32]
set result_1d [torch::atan -input $tensor_1d]

# 2D tensor
set tensor_2d [torch::ones {2 3}]
set result_2d [torch::atan -input $tensor_2d]

# 3D tensor
set tensor_3d [torch::zeros {2 2 2}]
set result_3d [torch::atan -input $tensor_3d]
```

## Error Handling

### Invalid Tensor Handle
```tcl
# This will generate an error
catch {torch::atan invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Parameters
```tcl
# Missing tensor argument
catch {torch::atan} error
puts $error  # Error message about missing parameter

# Missing value for named parameter
catch {torch::atan -input} error
puts $error  # "Missing value for parameter"
```

### Unknown Parameters
```tcl
set input_tensor [torch::zeros {2 2}]

# Invalid parameter name
catch {torch::atan -unknown_param $input_tensor} error
puts $error  # "Unknown parameter: -unknown_param"
```

## Comparison with Related Functions

| Function | Domain | Range | Description |
|----------|---------|-------|-------------|
| `torch::atan` | (-∞, ∞) | (-π/2, π/2) | Arctangent (inverse tangent) |
| `torch::atan2` | All (x,y) pairs | (-π, π] | Two-argument arctangent |
| `torch::tan` | All except odd multiples of π/2 | (-∞, ∞) | Tangent function |
| `torch::asin` | [-1, 1] | [-π/2, π/2] | Arcsine (inverse sine) |
| `torch::acos` | [-1, 1] | [0, π] | Arccosine (inverse cosine) |

## Migration Guide

If you're upgrading from positional to named parameter syntax:

```tcl
# Old syntax (still supported)
set result [torch::atan $input_tensor]

# New syntax (recommended)
set result [torch::atan -input $input_tensor]

# Both produce identical results
```

## Technical Notes

### Data Types
- Supports all floating-point data types
- Input tensor can have any shape
- Output tensor has the same shape and data type as input

### Performance
- Element-wise operation with no performance difference between syntaxes
- Optimized for both CPU and CUDA tensors
- Memory usage scales linearly with tensor size

### Numerical Considerations
- Results are computed using high-precision arithmetic
- Special handling for edge cases (very large positive/negative values)
- Maintains numerical stability across different input ranges

## See Also

- [`torch::atan2`](atan2.md) - Two-argument arctangent
- [`torch::tan`](tan.md) - Tangent function
- [`torch::asin`](asin.md) - Arcsine function
- [`torch::acos`](acos.md) - Arccosine function
- [`torch::sinh`](sinh.md) - Hyperbolic sine
- [`torch::atanh`](atanh.md) - Hyperbolic arctangent 