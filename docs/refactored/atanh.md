# torch::atanh

Computes the inverse hyperbolic tangent (arctanh) of the input tensor element-wise.

## Syntax

### Positional Parameters (Original)
```tcl
torch::atanh tensor
```

### Named Parameters (Refactored)
```tcl
torch::atanh -input tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` | string | Yes | Name of the input tensor |

## Return Value

Returns a new tensor handle containing the inverse hyperbolic tangent of each element in the input tensor.

## Mathematical Formula

For each element x in the input tensor:
```
atanh(x) = 0.5 * ln((1 + x) / (1 - x))
```

**Domain**: (-1, 1)  
**Range**: (-∞, ∞)

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::tensor_create {0.0 0.5 -0.5}]

# Using positional syntax
set result1 [torch::atanh $input]

# Using named parameter syntax  
set result2 [torch::atanh -input $input]

# Both produce the same result
```

### Mathematical Properties

```tcl
# atanh(0) = 0
set zero [torch::tensor_create {0.0}]
set result [torch::atanh -input $zero]
puts [torch::tensor_item $result]  ;# Output: 0.0

# Symmetric property: atanh(-x) = -atanh(x)
set x [torch::tensor_create {0.5}]
set neg_x [torch::tensor_create {-0.5}]

set atanh_x [torch::atanh -input $x]
set atanh_neg_x [torch::atanh -input $neg_x]

puts [torch::tensor_item $atanh_x]     ;# Output: ~0.5493
puts [torch::tensor_item $atanh_neg_x] ;# Output: ~-0.5493
```

### Multidimensional Tensors

```tcl
# 2D tensor
set input [torch::tensor_create {{0.0 0.3} {-0.3 0.8}}]
set result [torch::atanh -input $input]

# The operation is applied element-wise
```

### Different Data Types

```tcl
# Float32
set input_f32 [torch::tensor_create {0.5} float32]
set result_f32 [torch::atanh -input $input_f32]

# Float64 
set input_f64 [torch::tensor_create {0.5} float64]
set result_f64 [torch::atanh -input $input_f64]
```

## Special Values

| Input | Output |
|-------|--------|
| 0.0 | 0.0 |
| 1.0 | +∞ |
| -1.0 | -∞ |
| > 1.0 | NaN |
| < -1.0 | NaN |

## Error Handling

### Common Errors

```tcl
# Invalid tensor name
torch::atanh -input invalid_tensor
# Error: Invalid tensor name

# Missing required parameter
torch::atanh
# Error: Usage: torch::atanh tensor (positional) or -input required (named)

# Unknown parameter
torch::atanh -wrong_param $tensor
# Error: Unknown parameter: -wrong_param

# Missing parameter value
torch::atanh -input
# Error: Missing value for parameter
```

## Implementation Notes

- The function supports both positional and named parameter syntax for backward compatibility
- All PyTorch tensor data types are supported (float32, float64, etc.)
- The operation is performed element-wise across the entire tensor
- Input values outside the domain (-1, 1) will produce NaN values

## Mathematical Background

The inverse hyperbolic tangent function is the inverse of the hyperbolic tangent function. It's commonly used in:

- Statistical analysis (Fisher z-transformation)
- Machine learning (certain activation functions)
- Mathematical modeling involving hyperbolic functions

## Performance Considerations

- The operation is GPU-accelerated when input tensors are on CUDA devices
- Large tensors benefit from vectorized computation
- Memory usage is proportional to input tensor size (creates new output tensor)

## Related Functions

- `torch::tanh` - Hyperbolic tangent (inverse operation)
- `torch::atan` - Inverse tangent (circular version)
- `torch::asinh` - Inverse hyperbolic sine
- `torch::acosh` - Inverse hyperbolic cosine

## Migration Guide

### From Old Syntax to New Syntax

```tcl
# Old (still supported)
set result [torch::atanh $input]

# New (recommended)
set result [torch::atanh -input $input]
```

The new named parameter syntax provides better code readability and makes the function signature self-documenting.

## Version Information

- **Added**: LibTorch TCL Extension 1.0
- **Dual Syntax Support**: Added in refactoring initiative
- **Status**: ✅ Complete (dual syntax, tests, documentation)

---

**Note**: Both `torch::atanh` (positional) and `torch::atanh -input` (named) syntaxes are equivalent and can be used interchangeably. 