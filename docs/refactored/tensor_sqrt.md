# torch::tensor_sqrt

Element-wise square root operation for tensors.

## Syntax

### Positional Syntax (Original)
```tcl
torch::tensor_sqrt tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_sqrt -input tensor
```

### CamelCase Alias
```tcl
torch::tensorSqrt tensor
torch::tensorSqrt -input tensor
```

## Parameters

### Required Parameters
- **input** (`string`): Handle to the input tensor

## Return Value

Returns a string handle to a new tensor containing the element-wise square root of the input tensor.

## Description

The `torch::tensor_sqrt` command computes the element-wise square root of a tensor. For each element `x` in the input tensor, the corresponding element in the output tensor will be `√x`.

**Mathematical Properties:**
- `√0 = 0`
- `√1 = 1`
- `√(x²) = x` for positive x
- For negative inputs, the result is NaN (Not a Number)
- Preserves tensor shape and data type

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with values [4.0, 9.0, 16.0, 25.0]
set t1 [torch::full {4} 4.0]
# ... set other values ...

# Compute square root
set result [torch::tensor_sqrt $t1]
# Result contains [2.0, 3.0, 4.0, 5.0]
```

### Named Parameter Syntax
```tcl
# Create a tensor
set input_tensor [torch::full {1} 9.0]

# Compute square root using named parameters
set sqrt_result [torch::tensor_sqrt -input $input_tensor]

# Result: 3.0
puts [torch::tensor_item $sqrt_result]
```

### CamelCase Syntax
```tcl
# Using camelCase alias with positional syntax
set t1 [torch::full {1} 25.0]
set result [torch::tensorSqrt $t1]

# Using camelCase alias with named parameters
set result2 [torch::tensorSqrt -input $t1]
```

### Mathematical Properties
```tcl
# Square root of perfect squares
set t1 [torch::full {1} 36.0]
set result [torch::tensor_sqrt $t1]
# Result: 6.0

# Square root of zero
set zero_tensor [torch::zeros {1}]
set sqrt_zero [torch::tensor_sqrt $zero_tensor]
# Result: 0.0

# Square root of one
set one_tensor [torch::ones {1}]
set sqrt_one [torch::tensor_sqrt $one_tensor]
# Result: 1.0
```

### Chain Operations
```tcl
# Verify that sqrt(x²) = x
set original [torch::full {1} 7.0]
set squared [torch::tensor_mul $original $original]
set sqrt_result [torch::tensor_sqrt $squared]
# sqrt_result ≈ 7.0 (original value)
```

### Different Data Types
```tcl
# Works with different precision types
set t1 [torch::full {1} 4.0 float64]
set result [torch::tensor_sqrt $t1]
# Result maintains float64 precision
```

### Multi-dimensional Tensors
```tcl
# Works with tensors of any shape
set matrix [torch::full {2 3} 16.0]
set sqrt_matrix [torch::tensor_sqrt $matrix]
# Result: 2x3 matrix filled with 4.0
```

## Error Handling

### Invalid Tensor Handle
```tcl
catch {torch::tensor_sqrt invalid_tensor} error
# Error: "Invalid tensor name"
```

### Missing Parameters
```tcl
catch {torch::tensor_sqrt} error
# Error: "Usage: torch::tensor_sqrt tensor" or "Required parameter missing: input"
```

### Unknown Parameters
```tcl
set t1 [torch::ones {1}]
catch {torch::tensor_sqrt -invalid $t1} error
# Error: "Unknown parameter: -invalid"
```

### Missing Parameter Values
```tcl
catch {torch::tensor_sqrt -input} error
# Error: "Missing value for parameter"
```

## Implementation Notes

### Backward Compatibility
The original positional syntax remains fully supported. Existing code using `torch::tensor_sqrt tensor` will continue to work without modification.

### Performance
All three syntax variants (positional, named, camelCase) produce identical results and have the same performance characteristics.

### Precision
The operation preserves the input tensor's data type and device. Results maintain full precision for floating-point calculations.

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old style
set result [torch::tensor_sqrt $my_tensor]

# New style (equivalent)
set result [torch::tensor_sqrt -input $my_tensor]
```

### Adopting CamelCase
```tcl
# Snake case
set result [torch::tensor_sqrt -input $tensor]

# CamelCase (equivalent)  
set result [torch::tensorSqrt -input $tensor]
```

## Mathematical Background

The square root function is defined as:
- `√x = y` where `y² = x` and `y ≥ 0`

For numerical computation:
- **Domain**: Real numbers ≥ 0
- **Range**: Real numbers ≥ 0
- **Monotonicity**: Strictly increasing for x > 0
- **Derivatives**: `d/dx √x = 1/(2√x)` for x > 0

### Special Values
- `√0 = 0`
- `√1 = 1`
- `√∞ = ∞`
- `√(-x) = NaN` for x > 0

## See Also

- [`torch::tensor_pow`](tensor_pow.md) - General power operation
- [`torch::tensor_square`](tensor_square.md) - Element-wise square
- [`torch::tensor_rsqrt`](tensor_rsqrt.md) - Reciprocal square root
- [`torch::tensor_exp`](tensor_exp.md) - Exponential function
- [`torch::tensor_log`](tensor_log.md) - Natural logarithm 