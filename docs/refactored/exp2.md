# torch::exp2

Computes the base-2 exponential of the input tensor element-wise.

## Syntax

### Positional Parameters (Original)
```tcl
torch::exp2 tensor
```

### Named Parameters (Refactored)
```tcl
torch::exp2 -input tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` | string | Yes | Name of the input tensor |

## Return Value

Returns a new tensor handle containing the base-2 exponential of each element in the input tensor.

## Mathematical Formula

For each element x in the input tensor:
```
exp2(x) = 2^x
```

**Domain**: (-∞, ∞)  
**Range**: (0, ∞)

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::tensor_create {0.0 1.0 2.0 3.0}]

# Positional syntax
set result1 [torch::exp2 $input]

# Named parameter syntax  
set result2 [torch::exp2 -input $input]

# Both produce identical results
```

### Mathematical Examples

```tcl
# exp2(0) = 2^0 = 1
set zero [torch::tensor_create {0.0}]
set result_zero [torch::exp2 -input $zero]
puts [torch::tensor_item $result_zero]  ;# Output: 1.0

# exp2(1) = 2^1 = 2
set one [torch::tensor_create {1.0}]
set result_one [torch::exp2 $one]
puts [torch::tensor_item $result_one]   ;# Output: 2.0

# exp2(2) = 2^2 = 4
set two [torch::tensor_create {2.0}]
set result_two [torch::exp2 -input $two]
puts [torch::tensor_item $result_two]   ;# Output: 4.0

# exp2(3) = 2^3 = 8
set three [torch::tensor_create {3.0}]
set result_three [torch::exp2 $three]
puts [torch::tensor_item $result_three] ;# Output: 8.0
```

### Negative Exponents

```tcl
# exp2(-1) = 2^(-1) = 0.5
set neg_one [torch::tensor_create {-1.0}]
set result_neg [torch::exp2 -input $neg_one]
puts [torch::tensor_item $result_neg]   ;# Output: 0.5

# exp2(-2) = 2^(-2) = 0.25
set neg_two [torch::tensor_create {-2.0}]
set result_neg2 [torch::exp2 $neg_two]
puts [torch::tensor_item $result_neg2]  ;# Output: 0.25
```

### Fractional Exponents

```tcl
# exp2(0.5) = 2^0.5 = sqrt(2) ≈ 1.414
set half [torch::tensor_create {0.5}]
set result_half [torch::exp2 -input $half]
puts [torch::tensor_item $result_half]  ;# Output: ~1.414

# exp2(1.5) = 2^1.5 = 2 * sqrt(2) ≈ 2.828
set one_half [torch::tensor_create {1.5}]
set result_one_half [torch::exp2 $one_half]
puts [torch::tensor_item $result_one_half]  ;# Output: ~2.828
```

### Multi-dimensional Tensors

```tcl
# 2D tensor with exp2 applied element-wise
set matrix_data {0.0 1.0 2.0 -1.0}
set input_2d [torch::tensor_create $matrix_data]
set reshaped [torch::tensor_reshape $input_2d {2 2}]

# Apply exp2 to all elements
set result_2d [torch::exp2 -input $reshaped]

# Results: [[1.0, 2.0], [4.0, 0.5]]
```

### Different Data Types

```tcl
# Float32 (default)
set float32_tensor [torch::tensor_create {2.0} float32]
set result_f32 [torch::exp2 -input $float32_tensor]

# Float64 for higher precision
set float64_tensor [torch::tensor_create {2.0} float64]
set result_f64 [torch::exp2 $float64_tensor]

# Integer input (will be converted to float)
set int_tensor [torch::tensor_create {3} int32]
set result_int [torch::exp2 -input $int_tensor]
```

## Relationship to Other Exponential Functions

```tcl
# Comparison with different exponential functions
set x [torch::tensor_create {2.0}]

set exp_result [torch::tensor_exp $x]    ;# e^x ≈ 7.389
set exp2_result [torch::exp2 $x]         ;# 2^x = 4.0
set exp10_result [torch::exp10 $x]       ;# 10^x = 100.0

# Relationship: exp2(x) = exp(x * ln(2))
# where ln(2) ≈ 0.693147
```

## Data Type Support

| Input Type | Output Type | Notes |
|------------|-------------|-------|
| `float32` | `float32` | Standard precision |
| `float64` | `float64` | Double precision |
| `int32` | `float32` | Integer converted to float |
| `int64` | `float32` | Long integer converted to float |

## Performance Notes

- `exp2` is optimized for base-2 calculations and may be faster than using `pow(2, x)`
- For very large positive values, result may overflow to infinity
- For very large negative values, result approaches zero
- GPU acceleration available when tensor is on CUDA device

## Error Handling

### Common Errors

```tcl
# Missing input tensor
torch::exp2
# Error: Usage: torch::exp2 tensor

# Invalid tensor name
torch::exp2 invalid_tensor
# Error: Invalid tensor name

# Missing parameter value in named syntax
torch::exp2 -input
# Error: Missing value for parameter

# Unknown parameter
torch::exp2 -unknown value
# Error: Unknown parameter: -unknown
```

## Mathematical Properties

1. **Base identity**: `exp2(0) = 1`
2. **Multiplication**: `exp2(x + y) = exp2(x) * exp2(y)`
3. **Power rule**: `exp2(n * x) = (exp2(x))^n`
4. **Inverse**: `log2(exp2(x)) = x`
5. **Monotonic**: exp2 is strictly increasing

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (positional syntax)
set result [torch::exp2 $tensor]

# NEW (named parameter syntax)
set result [torch::exp2 -input $tensor]
```

### Benefits of Named Parameters

1. **Self-documenting**: Clear what the parameter represents
2. **Consistent**: Matches pattern of other refactored commands
3. **Future-proof**: Easy to extend with additional parameters if needed

## See Also

- [torch::exp](exp.md) - Natural exponential function (e^x)
- [torch::exp10](exp10.md) - Base-10 exponential function (10^x)
- [torch::log2](log2.md) - Base-2 logarithm (inverse of exp2)
- [torch::pow](pow.md) - General power function (x^y) 