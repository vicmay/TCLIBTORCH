# torch::exp10

Computes the base-10 exponential of the input tensor element-wise.

## Syntax

### Positional Parameters (Original)
```tcl
torch::exp10 tensor
```

### Named Parameters (Refactored)
```tcl
torch::exp10 -input tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` | string | Yes | Name of the input tensor |

## Return Value

Returns a new tensor handle containing the base-10 exponential of each element in the input tensor.

## Mathematical Formula

For each element x in the input tensor:
```
exp10(x) = 10^x
```

**Domain**: (-∞, ∞)  
**Range**: (0, ∞)

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::tensor_create {0.0 1.0 2.0 3.0}]

# Positional syntax
set result1 [torch::exp10 $input]

# Named parameter syntax  
set result2 [torch::exp10 -input $input]

# Both produce identical results
```

### Mathematical Examples

```tcl
# exp10(0) = 10^0 = 1
set zero [torch::tensor_create {0.0}]
set result_zero [torch::exp10 -input $zero]
puts [torch::tensor_item $result_zero]  ;# Output: 1.0

# exp10(1) = 10^1 = 10
set one [torch::tensor_create {1.0}]
set result_one [torch::exp10 $one]
puts [torch::tensor_item $result_one]   ;# Output: 10.0

# exp10(2) = 10^2 = 100
set two [torch::tensor_create {2.0}]
set result_two [torch::exp10 -input $two]
puts [torch::tensor_item $result_two]   ;# Output: 100.0

# exp10(3) = 10^3 = 1000
set three [torch::tensor_create {3.0}]
set result_three [torch::exp10 $three]
puts [torch::tensor_item $result_three] ;# Output: 1000.0
```

### Negative Exponents

```tcl
# exp10(-1) = 10^(-1) = 0.1
set neg_one [torch::tensor_create {-1.0}]
set result_neg [torch::exp10 -input $neg_one]
puts [torch::tensor_item $result_neg]   ;# Output: 0.1

# exp10(-2) = 10^(-2) = 0.01
set neg_two [torch::tensor_create {-2.0}]
set result_neg2 [torch::exp10 $neg_two]
puts [torch::tensor_item $result_neg2]  ;# Output: 0.01

# exp10(-3) = 10^(-3) = 0.001
set neg_three [torch::tensor_create {-3.0}]
set result_neg3 [torch::exp10 -input $neg_three]
puts [torch::tensor_item $result_neg3] ;# Output: 0.001
```

### Fractional Exponents

```tcl
# exp10(0.5) = 10^0.5 = sqrt(10) ≈ 3.162
set half [torch::tensor_create {0.5}]
set result_half [torch::exp10 -input $half]
puts [torch::tensor_item $result_half]  ;# Output: ~3.162

# exp10(1.5) = 10^1.5 = 10 * sqrt(10) ≈ 31.623
set one_half [torch::tensor_create {1.5}]
set result_one_half [torch::exp10 $one_half]
puts [torch::tensor_item $result_one_half]  ;# Output: ~31.623

# exp10(0.3010) ≈ 2 (since log10(2) ≈ 0.3010)
set log2 [torch::tensor_create {0.30103}]
set result_log2 [torch::exp10 -input $log2]
puts [torch::tensor_item $result_log2]  ;# Output: ~2.0
```

### Multi-dimensional Tensors

```tcl
# 2D tensor with exp10 applied element-wise
set matrix_data {0.0 1.0 2.0 -1.0}
set input_2d [torch::tensor_create $matrix_data]
set reshaped [torch::tensor_reshape $input_2d {2 2}]

# Apply exp10 to all elements
set result_2d [torch::exp10 -input $reshaped]

# Results: [[1.0, 10.0], [100.0, 0.1]]
```

### Different Data Types

```tcl
# Float32 (default)
set float32_tensor [torch::tensor_create {2.0} float32]
set result_f32 [torch::exp10 -input $float32_tensor]

# Float64 for higher precision
set float64_tensor [torch::tensor_create {2.0} float64]
set result_f64 [torch::exp10 $float64_tensor]

# Integer input (will be converted to float)
set int_tensor [torch::tensor_create {3} int32]
set result_int [torch::exp10 -input $int_tensor]
```

## Relationship to Other Exponential Functions

```tcl
# Comparison with different exponential functions
set x [torch::tensor_create {2.0}]

set exp_result [torch::tensor_exp $x]    ;# e^x ≈ 7.389
set exp2_result [torch::exp2 $x]         ;# 2^x = 4.0
set exp10_result [torch::exp10 $x]       ;# 10^x = 100.0

# Relationship: exp10(x) = exp(x * ln(10))
# where ln(10) ≈ 2.302585
```

## Common Use Cases

### Scientific Notation
```tcl
# Converting from log scale (common in scientific computing)
set log_values [torch::tensor_create {1.0 2.0 3.0 4.0}]  ;# log10 values
set linear_values [torch::exp10 -input $log_values]
# Results: [10, 100, 1000, 10000] - powers of 10
```

### pH Scale Calculations
```tcl
# pH scale: pH = -log10([H+])
# To get [H+] from pH: [H+] = 10^(-pH)
set ph_values [torch::tensor_create {7.0 6.0 8.0}]
set neg_ph [torch::tensor_mul $ph_values [torch::tensor_create {-1.0}]]
set hydrogen_concentration [torch::exp10 -input $neg_ph]
# pH 7: [H+] = 10^-7, pH 6: [H+] = 10^-6, pH 8: [H+] = 10^-8
```

### Decibel Calculations
```tcl
# Power ratio in decibels: dB = 10 * log10(P1/P2)
# To convert from dB to power ratio: ratio = 10^(dB/10)
set db_values [torch::tensor_create {10.0 20.0 30.0}]  ;# dB values
set db_scaled [torch::tensor_div $db_values [torch::tensor_create {10.0}]]
set power_ratios [torch::exp10 -input $db_scaled]
# 10 dB = 10x power, 20 dB = 100x power, 30 dB = 1000x power
```

## Data Type Support

| Input Type | Output Type | Notes |
|------------|-------------|-------|
| `float32` | `float32` | Standard precision |
| `float64` | `float64` | Double precision |
| `int32` | `float32` | Integer converted to float |
| `int64` | `float32` | Long integer converted to float |

## Performance Notes

- `exp10` uses `torch::pow(10.0, tensor)` internally for maximum accuracy
- For very large positive values, result may overflow to infinity
- For very large negative values, result approaches zero
- GPU acceleration available when tensor is on CUDA device
- More precise than computing `exp(x * ln(10))` for base-10 calculations

## Error Handling

### Common Errors

```tcl
# Missing input tensor
torch::exp10
# Error: Required parameter -input missing

# Invalid tensor name
torch::exp10 invalid_tensor
# Error: Invalid tensor name

# Missing parameter value in named syntax
torch::exp10 -input
# Error: Missing value for parameter

# Unknown parameter
torch::exp10 -unknown value
# Error: Unknown parameter: -unknown
```

## Mathematical Properties

1. **Base identity**: `exp10(0) = 1`
2. **Multiplication**: `exp10(x + y) = exp10(x) * exp10(y)`
3. **Power rule**: `exp10(n * x) = (exp10(x))^n`
4. **Inverse**: `log10(exp10(x)) = x`
5. **Monotonic**: exp10 is strictly increasing
6. **Base conversion**: `exp10(x) = exp(x * ln(10))` where `ln(10) ≈ 2.302585`

## Common Constants

```tcl
# Some useful exp10 values
set values [torch::tensor_create {0.0 0.30103 0.47712 0.69897 1.0}]
set results [torch::exp10 -input $values]
# exp10(0) = 1, exp10(0.30103) ≈ 2, exp10(0.47712) ≈ 3, 
# exp10(0.69897) ≈ 5, exp10(1) = 10
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (positional syntax)
set result [torch::exp10 $tensor]

# NEW (named parameter syntax)
set result [torch::exp10 -input $tensor]
```

### Benefits of Named Parameters

1. **Self-documenting**: Clear what the parameter represents
2. **Consistent**: Matches pattern of other refactored commands
3. **Scientific clarity**: Makes the base-10 nature explicit
4. **Future-proof**: Easy to extend with additional parameters if needed

## See Also

- [torch::exp](exp.md) - Natural exponential function (e^x)
- [torch::exp2](exp2.md) - Base-2 exponential function (2^x)
- [torch::log10](log10.md) - Base-10 logarithm (inverse of exp10)
- [torch::pow](pow.md) - General power function (x^y)
- [torch::log](log.md) - Natural logarithm function 