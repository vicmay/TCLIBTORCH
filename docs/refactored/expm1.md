# torch::expm1

**Compute exp(x) - 1 element-wise with improved numerical precision**

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::expm1 tensor
```

### Named Parameter Syntax (New)
```tcl
torch::expm1 -input tensor
```

### Both syntaxes are supported and produce identical results.

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `input` | tensor | Input tensor | Yes |

## Returns

Returns a new tensor with the same shape as the input tensor, where each element is the result of `exp(x) - 1` computed with improved numerical precision for small values.

## Description

The `torch::expm1` function calculates `exp(x) - 1` element-wise on the input tensor. Unlike computing `exp(x) - 1` directly, this function provides better numerical precision for small values of `x`, which is particularly important in scientific computing and machine learning applications.

### Mathematical Properties

- **expm1(0) = 0** (exactly)
- **expm1(1) ≈ e - 1 ≈ 1.718282**
- **expm1(ln(2)) = 1** (exactly)
- **For small x**: `expm1(x) ≈ x + x²/2 + x³/6 + ...` (Taylor series)
- **Numerical stability**: For very small `x`, `expm1(x)` is much more accurate than `exp(x) - 1`

## Examples

### Basic Usage
```tcl
# Create a tensor
set input [torch::tensor_create {0.0 1.0 -1.0}]

# Positional syntax
set result1 [torch::expm1 $input]

# Named parameter syntax  
set result2 [torch::expm1 -input $input]

# Both produce the same result
puts "Values: [torch::tensor_data $result1]"  ;# {0.0 1.718282 -0.632121}
```

### Mathematical Verification
```tcl
# Verify mathematical properties
set zero [torch::tensor_create {0.0}]
set result_zero [torch::expm1 -input $zero]
puts "expm1(0) = [torch::tensor_item $result_zero]"  ;# 0.0

set one [torch::tensor_create {1.0}]
set result_one [torch::expm1 $one]
puts "expm1(1) = [torch::tensor_item $result_one]"  ;# ≈ 1.718282

set ln2 [torch::tensor_create {0.693147}]
set result_ln2 [torch::expm1 -input $ln2]
puts "expm1(ln(2)) = [torch::tensor_item $result_ln2]"  ;# ≈ 1.0
```

### Numerical Precision Example
```tcl
# Demonstrate superior precision for small values
set small_x [torch::tensor_create {1e-8} float64]

# Using expm1 (accurate)
set result_expm1 [torch::expm1 -input $small_x]
set value_expm1 [torch::tensor_item $result_expm1]

# Manual calculation exp(x) - 1 (less accurate)
set exp_result [torch::tensor_exp $small_x]
set ones [torch::tensor_create {1.0} float64]
set manual_result [torch::tensor_sub $exp_result $ones]
set value_manual [torch::tensor_item $manual_result]

puts "expm1(1e-8) = $value_expm1"     ;# More accurate
puts "exp(1e-8)-1 = $value_manual"    ;# Less accurate for small values
```

### Multi-dimensional Tensors
```tcl
# 2D tensor example
set input_2d [torch::tensor_create {0.0 1.0 -0.5 0.693147}]
set reshaped [torch::tensor_reshape $input_2d {2 2}]
set result_2d [torch::expm1 -input $reshaped]

puts "Shape: [torch::tensor_shape $result_2d]"  ;# {2 2}
puts "Result: [torch::tensor_data $result_2d]"
```

### Scientific Computing Applications

#### 1. Compound Interest Calculation
```tcl
# For small interest rates, expm1 provides more accurate results
set interest_rate 0.001  ;# 0.1% interest rate
set rate_tensor [torch::tensor_create [list $interest_rate] float64]
set compound_factor [torch::expm1 $rate_tensor]
set factor_value [torch::tensor_item $compound_factor]

puts "Compound factor for 0.1% rate: $factor_value"
# More accurate than exp(rate) - 1 for small rates
```

#### 2. Probability and Statistics
```tcl
# In probability calculations involving small changes
set log_odds_change [torch::tensor_create {-0.01 0.01 -0.05 0.05}]
set probability_change [torch::expm1 -input $log_odds_change]

puts "Probability changes: [torch::tensor_data $probability_change]"
```

#### 3. Taylor Series Approximation
```tcl
# Verify Taylor series for small values
set x 0.01
set x_tensor [torch::tensor_create [list $x] float64]
set expm1_result [torch::expm1 $x_tensor]
set exact_value [torch::tensor_item $expm1_result]

# Taylor approximation: x + x²/2 + x³/6
set taylor_approx [expr {$x + ($x*$x)/2.0 + ($x*$x*$x)/6.0}]

puts "expm1($x) = $exact_value"
puts "Taylor approximation = $taylor_approx"
puts "Difference = [expr {abs($exact_value - $taylor_approx)}]"
```

## Data Type Support

The function supports all floating-point data types:

```tcl
# Float32
set f32_input [torch::tensor_create {1.0} float32]
set f32_result [torch::expm1 -input $f32_input]

# Float64 (double precision)
set f64_input [torch::tensor_create {1.0} float64]
set f64_result [torch::expm1 $f64_input]

# Integer tensors are automatically promoted to float
set int_input [torch::tensor_create {1}]
set int_result [torch::expm1 -input $int_input]
```

## Error Handling

```tcl
# Missing arguments
catch {torch::expm1} result
puts $result  ;# "Usage: torch::expm1 tensor OR torch::expm1 -input tensor"

# Invalid tensor name
catch {torch::expm1 invalid_tensor} result
puts $result  ;# "Invalid tensor name"

# Missing parameter value
catch {torch::expm1 -input} result
puts $result  ;# "Named parameter requires a value"

# Unknown parameter
set input [torch::tensor_create {1.0}]
catch {torch::expm1 -unknown $input} result
puts $result  ;# "Unknown parameter: -unknown"
```

## Mathematical Comparison

| Function | Formula | Best Use Case |
|----------|---------|---------------|
| `torch::expm1` | `exp(x) - 1` | Small values of x, numerical stability |
| `torch::tensor_exp` | `exp(x)` | General exponential calculations |
| `torch::tensor_log` | `log(x)` | General logarithmic calculations |
| `torch::exp2` | `2^x` | Base-2 exponential |
| `torch::exp10` | `10^x` | Base-10 exponential |

## Performance Notes

- **Optimized implementation**: Uses PyTorch's optimized `expm1` function
- **Numerical stability**: Superior precision for `|x| < 1e-5`
- **GPU support**: Fully supports CUDA tensors
- **Broadcasting**: Supports all PyTorch broadcasting rules

## Migration Guide

### From Old Syntax to New Syntax

```tcl
# Old positional syntax (still supported)
set result [torch::expm1 $input]

# New named parameter syntax (recommended)
set result [torch::expm1 -input $input]

# Both work identically - choose based on preference
```

### When to Use expm1 vs exp(x) - 1

```tcl
# Use expm1 for small values (better precision)
set small_values [torch::tensor_create {1e-8 1e-10 1e-12}]
set precise_result [torch::expm1 -input $small_values]

# Regular exp is fine for larger values
set large_values [torch::tensor_create {1.0 2.0 3.0}]
set exp_result [torch::tensor_exp $large_values]
set ones [torch::ones_like -input $exp_result]
set manual_result [torch::tensor_sub $exp_result $ones]
# For large values, both methods give similar precision
```

## See Also

- [torch::tensor_exp](exp.md) - General exponential function
- [torch::exp2](exp2.md) - Base-2 exponential function
- [torch::exp10](exp10.md) - Base-10 exponential function
- [torch::tensor_log](log.md) - Natural logarithm function
- [torch::log1p](log1p.md) - log(1 + x) with improved precision

## Technical Details

- **Implementation**: Uses PyTorch's `torch::expm1()` function
- **Precision**: IEEE 754 double precision (when using float64)
- **Domain**: All real numbers
- **Range**: (-1, ∞)
- **Numerical accuracy**: Superior to `exp(x) - 1` for `|x| < 0.1` 