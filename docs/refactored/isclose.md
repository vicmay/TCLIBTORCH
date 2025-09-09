# torch::isclose

Element-wise comparison of two tensors with configurable tolerance for numerical precision.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::isclose input other ?rtol? ?atol? ?equal_nan?
torch::isClose input other ?rtol? ?atol? ?equal_nan?
```

### Named Parameter Syntax  
```tcl
torch::isclose -input TENSOR -other TENSOR ?-rtol DOUBLE? ?-atol DOUBLE? ?-equal_nan BOOL?
torch::isClose -input TENSOR -other TENSOR ?-rtol DOUBLE? ?-atol DOUBLE? ?-equal_nan BOOL?
```

## Description

The `torch::isclose` command performs element-wise comparison between two tensors, returning a boolean tensor indicating where the elements are "close" within specified tolerances. This is essential for numerical computations where exact equality is unreliable due to floating-point precision limitations.

The comparison uses both absolute and relative tolerances:
- **Absolute tolerance (atol)**: The maximum absolute difference allowed
- **Relative tolerance (rtol)**: The maximum relative difference allowed as a fraction of the larger absolute value

Two elements are considered close if:
```
|input - other| ≤ atol + rtol × max(|input|, |other|)
```

## Parameters

### Required Parameters
- `input` / `-input` / `-tensor1`: First input tensor for comparison
- `other` / `-other` / `-tensor2`: Second input tensor for comparison

### Optional Parameters
- `rtol` / `-rtol` / `-relativeTolerance`: Relative tolerance (default: 1e-05)
- `atol` / `-atol` / `-absoluteTolerance`: Absolute tolerance (default: 1e-08)  
- `equal_nan` / `-equal_nan` / `-equalNan`: Whether to consider NaN values as equal (default: false)

## Return Value

Returns a boolean tensor of the same shape as the input tensors, with `true` where elements are close and `false` otherwise.

## Examples

### Basic Usage - Identical Tensors
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set result [torch::isclose $tensor1 $tensor2]
;# Returns: {1 1 1} (all true)
```

### Small Differences Within Default Tolerance
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.00001 2.00001 3.00001} -dtype float32]
set result [torch::isclose $tensor1 $tensor2]
;# Returns: {1 1 1} (small differences are close)
```

### Large Differences Beyond Default Tolerance  
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.1 2.1 3.1} -dtype float32]
set result [torch::isclose $tensor1 $tensor2]
;# Returns: {0 0 0} (large differences are not close)
```

### Custom Relative Tolerance
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.05 2.05} -dtype float32]
set result [torch::isclose $tensor1 $tensor2 0.1]
;# Returns: {1 1} (5% difference is close with rtol=0.1)
```

### Custom Absolute and Relative Tolerances
```tcl
set tensor1 [torch::tensor_create -data {0.0 1.0} -dtype float32]
set tensor2 [torch::tensor_create -data {0.05 1.05} -dtype float32]
set result [torch::isclose $tensor1 $tensor2 0.1 0.1]
;# Returns: {1 1} (both close with rtol=0.1, atol=0.1)
```

### Named Parameter Syntax
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.01 2.01} -dtype float32]
set result [torch::isclose -input $tensor1 -other $tensor2 -rtol 0.05 -atol 0.01]
;# Returns: {1 1}
```

### CamelCase Parameter Names
```tcl
set tensor1 [torch::tensor_create -data {1.0 0.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.05 0.05} -dtype float32]
set result [torch::isclose -input $tensor1 -other $tensor2 -relativeTolerance 0.1 -absoluteTolerance 0.1]
;# Returns: {1 1}
```

### CamelCase Alias
```tcl
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set tensor2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set result [torch::isClose $tensor1 $tensor2]
;# Returns: {1 1 1}
```

## Technical Details

### Tolerance Formula
The comparison formula accounts for both absolute and relative differences:
```
|a - b| ≤ atol + rtol × max(|a|, |b|)
```

This means:
- For values near zero, absolute tolerance dominates
- For large values, relative tolerance dominates
- Both tolerances work together for intermediate values

### Default Tolerances
- **rtol = 1e-05**: Allows 0.001% relative difference
- **atol = 1e-08**: Allows very small absolute differences

### Data Type Compatibility
- Works with all numeric tensor types (int8, int16, int32, int64, float16, float32, float64)
- Automatic type promotion follows PyTorch rules
- Output is always boolean tensor

### Special Values
- **NaN**: By default, NaN != NaN (use `equal_nan=1` to change this)
- **Infinity**: +∞ == +∞ and -∞ == -∞, but +∞ != -∞
- **Zero**: Handled correctly with absolute tolerance

## Migration from Positional to Named Syntax

### Before (Positional)
```tcl
set result [torch::isclose $tensor1 $tensor2 0.1 0.01 1]
```

### After (Named Parameters)
```tcl
set result [torch::isclose -input $tensor1 -other $tensor2 -rtol 0.1 -atol 0.01 -equal_nan 1]
```

### Alternative Parameter Names
```tcl
# Various acceptable parameter names
torch::isclose -tensor1 $t1 -tensor2 $t2              ;# tensor1/tensor2
torch::isclose -input $t1 -other $t2                  ;# input/other
torch::isclose -input $t1 -other $t2 -rtol 0.1        ;# rtol
torch::isclose -input $t1 -other $t2 -relativeTolerance 0.1  ;# relativeTolerance
torch::isclose -input $t1 -other $t2 -atol 0.01       ;# atol
torch::isclose -input $t1 -other $t2 -absoluteTolerance 0.01 ;# absoluteTolerance
torch::isclose -input $t1 -other $t2 -equal_nan 1     ;# equal_nan
torch::isclose -input $t1 -other $t2 -equalNan 1      ;# equalNan
```

## Mathematical Context

This command is equivalent to:
- **NumPy**: `numpy.isclose(a, b, rtol, atol, equal_nan)`
- **PyTorch**: `torch.isclose(input, other, rtol, atol, equal_nan)`
- **Mathematical**: `|a - b| ≤ atol + rtol × max(|a|, |b|)`

## Common Use Cases

1. **Unit Testing**: Verify numerical computations within tolerance
2. **Convergence Checking**: Determine if iterative algorithms have converged
3. **Model Validation**: Compare model outputs with expected results
4. **Gradient Checking**: Verify automatic differentiation accuracy
5. **Tensor Equality**: Robust comparison of floating-point tensors

## Error Handling

The command will raise an error if:
- Required tensors are not provided
- Tensor handles are invalid
- Tolerance values are negative
- Unknown parameters are specified
- Tensors have incompatible shapes (broadcasting rules apply)

## See Also

- [`torch::tensor_eq`](tensor_eq.md) - Exact equality comparison
- [`torch::tensor_ne`](tensor_ne.md) - Inequality comparison  
- [`torch::tensor_allclose`](tensor_allclose.md) - All elements close test
- [`torch::tensor_sub`](tensor_sub.md) - Tensor subtraction for manual difference calculation
- [`torch::tensor_abs`](tensor_abs.md) - Absolute value for manual tolerance checking 