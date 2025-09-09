# torch::isinf

Returns a tensor indicating which elements are infinite (positive or negative infinity).

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::isinf -input TENSOR
torch::isInf -input TENSOR  ;# camelCase alias
```

### Legacy Syntax (Backward Compatible)
```tcl
torch::isinf TENSOR
torch::isInf TENSOR  ;# camelCase alias
```

## Parameters

### Named Parameters (Modern Syntax)
- **`-input TENSOR`** *(required)*: Input tensor to check for infinite values
- **`-tensor TENSOR`** *(alias)*: Alternative name for `-input` parameter

### Positional Parameters (Legacy Syntax)
1. **`TENSOR`** *(required)*: Input tensor to check for infinite values

## Return Value

Returns a new tensor of the same shape as the input, containing boolean values:
- **`1` (true)**: Element is infinite (`+inf` or `-inf`)
- **`0` (false)**: Element is finite (including zero) or NaN

## Examples

### Basic Usage

```tcl
# Create a tensor with finite values
set t1 [torch::tensorCreate -data {1.0 2.5 -3.14} -dtype float32]

# Modern syntax (recommended)
set result1 [torch::isinf -input $t1]
set result2 [torch::isInf -input $t1]  ;# camelCase

# Legacy syntax (backward compatible)
set result3 [torch::isinf $t1]
set result4 [torch::isInf $t1]  ;# camelCase
```

### Working with Different Value Types

```tcl
# Finite values (all return false)
set finite_vals [torch::tensorCreate -data {1.0 0.0 -42.5 1e-10 1e10} -dtype float32]
set finite_check [torch::isinf -input $finite_vals]
;# Result: all elements are 0 (false) - none are infinite

# Zero values (not infinite)
set zeros [torch::tensorCreate -data {0.0 -0.0} -dtype float32]
set zero_check [torch::isinf -input $zeros]
;# Result: both elements are 0 (false) - zero is finite

# Integer values (never infinite)
set integers [torch::tensorCreate -data {1 -5 0 999} -dtype int32]
set int_check [torch::isinf -input $integers]
;# Result: all elements are 0 (false) - integers are always finite

# Very large finite values (still not infinite)
set large_vals [torch::tensorCreate -data {1e30 -1e30 1e-30} -dtype float32]
set large_check [torch::isinf -input $large_vals]
;# Result: all elements are 0 (false) - large but still finite
```

### Multi-dimensional Tensors

```tcl
# 2D tensor
set matrix [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32]
set matrix_inf [torch::isinf -input $matrix]
;# Result: 2x2 tensor with all elements as 0 (false) - all finite

# Preserves tensor shape
set shape [torch::tensor_shape $matrix_inf]
;# Returns: {2 2}
```

### Error Handling

```tcl
# Missing arguments
catch {torch::isinf} result
;# Error: Usage information

# Invalid tensor name
catch {torch::isinf -input "invalid_tensor"} result
;# Error: Invalid tensor name

# Unknown parameter
catch {torch::isinf -unknown $tensor} result
;# Error: Unknown parameter
```

## Mathematical Notes

### What is Infinite?
- **Infinite**: 
  - Positive infinity (`+inf`): Result of operations like `1.0/0.0`
  - Negative infinity (`-inf`): Result of operations like `-1.0/0.0`
- **Not Infinite**: 
  - Normal finite numbers: `1.0`, `-42.5`, `0.0`
  - Very large finite numbers: `1e30`, `-1e30`
  - Not a Number (`NaN`): Special case, not considered infinite

### Special Cases
- **Zero values**: Both `+0.0` and `-0.0` are finite, not infinite
- **Very large numbers**: `1e30` or `1e-30` are finite unless they overflow to infinity
- **Integer types**: All integer values are always finite, never infinite
- **NaN values**: `NaN` is neither finite nor infinite

### Relationship to Other Checks
```tcl
# For any tensor element x:
# - If isinf(x) is true, then isfinite(x) is false
# - If isfinite(x) is true, then isinf(x) is false  
# - NaN is neither finite nor infinite
```

## Data Type Support

| Data Type | Supported | Notes |
|-----------|-----------|-------|
| `float32` | ✅ | Can have inf values |
| `float64` | ✅ | Can have inf values |
| `int32`   | ✅ | Always returns false (integers never infinite) |
| `int64`   | ✅ | Always returns false (integers never infinite) |
| `bool`    | ✅ | Always returns false (booleans never infinite) |

## Performance Notes

- **Zero-copy operation**: Returns a new tensor without copying input data
- **Shape preservation**: Output tensor has identical shape to input
- **Memory efficient**: Boolean result tensor uses minimal memory
- **Element-wise operation**: Each element checked independently

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy - still works)
set result [torch::isinf $my_tensor]

# NEW (Modern - recommended)  
set result [torch::isinf -input $my_tensor]

# With camelCase (modern style)
set result [torch::isInf -input $my_tensor]
```

### Benefits of Modern Syntax
- **Explicit parameters**: Clear intent with named parameters
- **Self-documenting**: Code is easier to read and understand
- **IDE support**: Better autocomplete and parameter hints
- **Future-proof**: Consistent with other PyTorch operations

## Common Use Cases

### Numerical Stability Checking
```tcl
# Check for overflow in calculations
set result [torch::tensor_div $numerator $denominator]
set has_inf [torch::isinf -input $result]

# Count infinite values
set inf_count [torch::tensor_sum $has_inf]
```

### Data Validation
```tcl
# Validate input data doesn't contain infinities
proc validate_finite {tensor_name} {
    set inf_check [torch::isinf -input $tensor_name]
    set has_any_inf [torch::tensor_any $inf_check]
    if {[torch::tensor_item $has_any_inf]} {
        error "Input contains infinite values"
    }
}
```

### Debugging Numerical Issues
```tcl
# Find where infinities occur in calculations
set weights [torch::randn -shape {100 50}]
set result [torch::some_complex_operation $weights]
set inf_mask [torch::isinf -input $result]

# Print information about infinite locations
set inf_indices [torch::nonzero $inf_mask]
puts "Found infinities at indices: [torch::tensor_data $inf_indices]"
```

## Related Commands

- **`torch::isfinite`**: Check for finite values (opposite of infinite or NaN)
- **`torch::isnan`**: Check for NaN (Not a Number) values  
- **`torch::isclose`**: Compare tensors with tolerance for floating-point precision

## See Also

- [PyTorch torch.isinf documentation](https://pytorch.org/docs/stable/generated/torch.isinf.html)
- [LibTorch TCL Extension - Mathematical Operations](../mathematical_operations.md)
- [LibTorch TCL Extension - API Modernization Guide](../modernization_guide.md) 