# torch::isfinite

Returns a tensor indicating which elements are finite (not infinite or NaN).

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::isfinite -input TENSOR
torch::isFinite -input TENSOR  ;# camelCase alias
```

### Legacy Syntax (Backward Compatible)
```tcl
torch::isfinite TENSOR
torch::isFinite TENSOR  ;# camelCase alias
```

## Parameters

### Named Parameters (Modern Syntax)
- **`-input TENSOR`** *(required)*: Input tensor to check for finite values
- **`-tensor TENSOR`** *(alias)*: Alternative name for `-input` parameter

### Positional Parameters (Legacy Syntax)
1. **`TENSOR`** *(required)*: Input tensor to check for finite values

## Return Value

Returns a new tensor of the same shape as the input, containing boolean values:
- **`1` (true)**: Element is finite (not infinite, not NaN)
- **`0` (false)**: Element is infinite (`inf`, `-inf`) or NaN

## Examples

### Basic Usage

```tcl
# Create a tensor with finite values
set t1 [torch::tensorCreate -data {1.0 2.5 -3.14} -dtype float32]

# Modern syntax (recommended)
set result1 [torch::isfinite -input $t1]
set result2 [torch::isFinite -input $t1]  ;# camelCase

# Legacy syntax (backward compatible)
set result3 [torch::isfinite $t1]
set result4 [torch::isFinite $t1]  ;# camelCase
```

### Working with Different Value Types

```tcl
# Finite values (all return true)
set finite_vals [torch::tensorCreate -data {1.0 0.0 -42.5 1e-10 1e10} -dtype float32]
set finite_check [torch::isfinite -input $finite_vals]
;# Result: all elements are 1 (true)

# Zero values (considered finite)
set zeros [torch::tensorCreate -data {0.0 -0.0} -dtype float32]
set zero_check [torch::isfinite -input $zeros]
;# Result: both elements are 1 (true)

# Integer values (always finite)
set integers [torch::tensorCreate -data {1 -5 0 999} -dtype int32]
set int_check [torch::isfinite -input $integers]
;# Result: all elements are 1 (true)
```

### Multi-dimensional Tensors

```tcl
# 2D tensor
set matrix [torch::tensorCreate -data {{1.0 2.0} {3.0 4.0}} -dtype float32]
set matrix_finite [torch::isfinite -input $matrix]
;# Result: 2x2 tensor with all elements as 1 (true)

# Preserves tensor shape
set shape [torch::tensor_shape $matrix_finite]
;# Returns: {2 2}
```

### Error Handling

```tcl
# Missing arguments
catch {torch::isfinite} result
;# Error: Usage information

# Invalid tensor name
catch {torch::isfinite -input "invalid_tensor"} result
;# Error: Invalid tensor name

# Unknown parameter
catch {torch::isfinite -unknown $tensor} result
;# Error: Unknown parameter
```

## Mathematical Notes

### What is Finite?
- **Finite**: Normal real numbers including zero, positive/negative numbers
- **Not Finite**: 
  - Positive infinity (`+inf`)
  - Negative infinity (`-inf`) 
  - Not a Number (`NaN`)

### Special Cases
- **Zero values**: Both `+0.0` and `-0.0` are considered finite
- **Very large numbers**: `1e30` or `1e-30` are finite (unless they overflow to infinity)
- **Integer types**: All integer values are always finite by definition

## Data Type Support

| Data Type | Supported | Notes |
|-----------|-----------|-------|
| `float32` | ✅ | Can have inf/NaN values |
| `float64` | ✅ | Can have inf/NaN values |
| `int32`   | ✅ | Always finite |
| `int64`   | ✅ | Always finite |
| `bool`    | ✅ | Always finite |

## Performance Notes

- **Zero-copy operation**: Returns a new tensor without copying input data
- **Shape preservation**: Output tensor has identical shape to input
- **Memory efficient**: Boolean result tensor uses minimal memory

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# OLD (Legacy - still works)
set result [torch::isfinite $my_tensor]

# NEW (Modern - recommended)  
set result [torch::isfinite -input $my_tensor]

# With camelCase (modern style)
set result [torch::isFinite -input $my_tensor]
```

### Benefits of Modern Syntax
- **Explicit parameters**: Clear intent with named parameters
- **Self-documenting**: Code is easier to read and understand
- **IDE support**: Better autocomplete and parameter hints
- **Future-proof**: Consistent with other PyTorch operations

## Related Commands

- **`torch::isnan`**: Check for NaN (Not a Number) values  
- **`torch::isinf`**: Check for infinite values
- **`torch::isclose`**: Compare tensors with tolerance for floating-point precision

## See Also

- [PyTorch torch.isfinite documentation](https://pytorch.org/docs/stable/generated/torch.isfinite.html)
- [LibTorch TCL Extension - Mathematical Operations](../mathematical_operations.md)
- [LibTorch TCL Extension - API Modernization Guide](../modernization_guide.md) 