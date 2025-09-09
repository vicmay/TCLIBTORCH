# torch::tensor_std / torch::tensorStd

Calculate the standard deviation of tensor elements.

## Description

The `torch::tensor_std` command computes the standard deviation of tensor elements. It supports both positional and named parameter syntax, with full backward compatibility.

**Alias**: `torch::tensorStd` (camelCase)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_std tensor ?dim? ?unbiased?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_std -input tensor ?-dim dimension? ?-unbiased boolean?
torch::tensor_std -tensor tensor ?-dimension dimension? ?-unbiased boolean?
```

### CamelCase Alias
```tcl
torch::tensorStd tensor ?dim? ?unbiased?
torch::tensorStd -input tensor ?-dim dimension? ?-unbiased boolean?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Tensor handle to compute standard deviation |
| `dim` / `-dim` / `-dimension` | integer | No | Dimension along which to compute std (default: all elements) |
| `unbiased` / `-unbiased` | boolean | No | Whether to use unbiased estimation (default: true) |

## Return Value

Returns a tensor handle containing the standard deviation result.

## Examples

### Basic Usage

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_std $tensor]
puts "Standard deviation: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_std -input $tensor]
puts "Standard deviation: $result"
```

**CamelCase alias:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensorStd -input $tensor]
puts "Standard deviation: $result"
```

### With Dimension Specification

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
set result [torch::tensor_std $tensor 0]  ;# Along first dimension
puts "Std along dim 0: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6}}]
set result [torch::tensor_std -tensor $tensor -dim 0]
puts "Std along dim 0: $result"
```

### With Unbiased Parameter

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_std $tensor 0 1]  ;# Unbiased estimation
puts "Unbiased std: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_std -input $tensor -unbiased 1]
puts "Unbiased std: $result"
```

### Multi-dimensional Tensors

```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
set result1 [torch::tensor_std $tensor 0]  ;# Along first dimension
set result2 [torch::tensor_std $tensor 1]  ;# Along second dimension
puts "Std along dim 0: $result1"
puts "Std along dim 1: $result2"
```

### Different Data Types

```tcl
set tensor1 [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
set tensor2 [torch::tensor_create {1.5 2.5 3.5 4.5} float64]
set result1 [torch::tensor_std $tensor1]
set result2 [torch::tensor_std $tensor2]
puts "Float32 std: $result1"
puts "Float64 std: $result2"
```

## Error Handling

The command provides clear error messages for various error conditions:

```tcl
# Invalid tensor name
catch {torch::tensor_std invalid_tensor} result
puts $result  ;# Output: Invalid tensor name

# Missing tensor parameter
catch {torch::tensor_std} result
puts $result  ;# Output: Required input parameter missing

# Invalid dimension
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_std $tensor invalid} result
puts $result  ;# Output: Invalid dimension value

# Invalid unbiased value
catch {torch::tensor_std $tensor 0 invalid} result
puts $result  ;# Output: Invalid unbiased value

# Unknown named parameter
catch {torch::tensor_std -invalid $tensor} result
puts $result  ;# Output: Unknown parameter: -invalid

# Missing parameter value
catch {torch::tensor_std -input $tensor -dim} result
puts $result  ;# Output: Missing value for parameter
```

## Migration Guide

### From Positional to Named Parameters

**Old code:**
```tcl
set result [torch::tensor_std $tensor 0 1]
```

**New code (equivalent):**
```tcl
set result [torch::tensor_std -input $tensor -dim 0 -unbiased 1]
```

### Using CamelCase Alias

**Old code:**
```tcl
set result [torch::tensor_std $tensor]
```

**New code (equivalent):**
```tcl
set result [torch::tensorStd $tensor]
```

## Notes

- The `unbiased` parameter defaults to `true` (unbiased estimation)
- When no dimension is specified, the standard deviation is computed over all elements
- The command supports tensors of various data types (float32, float64, etc.)
- Integer tensors may not be supported for standard deviation computation
- The result tensor maintains the same data type as the input tensor
- For single-element tensors, the standard deviation is 0
- For zero tensors, the standard deviation is 0

## See Also

- `torch::tensor_var` - Compute variance
- `torch::tensor_mean` - Compute mean
- `torch::tensor_sum` - Compute sum
- `torch::tensor_min` - Find minimum values
- `torch::tensor_max` - Find maximum values 