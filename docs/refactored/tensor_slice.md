# torch::tensor_slice / torch::tensorSlice

Extract a slice from a tensor along a specified dimension.

## Description

The `torch::tensor_slice` command extracts a slice from a tensor along a specified dimension. It supports both positional and named parameter syntax, with full backward compatibility.

**Alias**: `torch::tensorSlice` (camelCase)

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_slice tensor dim start ?end? ?step?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_slice -tensor tensor -dim dim -start start ?-end end? ?-step step?
torch::tensor_slice -input tensor -dimension dim -start start ?-end end? ?-step step?
```

### CamelCase Alias
```tcl
torch::tensorSlice tensor dim start ?end? ?step?
torch::tensorSlice -tensor tensor -dim dim -start start ?-end end? ?-step step?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-tensor` / `-input` | string | Yes | Tensor handle to slice |
| `dim` / `-dim` / `-dimension` | integer | Yes | Dimension along which to slice |
| `start` / `-start` | integer | Yes | Starting index for the slice |
| `end` / `-end` | integer | No | Ending index for the slice (default: end of dimension) |
| `step` / `-step` | integer | No | Step size for the slice (default: 1) |

## Return Value

Returns a tensor handle containing the sliced result.

## Examples

### Basic Usage

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_slice $tensor 0 1 4]
puts "Slice result: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_slice -tensor $tensor -dim 0 -start 1 -end 4]
puts "Slice result: $result"
```

**CamelCase alias:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensorSlice -input $tensor -dimension 0 -start 1 -end 4]
puts "Slice result: $result"
```

### With Step Parameter

**Positional syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_slice $tensor 0 0 8 2]  ;# Every other element
puts "Stepped slice: $result"
```

**Named parameter syntax:**
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8}]
set result [torch::tensor_slice -tensor $tensor -dim 0 -start 0 -end 8 -step 2]
puts "Stepped slice: $result"
```

### Multi-dimensional Tensors

```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9}}]
set result1 [torch::tensor_slice $tensor 0 1 2]  ;# Slice first dimension
set result2 [torch::tensor_slice $tensor 1 0 2]  ;# Slice second dimension
puts "Slice along dim 0: $result1"
puts "Slice along dim 1: $result2"
```

### Using Negative Indices

```tcl
set tensor [torch::tensor_create {1 2 3 4 5}]
set result [torch::tensor_slice $tensor 0 -3 -1]  ;# Last 3 elements
puts "Negative slice: $result"
```

### Different Data Types

```tcl
set tensor1 [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
set tensor2 [torch::tensor_create {1.5 2.5 3.5 4.5} float64]
set result1 [torch::tensor_slice $tensor1 0 1 3]
set result2 [torch::tensor_slice $tensor2 0 1 3]
puts "Float32 slice: $result1"
puts "Float64 slice: $result2"
```

## Error Handling

The command provides clear error messages for various error conditions:

```tcl
# Invalid tensor name
catch {torch::tensor_slice invalid_tensor 0 1} result
puts $result  ;# Output: Tensor not found

# Missing tensor parameter
catch {torch::tensor_slice} result
puts $result  ;# Output: Required tensor parameter missing

# Invalid dimension
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_slice $tensor invalid 1} result
puts $result  ;# Output: Invalid dimension value

# Invalid start
catch {torch::tensor_slice $tensor 0 invalid} result
puts $result  ;# Output: Invalid start value

# Invalid end
catch {torch::tensor_slice $tensor 0 1 invalid} result
puts $result  ;# Output: Invalid end value

# Invalid step
catch {torch::tensor_slice $tensor 0 1 2 invalid} result
puts $result  ;# Output: Invalid step value

# Unknown named parameter
catch {torch::tensor_slice -invalid $tensor} result
puts $result  ;# Output: Unknown parameter: -invalid

# Missing parameter value
catch {torch::tensor_slice -tensor $tensor -dim} result
puts $result  ;# Output: Missing value for parameter
```

## Migration Guide

### From Positional to Named Parameters

**Old code:**
```tcl
set result [torch::tensor_slice $tensor 0 1 4 2]
```

**New code (equivalent):**
```tcl
set result [torch::tensor_slice -tensor $tensor -dim 0 -start 1 -end 4 -step 2]
```

### Using CamelCase Alias

**Old code:**
```tcl
set result [torch::tensor_slice $tensor 0 1 4]
```

**New code (equivalent):**
```tcl
set result [torch::tensorSlice $tensor 0 1 4]
```

## Notes

- The `start` parameter is inclusive (includes the element at that index)
- The `end` parameter is exclusive (excludes the element at that index)
- If `end` is not specified, the slice extends to the end of the dimension
- The `step` parameter defaults to 1 if not specified
- Negative indices are supported and count from the end of the dimension
- The step parameter must be positive (PyTorch limitation)
- The command supports tensors of various data types (float32, float64, int32, etc.)
- The result tensor maintains the same data type as the input tensor
- Slicing preserves the tensor's memory layout when possible

## See Also

- `torch::tensor_reshape` - Reshape tensor dimensions
- `torch::tensor_permute` - Permute tensor dimensions
- `torch::tensor_cat` - Concatenate tensors
- `torch::tensor_stack` - Stack tensors
- `torch::tensor_split` - Split tensor into chunks 