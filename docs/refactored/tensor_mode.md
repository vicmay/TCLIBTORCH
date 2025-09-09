# torch::tensor_mode

Find the mode (most frequent value) of a tensor, optionally along a dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_mode tensor_handle ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_mode -input tensor_handle ?-dim dim?
torch::tensor_mode -tensor tensor_handle ?-dim dim?
```

### CamelCase Alias
```tcl
torch::tensorMode tensor_handle ?dim?
torch::tensorMode -input tensor_handle ?-dim dim?
torch::tensorMode -tensor tensor_handle ?-dim dim?
```

## Parameters

| Parameter         | Type   | Required | Description                                 |
|------------------|--------|----------|---------------------------------------------|
| `tensor_handle`  | string | Yes      | The handle of the tensor to find mode       |
| `-dim`           | int    | No       | Dimension along which to find the mode      |

## Return Value

Returns a tensor handle containing the mode value(s).

- If no dimension is specified, returns the mode of all elements.
- If a dimension is specified, returns the mode along that dimension.
- In case of ties, returns the first occurrence.

## Examples

### Basic Usage

```tcl
set t [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
set mode_val [torch::tensor_mode $t]
puts "Mode: [torch::tensor_to_list $mode_val]"  ;# Output: Mode: 1.0
```

### With Dimension

```tcl
set t [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
set mode_val [torch::tensor_mode $t 1]
puts "Mode: [torch::tensor_to_list $mode_val]"  ;# Output: Mode: {1.0 3.0}
```

### Named Parameter Syntax

```tcl
set t [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
set mode_val [torch::tensor_mode -input $t]
puts "Mode: [torch::tensor_to_list $mode_val]"

set t2 [torch::tensor_create {{1.0 2.0 1.0} {3.0 1.0 3.0}} float32 cpu true]
set mode_val2 [torch::tensor_mode -input $t2 -dim 1]
puts "Mode: [torch::tensor_to_list $mode_val2]"
```

### CamelCase Alias

```tcl
set t [torch::tensor_create {1.0 2.0 1.0 3.0 1.0} float32 cpu true]
set mode_val [torch::tensorMode $t]
puts "Mode: [torch::tensor_to_list $mode_val]"
```

### Error Handling

```tcl
catch {torch::tensor_mode invalid_tensor} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

catch {torch::tensor_mode} result
puts "Error: $result"  ;# Output: Error: Required input parameter missing

set t [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
catch {torch::tensor_mode -unknown $t} result
puts "Error: $result"  ;# Output: Error: Unknown parameter: -unknown
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set mode_val [torch::tensor_mode $tensor]
set mode_val [torch::tensor_mode $tensor 1]
```

**New (Named Parameters):**
```tcl
set mode_val [torch::tensor_mode -input $tensor]
set mode_val [torch::tensor_mode -input $tensor -dim 1]
```

**New (CamelCase):**
```tcl
set mode_val [torch::tensorMode $tensor]
set mode_val [torch::tensorMode -input $tensor -dim 1]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
set mode_val [torch::tensor_mode $tensor]
```

## Notes

- Both snake_case and camelCase versions are functionally identical
- Works with tensors of any data type or shape
- Returns the most frequent value(s) as a tensor handle
- In case of ties, returns the first occurrence
- For single-element tensors, returns that element
- For empty tensors, behavior is undefined

## Related Commands

- `torch::tensor_min` - Find minimum values
- `torch::tensor_max` - Find maximum values
- `torch::tensor_median` - Compute median
- `torch::tensor_mean` - Compute mean
- `torch::tensor_create` - Create tensors 