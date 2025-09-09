# torch::tensor_min

Find the minimum value(s) of a tensor, optionally along a dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_min tensor_handle ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_min -input tensor_handle ?-dim dim?
torch::tensor_min -tensor tensor_handle ?-dim dim?
```

### CamelCase Alias
```tcl
torch::tensorMin tensor_handle ?dim?
torch::tensorMin -input tensor_handle ?-dim dim?
torch::tensorMin -tensor tensor_handle ?-dim dim?
```

## Parameters

| Parameter         | Type   | Required | Description                                 |
|------------------|--------|----------|---------------------------------------------|
| `tensor_handle`  | string | Yes      | The handle of the tensor to find minimum     |
| `-dim`           | int    | No       | Dimension along which to find the minimum    |

## Return Value

Returns a tensor handle containing the minimum value(s).

- If no dimension is specified, returns the minimum of all elements.
- If a dimension is specified, returns the minimum along that dimension.

## Examples

### Basic Usage

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set min_val [torch::tensor_min $t]
puts "Minimum: [torch::tensor_to_list $min_val]"  ;# Output: Minimum: 1.0
```

### With Dimension

```tcl
set t [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
set min_val [torch::tensor_min $t 1]
puts "Minimum: [torch::tensor_to_list $min_val]"  ;# Output: Minimum: {1.0 4.0}
```

### Named Parameter Syntax

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set min_val [torch::tensor_min -input $t]
puts "Minimum: [torch::tensor_to_list $min_val]"

set t2 [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
set min_val2 [torch::tensor_min -input $t2 -dim 1]
puts "Minimum: [torch::tensor_to_list $min_val2]"
```

### CamelCase Alias

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set min_val [torch::tensorMin $t]
puts "Minimum: [torch::tensor_to_list $min_val]"
```

### Error Handling

```tcl
catch {torch::tensor_min invalid_tensor} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

catch {torch::tensor_min} result
puts "Error: $result"  ;# Output: Error: Required input parameter missing

set t [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
catch {torch::tensor_min -unknown $t} result
puts "Error: $result"  ;# Output: Error: Unknown parameter: -unknown
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set min_val [torch::tensor_min $tensor]
set min_val [torch::tensor_min $tensor 1]
```

**New (Named Parameters):**
```tcl
set min_val [torch::tensor_min -input $tensor]
set min_val [torch::tensor_min -input $tensor -dim 1]
```

**New (CamelCase):**
```tcl
set min_val [torch::tensorMin $tensor]
set min_val [torch::tensorMin -input $tensor -dim 1]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
set min_val [torch::tensor_min $tensor]
```

## Notes

- Both snake_case and camelCase versions are functionally identical
- Works with tensors of any data type or shape
- Returns the minimum value(s) as a tensor handle

## Related Commands

- `torch::tensor_max` - Find maximum values
- `torch::tensor_mean` - Compute mean
- `torch::tensor_median` - Compute median
- `torch::tensor_create` - Create tensors 