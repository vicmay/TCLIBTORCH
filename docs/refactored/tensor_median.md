# torch::tensor_median

Find the median value(s) of a tensor, optionally along a dimension.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_median tensor_handle ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_median -input tensor_handle ?-dim dim?
torch::tensor_median -tensor tensor_handle ?-dim dim?
torch::tensor_median -input tensor_handle ?-dimension dim?
```

### CamelCase Alias
```tcl
torch::tensorMedian tensor_handle ?dim?
torch::tensorMedian -input tensor_handle ?-dim dim?
torch::tensorMedian -tensor tensor_handle ?-dim dim?
```

## Parameters

| Parameter         | Type   | Required | Description                                 |
|------------------|--------|----------|---------------------------------------------|
| `tensor_handle`  | string | Yes      | The handle of the tensor to compute median   |
| `-dim`           | int    | No       | Dimension along which to compute the median  |
| `-dimension`     | int    | No       | Alias for `-dim`                            |

## Return Value

Returns a tensor handle containing the median value(s).

- If no dimension is specified, returns the median of all elements.
- If a dimension is specified, returns the median along that dimension.

## Examples

### Basic Usage

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set median [torch::tensor_median $t]
puts "Median: [torch::tensor_to_list $median]"  ;# Output: Median: 3.0
```

### With Dimension

```tcl
set t [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
set median [torch::tensor_median $t 1]
puts "Median: [torch::tensor_to_list $median]"  ;# Output: Median: {2.0 5.0}
```

### Named Parameter Syntax

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set median [torch::tensor_median -input $t]
puts "Median: [torch::tensor_to_list $median]"

set t2 [torch::tensor_create {{1.0 3.0 2.0} {5.0 4.0 6.0}} float32 cpu true]
set median2 [torch::tensor_median -input $t2 -dim 1]
puts "Median: [torch::tensor_to_list $median2]"
```

### CamelCase Alias

```tcl
set t [torch::tensor_create {1.0 3.0 2.0 5.0 4.0} float32 cpu true]
set median [torch::tensorMedian $t]
puts "Median: [torch::tensor_to_list $median]"
```

### Error Handling

```tcl
catch {torch::tensor_median invalid_tensor} result
puts "Error: $result"  ;# Output: Error: Invalid tensor name

catch {torch::tensor_median} result
puts "Error: $result"  ;# Output: Error: Required input parameter missing

set t [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
catch {torch::tensor_median -unknown $t} result
puts "Error: $result"  ;# Output: Error: Unknown parameter: -unknown
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set median [torch::tensor_median $tensor]
set median [torch::tensor_median $tensor 1]
```

**New (Named Parameters):**
```tcl
set median [torch::tensor_median -input $tensor]
set median [torch::tensor_median -input $tensor -dim 1]
```

**New (CamelCase):**
```tcl
set median [torch::tensorMedian $tensor]
set median [torch::tensorMedian -input $tensor -dim 1]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
set median [torch::tensor_median $tensor]
```

## Notes

- For even number of elements, the median is the lower of the two middle values (PyTorch behavior)
- Both snake_case and camelCase versions are functionally identical
- Works with tensors of any data type or shape

## Related Commands

- `torch::tensor_mean` - Compute mean
- `torch::tensor_mode` - Compute mode
- `torch::tensor_quantile` - Compute quantiles
- `torch::tensor_create` - Create tensors 