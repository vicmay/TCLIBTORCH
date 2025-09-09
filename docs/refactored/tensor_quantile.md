# torch::tensor_quantile

Computes the q-th quantile of the elements in a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_quantile tensor q ?dim?
```

### Named Parameter Syntax
```tcl
torch::tensor_quantile -input tensor -q q ?-dim dim?
torch::tensor_quantile -tensor tensor -quantile q ?-dimension dim?
```

### CamelCase Alias
```tcl
torch::tensorQuantile tensor q ?dim?
torch::tensorQuantile -input tensor -q q ?-dim dim?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Tensor handle name |
| `q` / `-q` / `-quantile` | double | Yes | Quantile value between 0.0 and 1.0 |
| `dim` / `-dim` / `-dimension` | int | No | Dimension along which to compute quantile |

## Description

The `torch::tensor_quantile` command computes the q-th quantile of the elements in a tensor. The quantile value `q` must be between 0.0 and 1.0, where:
- `q = 0.0` returns the minimum value
- `q = 0.5` returns the median value  
- `q = 1.0` returns the maximum value

If no dimension is specified, the quantile is computed over all elements of the tensor. If a dimension is specified, the quantile is computed along that dimension.

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]

# Compute median (50th percentile)
set median [torch::tensor_quantile $tensor 0.5]
puts [torch::tensor_to_list $median]  ;# Output: 3.0

# Compute 25th percentile
set q25 [torch::tensor_quantile $tensor 0.25]
puts [torch::tensor_to_list $q25]    ;# Output: 2.0

# Compute 75th percentile
set q75 [torch::tensor_quantile $tensor 0.75]
puts [torch::tensor_to_list $q75]    ;# Output: 4.0
```

### Named Parameter Syntax

```tcl
# Using named parameters
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]

# Compute median using named parameters
set median [torch::tensor_quantile -input $tensor -q 0.5]
puts [torch::tensor_to_list $median]  ;# Output: 3.0

# Using alternative parameter names
set q25 [torch::tensor_quantile -tensor $tensor -quantile 0.25]
puts [torch::tensor_to_list $q25]     ;# Output: 2.0
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1.0 2.0 3.0} {4.0 5.0 6.0}} float32 cpu true]

# Compute quantile along dimension 0 (rows)
set result [torch::tensor_quantile $tensor 0.5 0]
puts [torch::tensor_to_list $result]  ;# Output: {2.5 3.5 4.5}

# Compute quantile along dimension 1 (columns)
set result [torch::tensor_quantile $tensor 0.5 1]
puts [torch::tensor_to_list $result]  ;# Output: {2.0 5.0}
```

### CamelCase Alias

```tcl
# Using camelCase alias
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]

# Compute median using camelCase
set median [torch::tensorQuantile $tensor 0.5]
puts [torch::tensor_to_list $median]  ;# Output: 3.0

# Using camelCase with named parameters
set q25 [torch::tensorQuantile -input $tensor -q 0.25]
puts [torch::tensor_to_list $q25]     ;# Output: 2.0
```

### Edge Cases

```tcl
# Single element tensor
set tensor [torch::tensor_create {42.0} float32 cpu true]
set result [torch::tensor_quantile $tensor 0.5]
puts [torch::tensor_to_list $result]  ;# Output: 42.0

# Negative values
set tensor [torch::tensor_create -data {-5.0 -3.0 -1.0 1.0 3.0 5.0} -dtype float32 -device cpu -requiresGrad true]
set result [torch::tensor_quantile $tensor 0.5]
puts [torch::tensor_to_list $result]  ;# Output: 0.0

# Extreme quantiles
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0 5.0} float32 cpu true]

# Minimum value (0th percentile)
set min_val [torch::tensor_quantile $tensor 0.0]
puts [torch::tensor_to_list $min_val] ;# Output: 1.0

# Maximum value (100th percentile)
set max_val [torch::tensor_quantile $tensor 1.0]
puts [torch::tensor_to_list $max_val] ;# Output: 5.0
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_quantile invalid_tensor 0.5} result
puts $result  ;# Output: Invalid tensor name
```

### Invalid Quantile Value
```tcl
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# Negative quantile
catch {torch::tensor_quantile $tensor -0.1} result
puts $result  ;# Output: Required parameters missing or invalid

# Quantile > 1.0
catch {torch::tensor_quantile $tensor 1.5} result
puts $result  ;# Output: Required parameters missing or invalid
```

### Invalid Dimension
```tcl
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# Dimension out of range
catch {torch::tensor_quantile $tensor 0.5 10} result
puts $result  ;# Output: Dimension out of range (expected to be in range of [-1, 0], but got 10)
```

### Missing Required Parameters
```tcl
# No arguments
catch {torch::tensor_quantile} result
puts $result  ;# Output: Required parameters missing or invalid

# Missing quantile value
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
catch {torch::tensor_quantile $tensor} result
puts $result  ;# Output: Required parameters missing or invalid
```

### Unknown Named Parameter
```tcl
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

catch {torch::tensor_quantile -input $tensor -q 0.5 -unknown param} result
puts $result  ;# Output: Unknown parameter: -unknown
```

## Return Value

Returns a tensor handle (string) containing the computed quantile value(s). The returned tensor has the same data type as the input tensor.

## Notes

- **Backward Compatibility**: The positional syntax is fully backward compatible with existing code.
- **Quantile Range**: The quantile value `q` must be between 0.0 and 1.0 inclusive.
- **Dimension Handling**: If no dimension is specified, the quantile is computed over all elements. If a dimension is specified, the quantile is computed along that dimension.
- **Memory Management**: The returned tensor handle should be managed by the calling code. Tensors are automatically cleaned up when the Tcl interpreter is destroyed.
- **Performance**: For large tensors, computing quantiles can be computationally expensive, especially when no dimension is specified.

## Migration Guide

### From Positional to Named Syntax

**Old (Positional):**
```tcl
set result [torch::tensor_quantile $tensor 0.5]
set result [torch::tensor_quantile $tensor 0.5 0]
```

**New (Named Parameters):**
```tcl
set result [torch::tensor_quantile -input $tensor -q 0.5]
set result [torch::tensor_quantile -input $tensor -q 0.5 -dim 0]
```

### Using CamelCase Alias

**Snake_case:**
```tcl
set result [torch::tensor_quantile $tensor 0.5]
```

**CamelCase:**
```tcl
set result [torch::tensorQuantile $tensor 0.5]
```

## Related Commands

- `torch::tensor_median` - Compute the median of tensor elements
- `torch::tensor_min` - Find the minimum value in a tensor
- `torch::tensor_max` - Find the maximum value in a tensor
- `torch::tensor_mean` - Compute the mean of tensor elements
- `torch::tensor_sum` - Compute the sum of tensor elements 