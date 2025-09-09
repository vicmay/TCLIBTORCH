# torch::tensor_split

Splits a tensor into multiple tensors along a specified dimension.

## Description

The `torch::tensor_split` command splits a tensor into multiple tensors along a specified dimension. It can split either by specifying the number of sections or by providing specific indices where to split.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_split tensor sections_or_indices ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_split -input tensor -sections sections_or_indices ?-dim dim?
torch::tensor_split -tensor tensor -indices indices ?-dim dim?
torch::tensor_split -input tensor -sections sections_or_indices ?-dimension dim?
```

### CamelCase Alias
```tcl
torch::tensorSplit tensor sections_or_indices ?dim?
torch::tensorSplit -input tensor -sections sections_or_indices ?-dim dim?
torch::tensorSplit -tensor tensor -indices indices ?-dim dim?
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `tensor` / `-input` / `-tensor` | string | Tensor handle to split | Yes |
| `sections_or_indices` / `-sections` / `-indices` | int or list | Number of sections or list of split indices | Yes |
| `dim` / `-dim` / `-dimension` | int | Dimension along which to split (default: 0) | No |

## Return Value

Returns a list of tensor handles representing the split tensors.

## Examples

### Basic Usage - Split by Number of Sections

```tcl
# Create a 1D tensor
set tensor [torch::tensor_create {1 2 3 4 5 6}]

# Split into 3 equal parts (positional syntax)
set result [torch::tensor_split $tensor 3]
puts [llength $result]  ;# Output: 3

# Same using named parameters
set result [torch::tensor_split -input $tensor -sections 3]
puts [llength $result]  ;# Output: 3

# Using camelCase alias
set result [torch::tensorSplit -input $tensor -sections 3]
puts [llength $result]  ;# Output: 3
```

### Split by Indices

```tcl
# Create a 1D tensor
set tensor [torch::tensor_create {1 2 3 4 5 6}]

# Split at indices 2 and 4 (positional syntax)
set result [torch::tensor_split $tensor {2 4}]
puts [llength $result]  ;# Output: 3

# Same using named parameters
set result [torch::tensor_split -tensor $tensor -indices {2 4}]
puts [llength $result]  ;# Output: 3
```

### Split Along Different Dimensions

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}}]

# Split along first dimension (rows)
set result [torch::tensor_split $tensor 2 0]
puts [llength $result]  ;# Output: 2

# Split along second dimension (columns)
set result [torch::tensor_split -input $tensor -sections 2 -dim 1]
puts [llength $result]  ;# Output: 2

# Using alternative parameter name
set result [torch::tensor_split -tensor $tensor -sections 2 -dimension 1]
puts [llength $result]  ;# Output: 2
```

### Complex Splitting Scenarios

```tcl
# Create a larger tensor
set tensor [torch::tensor_create {1 2 3 4 5 6 7 8 9 10}]

# Split at multiple indices
set result [torch::tensor_split $tensor {2 5 8}]
puts [llength $result]  ;# Output: 4

# Split more parts than elements (creates empty tensors)
set tensor [torch::tensor_create {1 2 3}]
set result [torch::tensor_split $tensor 5]
puts [llength $result]  ;# Output: 5
```

## Edge Cases

### Split into Single Part
```tcl
set tensor [torch::tensor_create {1 2 3}]
set result [torch::tensor_split $tensor 1]
puts [llength $result]  ;# Output: 1 (returns original tensor)
```

### Split at Boundaries
```tcl
set tensor [torch::tensor_create {1 2 3 4 5 6}]
set result [torch::tensor_split $tensor {3 6}]
puts [llength $result]  ;# Output: 3
```

### Split More Parts Than Elements
```tcl
set tensor [torch::tensor_create {1 2 3}]
set result [torch::tensor_split $tensor 5]
puts [llength $result]  ;# Output: 5 (some parts may be empty)
```

## Data Type Support

The command works with all tensor data types:

```tcl
# Float32 tensor
set tensor [torch::tensor_create {1.5 2.5 3.5 4.5} float32]
set result [torch::tensor_split $tensor 2]
puts [llength $result]  ;# Output: 2

# Int64 tensor
set tensor [torch::tensor_create {1 2 3 4 5 6} int64]
set result [torch::tensor_split $tensor 3]
puts [llength $result]  ;# Output: 3
```

## Multi-dimensional Tensor Examples

### 2D Tensor - Split Along First Dimension
```tcl
set tensor [torch::tensor_create {{1 2} {3 4} {5 6} {7 8}}]
set result [torch::tensor_split $tensor 2 0]
puts [llength $result]  ;# Output: 2
```

### 2D Tensor - Split Along Second Dimension
```tcl
set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8}}]
set result [torch::tensor_split $tensor 2 1]
puts [llength $result]  ;# Output: 2
```

### 2D Tensor - Split by Indices
```tcl
set tensor [torch::tensor_create {{1 2 3} {4 5 6} {7 8 9} {10 11 12}}]
set result [torch::tensor_split $tensor {1 3} 0]
puts [llength $result]  ;# Output: 3
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_split invalid_tensor 3} result
puts $result  ;# Output: Invalid tensor name
```

### Missing Sections Parameter
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_split $tensor} result
puts $result  ;# Output: Error in tensor_split: Invalid number of arguments
```

### Empty Sections List
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_split $tensor {}} result
puts $result  ;# Output: Error in tensor_split: Required parameters missing: input tensor and sections/indices are required
```

### Invalid Named Parameter
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_split -invalid $tensor -sections 3} result
puts $result  ;# Output: Error in tensor_split: Unknown parameter: -invalid
```

### Missing Parameter Value
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_split -input $tensor -sections} result
puts $result  ;# Output: Error in tensor_split: Missing value for parameter
```

### Invalid Dimension
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_split -input $tensor -sections 3 -dim invalid} result
puts $result  ;# Output: Error in tensor_split: Invalid dimension value
```

## Migration Guide

### From Positional to Named Parameters

**Old Code:**
```tcl
set result [torch::tensor_split $tensor 3]
set result [torch::tensor_split $tensor {2 4} 0]
```

**New Code:**
```tcl
set result [torch::tensor_split -input $tensor -sections 3]
set result [torch::tensor_split -tensor $tensor -indices {2 4} -dim 0]
```

### Using CamelCase Alias

**Old Code:**
```tcl
set result [torch::tensor_split $tensor 3]
```

**New Code:**
```tcl
set result [torch::tensorSplit $tensor 3]
# or
set result [torch::tensorSplit -input $tensor -sections 3]
```

## Notes

- When splitting by number of sections, the tensor is divided as evenly as possible
- When splitting by indices, the tensor is split at the specified positions
- If more sections are requested than elements in the dimension, some resulting tensors may be empty
- The default dimension is 0 (first dimension)
- Both positional and named parameter syntax produce identical results
- The camelCase alias (`torch::tensorSplit`) is functionally identical to the snake_case version
- The returned list contains tensor handles that can be used with other tensor operations

## See Also

- `torch::hsplit` - Split tensor horizontally
- `torch::vsplit` - Split tensor vertically
- `torch::dsplit` - Split tensor along depth dimension
- `torch::tensor_chunk` - Split tensor into chunks
- `torch::tensor_unbind` - Remove a tensor dimension 