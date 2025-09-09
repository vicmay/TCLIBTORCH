# torch::tensor_repeat

Repeats a tensor along specified dimensions.

## Description

The `torch::tensor_repeat` command creates a new tensor by repeating the input tensor along specified dimensions. This is useful for broadcasting and expanding tensors to match desired shapes.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_repeat tensor repeats
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_repeat -input tensor -repeats repeats
torch::tensor_repeat -tensor tensor -repeats repeats
```

### CamelCase Alias
```tcl
torch::tensorRepeat tensor repeats
torch::tensorRepeat -input tensor -repeats repeats
torch::tensorRepeat -tensor tensor -repeats repeats
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `tensor` / `-input` / `-tensor` | string | Tensor handle to repeat | Yes |
| `repeats` / `-repeats` | list | List of integers specifying how many times to repeat along each dimension | Yes |

## Return Value

Returns a string handle to the new repeated tensor.

## Examples

### Basic Usage

```tcl
# Create a simple 1D tensor
set tensor [torch::tensor_create {1 2 3}]

# Repeat it 2 times (positional syntax)
set result [torch::tensor_repeat $tensor {2}]
puts [torch::tensor_shape $result]  ;# Output: 6

# Same using named parameters
set result [torch::tensor_repeat -input $tensor -repeats {2}]
puts [torch::tensor_shape $result]  ;# Output: 6

# Using camelCase alias
set result [torch::tensorRepeat -input $tensor -repeats {2}]
puts [torch::tensor_shape $result]  ;# Output: 6
```

### 2D Tensor Repeating

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1 2} {3 4}}]

# Repeat 2 times along first dimension, 3 times along second
set result [torch::tensor_repeat $tensor {2 3}]
puts [torch::tensor_shape $result]  ;# Output: 4 6

# Using named parameters
set result [torch::tensor_repeat -tensor $tensor -repeats {2 3}]
puts [torch::tensor_shape $result]  ;# Output: 4 6
```

### Scalar Tensor Repeating

```tcl
# Create a scalar tensor
set tensor [torch::tensor_create 5]

# Repeat to create a 3x2 tensor
set result [torch::tensor_repeat $tensor {3 2}]
puts [torch::tensor_shape $result]  ;# Output: 3 2
```

### Complex Repeating Patterns

```tcl
# Create a 1D tensor
set tensor [torch::tensor_create {1 2}]

# Repeat 3 times along first dimension, 2 times along second
set result [torch::tensor_repeat -input $tensor -repeats {3 2}]
puts [torch::tensor_shape $result]  ;# Output: 3 4
```

## Edge Cases

### Repeat by 1 (No Change)
```tcl
set tensor [torch::tensor_create {1 2 3}]
set result [torch::tensor_repeat $tensor {1}]
puts [torch::tensor_shape $result]  ;# Output: 3 (same as original)
```

### Large Repeats
```tcl
set tensor [torch::tensor_create {1 2}]
set result [torch::tensor_repeat $tensor {10 5}]
puts [torch::tensor_shape $result]  ;# Output: 10 10
```

### Zero Tensor
```tcl
set tensor [torch::tensor_create 0]
set result [torch::tensor_repeat $tensor {3 2}]
puts [torch::tensor_shape $result]  ;# Output: 3 2
```

## Data Type Support

The command works with all tensor data types:

```tcl
# Float32 tensor
set tensor [torch::tensor_create {1.5 2.5 3.5} float32]
set result [torch::tensor_repeat $tensor {2}]
puts [torch::tensor_shape $result]  ;# Output: 6

# Int64 tensor
set tensor [torch::tensor_create {1 2 3} int64]
set result [torch::tensor_repeat $tensor {3}]
puts [torch::tensor_shape $result]  ;# Output: 9
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_repeat invalid_tensor {2 3}} result
puts $result  ;# Output: Invalid tensor name
```

### Missing Repeats Parameter
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_repeat $tensor} result
puts $result  ;# Output: Invalid number of arguments
```

### Empty Repeats List
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_repeat $tensor {}} result
puts $result  ;# Output: Required parameters missing: input tensor and repeats are required
```

### Invalid Named Parameter
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_repeat -invalid $tensor -repeats {2}} result
puts $result  ;# Output: Unknown parameter: -invalid
```

### Missing Parameter Value
```tcl
set tensor [torch::tensor_create {1 2 3}]
catch {torch::tensor_repeat -input $tensor -repeats} result
puts $result  ;# Output: Missing value for parameter
```

## Migration Guide

### From Positional to Named Parameters

**Old Code:**
```tcl
set result [torch::tensor_repeat $tensor {2 3}]
```

**New Code:**
```tcl
set result [torch::tensor_repeat -input $tensor -repeats {2 3}]
# or
set result [torch::tensor_repeat -tensor $tensor -repeats {2 3}]
```

### Using CamelCase Alias

**Old Code:**
```tcl
set result [torch::tensor_repeat $tensor {2 3}]
```

**New Code:**
```tcl
set result [torch::tensorRepeat $tensor {2 3}]
# or
set result [torch::tensorRepeat -input $tensor -repeats {2 3}]
```

## Notes

- The `repeats` parameter must be a list of positive integers
- The length of the repeats list determines the number of dimensions in the output tensor
- If the repeats list is shorter than the input tensor's dimensions, the remaining dimensions are not repeated
- If the repeats list is longer than the input tensor's dimensions, the input tensor is treated as a scalar and repeated accordingly
- Both positional and named parameter syntax produce identical results
- The camelCase alias (`torch::tensorRepeat`) is functionally identical to the snake_case version

## See Also

- `torch::tensor_expand` - Expand tensor dimensions without copying data
- `torch::tensor_reshape` - Reshape tensor dimensions
- `torch::tensor_view` - Create a view of tensor with different shape 