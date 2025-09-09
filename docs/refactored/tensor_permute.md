# torch::tensor_permute

Permutes the dimensions of a tensor according to a given order.

## Description

The `torch::tensor_permute` command rearranges the dimensions of the input tensor according to the specified order. This is useful for changing the shape of tensors for operations like transposing, reshaping for neural networks, or preparing data for different APIs.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_permute tensor dims
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_permute -input tensor -dims {order}
```

### CamelCase Alias
```tcl
torch::tensorPermute -input tensor -dims {order}
```

## Parameters

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| input     | string | Yes      | Name of the input tensor           |
| dims      | list   | Yes      | New order of dimensions (e.g. {2 1 0}) |

## Return Value

Returns a string containing the handle name of the resulting permuted tensor.

## Examples

### Basic Usage
```tcl
# Create a 3D tensor
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
set a3d [torch::tensor_reshape $a {1 2 3}]

# Using positional syntax
set result1 [torch::tensor_permute $a3d {2 1 0}]

# Using named parameter syntax
set result2 [torch::tensor_permute -input $a3d -dims {2 1 0}]

# Using camelCase alias
set result3 [torch::tensorPermute -input $a3d -dims {2 1 0}]
```

### Edge Cases
```tcl
# Single dimension
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set a1d [torch::tensor_reshape $a {3}]
set result [torch::tensor_permute $a1d {0}]

# Reverse dimensions
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set a2d [torch::tensor_reshape $a {2 2}]
set result [torch::tensor_permute $a2d {1 0}]
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_permute invalid_tensor {0 1 2}} result
puts $result  # Output: Invalid tensor name
```

### Missing Parameters
```tcl
catch {torch::tensor_permute} result
puts $result  # Output: Required parameters missing: input and dims
```

### Unknown Parameter
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_permute -input $a -dims {0} -unknown_param value} result
puts $result  # Output: Unknown parameter: -unknown_param
```

## Migration Guide

### From Old Syntax to New Syntax
**Before (Positional Only):**
```tcl
set result [torch::tensor_permute $a3d {2 1 0}]
```
**After (Named Parameters):**
```tcl
set result [torch::tensor_permute -input $a3d -dims {2 1 0}]
```
**After (CamelCase):**
```tcl
set result [torch::tensorPermute -input $a3d -dims {2 1 0}]
```

### Benefits of New Syntax
- Clarity: Parameter names make the code more readable
- Maintainability: Easier to understand and modify
- Consistency: Follows modern API design patterns
- Backward Compatibility: Old syntax still works

## Technical Notes
- The `dims` list must be a valid permutation of the tensor's dimensions.
- The output tensor will have the same data as the input but with permuted axes.
- This operation does not copy data unless necessary (view semantics).

## Related Commands
- `torch::tensor_reshape` - Reshape tensor
- `torch::tensor_stack` - Stack tensors along a new dimension
- `torch::tensor_cat` - Concatenate tensors along an existing dimension 