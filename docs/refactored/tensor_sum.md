# torch::tensor_sum

Computes the sum of tensor elements, optionally along specified dimensions.

## Description

The `torch::tensor_sum` command computes the sum of all elements in a tensor. When a dimension is specified, it computes the sum along that dimension, reducing the tensor's dimensionality. When no dimension is specified, it computes the sum of all elements, returning a scalar tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_sum tensor ?dim?
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_sum -input tensor [-dim dimension]
```

### CamelCase Alias
```tcl
torch::tensorSum -input tensor [-dim dimension]
```

## Parameters

| Parameter | Type   | Required | Default | Description                        |
|-----------|--------|----------|---------|------------------------------------|
| input     | string | Yes      | -       | Name of the input tensor           |
| dim       | int    | No       | -       | Dimension along which to sum (optional) |

## Return Value

Returns a string containing the handle name of the resulting tensor.

## Examples

### Basic Usage
```tcl
# Create a tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]

# Using positional syntax - sum all elements
set result1 [torch::tensor_sum $t]

# Using named parameter syntax - sum all elements
set result2 [torch::tensor_sum -input $t]

# Using camelCase alias - sum all elements
set result3 [torch::tensorSum -input $t]
```

### Sum Along Dimensions
```tcl
# Create a 2D tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
set t2d [torch::tensor_reshape $t {2 3}]

# Sum along dimension 0 (rows)
set result1 [torch::tensor_sum $t2d 0]

# Sum along dimension 1 (columns) using named parameters
set result2 [torch::tensor_sum -input $t2d -dim 1]

# Sum along dimension 0 using camelCase
set result3 [torch::tensorSum -input $t2d -dim 0]
```

### Multi-dimensional Examples
```tcl
# Create a 3D tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set t3d [torch::tensor_reshape $t {2 2 2}]

# Sum all elements (scalar result)
set total [torch::tensor_sum $t3d]

# Sum along dimension 0 (reduces first dimension)
set sum_dim0 [torch::tensor_sum -input $t3d -dim 0]

# Sum along dimension 1 (reduces second dimension)
set sum_dim1 [torch::tensor_sum -input $t3d -dim 1]

# Sum along dimension 2 (reduces third dimension)
set sum_dim2 [torch::tensor_sum -input $t3d -dim 2]
```

### Mathematical Examples
```tcl
# Example: Sum of [1, 2, 3, 4] = 10
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set result [torch::tensor_sum $t]

# Example: 2D tensor [[1, 2, 3], [4, 5, 6]]
# Sum along dim 0: [1+4, 2+5, 3+6] = [5, 7, 9]
# Sum along dim 1: [1+2+3, 4+5+6] = [6, 15]
set t2d [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
set t2d [torch::tensor_reshape $t2d {2 3}]
set sum_dim0 [torch::tensor_sum -input $t2d -dim 0]
set sum_dim1 [torch::tensor_sum -input $t2d -dim 1]
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_sum invalid_tensor} result
puts $result  # Output: Invalid tensor name
```

### Missing Parameters
```tcl
catch {torch::tensor_sum} result
puts $result  # Output: Required parameter missing: input
```

### Too Many Parameters
```tcl
set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_sum $t 0 extra} result
puts $result  # Output: Usage: torch::tensor_sum tensor ?dim?
```

### Unknown Parameter
```tcl
set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_sum -input $t -unknown_param value} result
puts $result  # Output: Unknown parameter: -unknown_param
```

### Dimension Out of Bounds
```tcl
set t [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_sum -input $t -dim 5} result
puts $result  # Output: Error message from PyTorch
```

## Migration Guide

### From Old Syntax to New Syntax
**Before (Positional Only):**
```tcl
set result [torch::tensor_sum $t]
set result [torch::tensor_sum $t 0]
```
**After (Named Parameters):**
```tcl
set result [torch::tensor_sum -input $t]
set result [torch::tensor_sum -input $t -dim 0]
```
**After (CamelCase):**
```tcl
set result [torch::tensorSum -input $t]
set result [torch::tensorSum -input $t -dim 0]
```

### Benefits of New Syntax
- **Clarity**: Parameter names make the code more readable
- **Flexibility**: Easy to specify optional parameters
- **Maintainability**: Easier to understand and modify
- **Consistency**: Follows modern API design patterns
- **Backward Compatibility**: Old syntax still works

## Mathematical Details

### Sum All Elements
When no dimension is specified, the operation computes:
```
result = sum(tensor)
```

### Sum Along Dimension
When a dimension is specified, the operation reduces that dimension:
```
result = sum(tensor, dim=dimension)
```

### Examples of Mathematical Operations
```tcl
# 1D tensor: [1, 2, 3, 4]
# sum() = 1 + 2 + 3 + 4 = 10

# 2D tensor: [[1, 2, 3], [4, 5, 6]]
# sum(dim=0) = [1+4, 2+5, 3+6] = [5, 7, 9]
# sum(dim=1) = [1+2+3, 4+5+6] = [6, 15]

# 3D tensor: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
# sum(dim=0) = [[1+5, 2+6], [3+7, 4+8]] = [[6, 8], [10, 12]]
# sum(dim=1) = [[1+3, 2+4], [5+7, 6+8]] = [[4, 6], [12, 14]]
# sum(dim=2) = [[1+2, 3+4], [5+6, 7+8]] = [[3, 7], [11, 15]]
```

## Technical Notes
- When no dimension is specified, the result is a scalar tensor (0-dimensional)
- When a dimension is specified, the result has one fewer dimension than the input
- The operation supports tensors of any dimensionality
- The operation preserves the data type of the input tensor
- This operation does not modify the original tensor
- For empty tensors, the sum is 0 (or the appropriate zero value for the data type)

## Edge Cases

### Single Element Tensor
```tcl
set t [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
set result [torch::tensor_sum $t]  # Result: 5.0
```

### Zero Tensor
```tcl
set t [torch::tensor_create -data {0.0 0.0 0.0 0.0} -dtype float32 -device cpu]
set result [torch::tensor_sum $t]  # Result: 0.0
```

### Negative Values
```tcl
set t [torch::tensor_create -data {-1.0 -2.0 -3.0 -4.0} -dtype float32 -device cpu]
set result [torch::tensor_sum $t]  # Result: -10.0
```

## Related Commands
- `torch::tensor_mean` - Compute mean of tensor elements
- `torch::tensor_max` - Find maximum value in tensor
- `torch::tensor_min` - Find minimum value in tensor
- `torch::tensor_prod` - Compute product of tensor elements
- `torch::tensor_std` - Compute standard deviation of tensor elements 