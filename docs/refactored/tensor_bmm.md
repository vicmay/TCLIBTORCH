# torch::tensor_bmm

Performs batch matrix multiplication between two 3D tensors.

## Description

The `torch::tensor_bmm` command performs batch matrix multiplication between two 3D tensors. This operation is useful for processing multiple matrices simultaneously, commonly used in neural networks for batch processing.

**Note**: Both input tensors must be 3D tensors with compatible dimensions for matrix multiplication.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_bmm input_tensor other_tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_bmm -input input_tensor -other other_tensor
```

### CamelCase Alias
```tcl
torch::tensorBmm -input input_tensor -other other_tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Name of the first 3D tensor |
| `other` | string | Yes | Name of the second 3D tensor |

## Return Value

Returns a string containing the handle name of the resulting tensor.

## Examples

### Basic Usage

```tcl
# Create 3D tensors for batch matrix multiplication
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set a3d [torch::tensor_reshape $a {2 2 2}]

set b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
set b3d [torch::tensor_reshape $b {2 2 2}]

# Using positional syntax
set result1 [torch::tensor_bmm $a3d $b3d]

# Using named parameter syntax
set result2 [torch::tensor_bmm -input $a3d -other $b3d]

# Using camelCase alias
set result3 [torch::tensorBmm -input $a3d -other $b3d]
```

### Different Batch Sizes

```tcl
# Single batch (1x2x2)
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set a3d [torch::tensor_reshape $a {1 2 2}]

set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set b3d [torch::tensor_reshape $b {1 2 2}]

set result [torch::tensor_bmm -input $a3d -other $b3d]
```

### Larger Batch Size

```tcl
# 3x3x3 batch
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0 17.0 18.0 19.0 20.0 21.0 22.0 23.0 24.0 25.0 26.0 27.0} -dtype float32 -device cpu]
set a3d [torch::tensor_reshape $a {3 3 3}]

set b [torch::tensor_create -data {28.0 29.0 30.0 31.0 32.0 33.0 34.0 35.0 36.0 37.0 38.0 39.0 40.0 41.0 42.0 43.0 44.0 45.0 46.0 47.0 48.0 49.0 50.0 51.0 52.0 53.0 54.0} -dtype float32 -device cpu]
set b3d [torch::tensor_reshape $b {3 3 3}]

set result [torch::tensor_bmm -input $a3d -other $b3d]
```

## Error Handling

### Invalid Tensor Names
```tcl
# This will throw an error
catch {torch::tensor_bmm invalid_tensor $b3d} result
puts $result
# Output: Invalid input tensor name
```

### Missing Parameters
```tcl
# This will throw an error
catch {torch::tensor_bmm} result
puts $result
# Output: Required parameters missing: input and other
```

### Unknown Parameters
```tcl
# This will throw an error
catch {torch::tensor_bmm -input $a3d -other $b3d -unknown_param value} result
puts $result
# Output: Unknown parameter: -unknown_param
```

## Migration Guide

### From Old Syntax to New Syntax

**Before (Positional Only):**
```tcl
set result [torch::tensor_bmm $a3d $b3d]
```

**After (Named Parameters):**
```tcl
set result [torch::tensor_bmm -input $a3d -other $b3d]
```

**After (CamelCase):**
```tcl
set result [torch::tensorBmm -input $a3d -other $b3d]
```

### Benefits of New Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Maintainability**: Easier to understand and modify
3. **Consistency**: Follows modern API design patterns
4. **Backward Compatibility**: Old syntax still works

## Technical Notes

- **Tensor Requirements**: Both input tensors must be 3D tensors
- **Dimension Compatibility**: The inner dimensions must be compatible for matrix multiplication
- **Batch Processing**: This operation is optimized for processing multiple matrices in parallel
- **Memory Usage**: The operation creates a new tensor for the result

## Related Commands

- `torch::tensor_matmul` - Matrix multiplication for 2D tensors
- `torch::tensor_mul` - Element-wise multiplication
- `torch::tensor_reshape` - Reshape tensors for batch operations 