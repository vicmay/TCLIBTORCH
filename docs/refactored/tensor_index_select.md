# torch::tensor_index_select

Selects elements from a tensor along a specified dimension using index values.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_index_select tensor dim indices
```

### Named Parameter Syntax
```tcl
torch::tensor_index_select -input tensor -dim dim -indices indices
torch::tensor_index_select -tensor tensor -dimension dim -indices indices
```

### CamelCase Alias
```tcl
torch::tensorIndexSelect tensor dim indices
torch::tensorIndexSelect -input tensor -dim dim -indices indices
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Name of the input tensor |
| `dim` / `-dim` / `-dimension` | integer | Yes | Dimension along which to select elements (0-based) |
| `indices` / `-indices` | string | Yes | Name of the indices tensor (must be int32 or int64 dtype) |

## Description

The `torch::tensor_index_select` command selects elements from a tensor along a specified dimension using the provided indices. This is equivalent to PyTorch's `torch.index_select()` function.

The indices tensor must contain integer values (int32 or int64 dtype) that specify which elements to select along the given dimension. The result tensor will have the same shape as the input tensor, except for the specified dimension, which will have a size equal to the number of indices.

## Examples

### Basic Usage

```tcl
# Create a 2x3 tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]

# Create indices tensor (must be int32 or int64)
set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]

# Select rows 0 and 1 along dimension 0
set result [torch::tensor_index_select $tensor 0 $indices]
# Result: tensor with shape [2, 3] containing rows 0 and 1

# Select columns 0 and 2 along dimension 1
set result [torch::tensor_index_select $tensor 1 $indices]
# Result: tensor with shape [2, 2] containing columns 0 and 2
```

### Named Parameter Syntax

```tcl
# Using -input parameter
set result [torch::tensor_index_select -input $tensor -dim 0 -indices $indices]

# Using -tensor alias
set result [torch::tensor_index_select -tensor $tensor -dim 0 -indices $indices]

# Using -dimension alias
set result [torch::tensor_index_select -input $tensor -dimension 1 -indices $indices]
```

### CamelCase Alias

```tcl
# Using camelCase alias with positional syntax
set result [torch::tensorIndexSelect $tensor 0 $indices]

# Using camelCase alias with named parameters
set result [torch::tensorIndexSelect -input $tensor -dim 0 -indices $indices]
```

### 3D Tensor Example

```tcl
# Create a 2x2x2 tensor
set tensor [torch::tensor_create -data {1 2 3 4 5 6 7 8} -shape {2 2 2}]

# Select along dimension 0
set indices [torch::tensor_create -data {0} -shape {1} -dtype int32]
set result [torch::tensor_index_select $tensor 0 $indices]
# Result: tensor with shape [1, 2, 2] containing the first 2x2 slice

# Select along dimension 2 (last dimension)
set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
set result [torch::tensor_index_select $tensor 2 $indices]
# Result: tensor with shape [2, 2, 2] containing all elements
```

### Different Data Types

```tcl
# Float tensor
set tensor [torch::tensor_create -data {1.5 2.5 3.5 4.5 5.5 6.5} -shape {2 3} -dtype float32]
set indices [torch::tensor_create -data {0 1} -shape {2} -dtype int32]
set result [torch::tensor_index_select $tensor 0 $indices]
```

## Migration Guide

### From Positional to Named Parameters

**Old syntax:**
```tcl
torch::tensor_index_select $tensor 0 $indices
```

**New syntax:**
```tcl
torch::tensor_index_select -input $tensor -dim 0 -indices $indices
```

### From snake_case to camelCase

**Old syntax:**
```tcl
torch::tensor_index_select $tensor 0 $indices
```

**New syntax:**
```tcl
torch::tensorIndexSelect $tensor 0 $indices
```

## Error Handling

The command provides clear error messages for various error conditions:

### Invalid Tensor Name
```tcl
torch::tensor_index_select nonexistent_tensor 0 $indices
# Error: Invalid tensor name
```

### Invalid Indices Tensor Name
```tcl
torch::tensor_index_select $tensor 0 nonexistent_indices
# Error: Invalid indices tensor name
```

### Invalid Dimension Value
```tcl
torch::tensor_index_select $tensor invalid_dim $indices
# Error: Invalid dimension value
```

### Missing Required Parameters
```tcl
torch::tensor_index_select -input $tensor -dim 0
# Error: Required parameters missing: input tensor and indices tensor are required
```

### Unknown Parameter
```tcl
torch::tensor_index_select -input $tensor -dim 0 -indices $indices -unknown_param value
# Error: Unknown parameter: -unknown_param
```

### Missing Value for Parameter
```tcl
torch::tensor_index_select -input $tensor -dim
# Error: Missing value for parameter
```

### Wrong Number of Arguments
```tcl
torch::tensor_index_select $tensor 0
# Error: Invalid number of arguments
```

## Important Notes

1. **Indices Data Type**: The indices tensor must have dtype int32 or int64. Using other data types will result in a runtime error.

2. **Dimension Bounds**: The dimension value must be within the valid range for the input tensor (0 to tensor.ndim - 1).

3. **Index Bounds**: The indices values should be within the valid range for the specified dimension (0 to tensor.size(dim) - 1).

4. **Result Shape**: The result tensor will have the same shape as the input tensor, except for the specified dimension, which will have a size equal to the number of indices.

5. **Memory Management**: The command automatically manages tensor memory. No manual cleanup is required.

## Related Commands

- `torch::tensor_create` - Create tensors
- `torch::tensor_to_list` - Convert tensor to list
- `torch::tensor_shape` - Get tensor shape
- `torch::tensor_slice` - Slice tensors
- `torch::tensor_advanced_index` - Advanced indexing operations

## PyTorch Equivalent

This command is equivalent to PyTorch's `torch.index_select()` function:

```python
# PyTorch equivalent
result = torch.index_select(tensor, dim, indices)
```

## Performance Considerations

- The operation is efficient for small to medium-sized tensors
- For large tensors, consider using other indexing methods like `torch::tensor_slice` or `torch::tensor_advanced_index`
- The indices tensor should be on the same device as the input tensor for optimal performance 