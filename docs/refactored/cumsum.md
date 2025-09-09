# torch::cumsum

Computes the cumulative sum of elements along a specified dimension.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::cumsum -input tensor_name -dim dimension
torch::cumSum -input tensor_name -dim dimension  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::cumsum tensor_name dimension
torch::cumSum tensor_name dimension  # camelCase alias
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | string | Yes | - | Name of the input tensor |
| `-dim` | integer | No | 0 | Dimension along which to compute cumulative sum |

## Return Value

Returns a new tensor handle containing the cumulative sum values along the specified dimension.

## Description

The `torch::cumsum` function computes the cumulative sum of tensor elements along a specified dimension. For each position in the output tensor, the value is the sum of all elements from the beginning of the dimension up to that position.

### Mathematical Operation

For a 1D tensor `[a, b, c, d]`, the cumulative sum would be:
- `[a, a+b, a+b+c, a+b+c+d]`

For multi-dimensional tensors, the operation is applied along the specified dimension while preserving the structure of other dimensions.

## Examples

### Basic Usage
```tcl
# Create a test tensor
set data {1.0 2.0 3.0 4.0}
set tensor [torch::tensor_create -data $data -shape {4} -dtype float32]

# Compute cumulative sum using named parameters
set result [torch::cumsum -input $tensor -dim 0]

# Using camelCase alias
set result2 [torch::cumSum -input $tensor -dim 0]

# Legacy positional syntax
set result3 [torch::cumsum $tensor 0]
```

### Multi-dimensional Example
```tcl
# Create a 2D tensor
set data {1.0 2.0 3.0 4.0 5.0 6.0}
set tensor [torch::tensor_create -data $data -shape {2 3} -dtype float32]

# Cumulative sum along rows (dimension 0)
set row_cumsum [torch::cumsum -input $tensor -dim 0]

# Cumulative sum along columns (dimension 1)
set col_cumsum [torch::cumsum -input $tensor -dim 1]
```

### Working with Different Data Types
```tcl
# Integer tensor
set int_data {1 2 3 4}
set int_tensor [torch::tensor_create -data $int_data -shape {4} -dtype int32]
set int_result [torch::cumsum -input $int_tensor -dim 0]

# Float64 tensor
set float_data {1.5 2.5 3.5 4.5}
set float_tensor [torch::tensor_create -data $float_data -shape {4} -dtype float64]
set float_result [torch::cumsum -input $float_tensor -dim 0]
```

## Error Handling

The function performs comprehensive validation and provides clear error messages:

```tcl
# Missing required parameter
torch::cumsum -dim 0
# Error: Required parameter missing: -input

# Invalid tensor name
torch::cumsum -input "nonexistent" -dim 0
# Error: Invalid tensor name

# Invalid dimension value
torch::cumsum -input $tensor -dim "invalid"
# Error: Invalid dim value. Expected integer.

# Wrong number of positional arguments
torch::cumsum $tensor
# Error: Wrong number of arguments for positional syntax. Expected: torch::cumsum tensor dim
```

## Technical Notes

### Performance Considerations
- The cumulative sum operation has O(n) time complexity for the specified dimension
- Memory usage is the same as the input tensor size
- Large tensors are handled efficiently through PyTorch's optimized implementation

### Numerical Stability
- For integer tensors, be aware of potential overflow with large values
- For floating-point tensors, accumulated precision errors may occur with very long sequences
- Consider using appropriate data types for your precision requirements

### Dimension Handling
- The `dim` parameter must be within the valid range `[0, tensor.dim()-1]`
- Negative dimension indices are supported following PyTorch conventions (e.g., -1 for last dimension)
- The output tensor has the same shape as the input tensor

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::cumsum $tensor 0]
```

**New named parameter syntax:**
```tcl
set result [torch::cumsum -input $tensor -dim 0]
```

**Or using camelCase alias:**
```tcl
set result [torch::cumSum -input $tensor -dim 0]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and understand code
- **Error Prevention**: Less likely to mix up parameter positions

## Related Commands

- `torch::cumprod` - Cumulative product along a dimension
- `torch::cummax` - Cumulative maximum along a dimension  
- `torch::cummin` - Cumulative minimum along a dimension
- `torch::sum` - Sum of all elements or along a dimension
- `torch::diff` - Differences between consecutive elements

## See Also

- [PyTorch cumsum documentation](https://pytorch.org/docs/stable/generated/torch.cumsum.html)
- [Tensor Creation Operations](../tensor_creation_ops.md)
- [Mathematical Operations](../mathematical_operations.md) 