# torch::cumprod

Computes the cumulative product of elements along a specified dimension.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::cumprod -input tensor_name -dim dimension
torch::cumProd -input tensor_name -dim dimension  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::cumprod tensor_name dimension
torch::cumProd tensor_name dimension  # camelCase alias
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | string | Yes | - | Name of the input tensor |
| `-dim` | integer | No | 0 | Dimension along which to compute cumulative product |

## Return Value

Returns a new tensor handle containing the cumulative product values along the specified dimension.

## Description

The `torch::cumprod` function computes the cumulative product of tensor elements along a specified dimension. For each position in the output tensor, the value is the product of all elements from the beginning of the dimension up to that position.

### Mathematical Operation

For a 1D tensor `[a, b, c, d]`, the cumulative product would be:
- `[a, a*b, a*b*c, a*b*c*d]`

For multi-dimensional tensors, the operation is applied along the specified dimension while preserving the structure of other dimensions.

## Examples

### Basic Usage
```tcl
# Create a test tensor
set data {1.0 2.0 3.0 4.0}
set tensor [torch::tensor -data $data -shape {4} -dtype float32]

# Compute cumulative product using named parameters
set result [torch::cumprod -input $tensor -dim 0]

# Using camelCase alias
set result2 [torch::cumProd -input $tensor -dim 0]

# Legacy positional syntax
set result3 [torch::cumprod $tensor 0]
```

### Multi-dimensional Example
```tcl
# Create a 2D tensor
set data {1.0 2.0 3.0 4.0 5.0 6.0}
set tensor [torch::tensor -data $data -shape {2 3} -dtype float32]

# Cumulative product along rows (dimension 0)
set row_cumprod [torch::cumprod -input $tensor -dim 0]

# Cumulative product along columns (dimension 1)
set col_cumprod [torch::cumprod -input $tensor -dim 1]
```

### Working with Different Data Types
```tcl
# Integer tensor
set int_data {1 2 3 4}
set int_tensor [torch::tensor -data $int_data -shape {4} -dtype int32]
set int_result [torch::cumprod -input $int_tensor -dim 0]

# Float tensor with small values
set float_data {0.5 2.0 0.25 4.0}
set float_tensor [torch::tensor -data $float_data -shape {4} -dtype float32]
set float_result [torch::cumprod -input $float_tensor -dim 0]
```

## Error Handling

The function performs comprehensive validation and provides clear error messages:

```tcl
# Missing required parameter
torch::cumprod -dim 0
# Error: Required parameter missing: -input

# Invalid tensor name
torch::cumprod -input "nonexistent" -dim 0
# Error: Invalid tensor name

# Invalid dimension value
torch::cumprod -input $tensor -dim "invalid"
# Error: Invalid dim value. Expected integer.

# Wrong number of positional arguments
torch::cumprod $tensor
# Error: Wrong number of arguments for positional syntax. Expected: torch::cumprod tensor dim
```

## Technical Notes

### Performance Considerations
- The cumulative product operation has O(n) time complexity for the specified dimension
- Memory usage is the same as the input tensor size
- Consider numerical precision when working with very large products

### Numerical Stability
- Be aware of potential overflow with large integer values
- For floating-point tensors, very large products may result in infinity
- Very small values may underflow to zero

### Dimension Handling
- The `dim` parameter must be within the valid range `[0, tensor.dim()-1]`
- Negative dimension indices are supported following PyTorch conventions
- The output tensor has the same shape as the input tensor

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::cumprod $tensor 0]
```

**New named parameter syntax:**
```tcl
set result [torch::cumprod -input $tensor -dim 0]
```

**Or using camelCase alias:**
```tcl
set result [torch::cumProd -input $tensor -dim 0]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and understand code
- **Error Prevention**: Less likely to mix up parameter positions

## Related Commands

- `torch::cumsum` - Cumulative sum along a dimension
- `torch::cummax` - Cumulative maximum along a dimension  
- `torch::cummin` - Cumulative minimum along a dimension
- `torch::prod` - Product of all elements or along a dimension
- `torch::cumprod` - This command

## See Also

- [PyTorch cumprod documentation](https://pytorch.org/docs/stable/generated/torch.cumprod.html)
- [Tensor Creation Operations](../tensor_creation_ops.md)
- [Mathematical Operations](../mathematical_operations.md) 