# torch::tensor_expand

Expands a tensor to a larger size by broadcasting.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::tensor_expand -input tensor -sizes shape_list
torch::tensorExpand -input tensor -sizes shape_list
```

### Positional Parameters (Legacy)
```tcl
torch::tensor_expand tensor shape_list
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input`, `-tensor` | string | Input tensor handle | Yes |
| `-sizes`, `-shape` | list | Target shape for expansion | Yes |

## Description

The `tensor_expand` operation expands a tensor to a larger size using broadcasting rules. This operation does not allocate new memory; instead, it returns a new view of the existing tensor with the specified size.

### Broadcasting Rules
- Each dimension is expanded by repeating the tensor along that dimension
- The tensor can only be expanded along dimensions where the original size is 1
- You cannot expand dimensions where the original size is greater than 1

## Examples

### Named Parameter Syntax
```tcl
# Create a simple tensor
set tensor [torch::tensor_create {1.0 2.0} float32 cpu false]

# Expand to broadcast the tensor
set expanded [torch::tensor_expand -input $tensor -sizes {3 2}]

# Alternative parameter names
set expanded2 [torch::tensor_expand -tensor $tensor -shape {3 2}]

# Using camelCase alias
set expanded3 [torch::tensorExpand -input $tensor -sizes {3 2}]
```

### Positional Parameter Syntax (Legacy)
```tcl
# Same operation using legacy syntax
set tensor [torch::tensor_create {1.0 2.0} float32 cpu false]
set expanded [torch::tensor_expand $tensor {3 2}]
```

### Multi-dimensional Expansion
```tcl
# Create a tensor with shape [1, 3]
set data [torch::tensor_create {1.0 2.0 3.0} float32 cpu false]
set reshaped [torch::tensor_reshape $data {1 3}]

# Expand to [2, 3]
set expanded [torch::tensor_expand -input $reshaped -sizes {2 3}]

# The result will have the same values repeated:
# [[1.0, 2.0, 3.0],
#  [1.0, 2.0, 3.0]]
```

### Single Value Broadcasting
```tcl
# Create a scalar-like tensor
set scalar [torch::tensor_create {5.0} float32 cpu false]

# Expand to any shape
set matrix [torch::tensor_expand -input $scalar -sizes {3 4}]
# Results in a 3x4 matrix filled with 5.0
```

## Return Value

Returns a string handle to the new expanded tensor view.

## Error Handling

The command will raise an error in the following cases:

- **Missing parameters**: Both input tensor and sizes must be provided
- **Invalid tensor**: The specified tensor handle does not exist
- **Invalid expansion**: Attempting to expand a dimension that is not size 1
- **Unknown parameters**: Using parameter names not recognized by the command

### Error Examples
```tcl
# Error: Missing required parameters
catch {torch::tensor_expand -sizes {2 2}} error
# Returns: "Required parameters missing: -input and -sizes"

# Error: Invalid tensor handle
catch {torch::tensor_expand -input "bad_tensor" -sizes {2 2}} error
# Returns: "Invalid tensor name"

# Error: Cannot expand dimension > 1
set tensor [torch::tensor_create {1.0 2.0 3.0} float32 cpu false]  # shape [3]
catch {torch::tensor_expand -input $tensor -sizes {2 3}} error
# Returns error from PyTorch about invalid expansion
```

## Memory Efficiency

Important: `tensor_expand` does not copy data. It creates a new view of the existing tensor with expanded dimensions. This makes it very memory efficient for large tensors.

```tcl
# Both tensors share the same underlying data
set original [torch::tensor_create {1.0 2.0} float32 cpu false]
set expanded [torch::tensor_expand -input $original -sizes {10 2}]
# The expanded tensor uses the same memory as the original
```

## Migration Guide

### From Positional to Named Parameters

**Old Syntax:**
```tcl
set result [torch::tensor_expand $tensor {3 4 5}]
```

**New Syntax:**
```tcl
set result [torch::tensor_expand -input $tensor -sizes {3 4 5}]
# or using camelCase
set result [torch::tensorExpand -input $tensor -sizes {3 4 5}]
```

### Parameter Mapping

| Positional Order | Named Parameter | Alternative Names |
|------------------|-----------------|-------------------|
| 1st argument | `-input` | `-tensor` |
| 2nd argument | `-sizes` | `-shape` |

## Compatibility

- ✅ **Backward Compatible**: All existing code using positional syntax continues to work
- ✅ **New Features**: Named parameters provide better readability and maintainability
- ✅ **Flexible**: Multiple parameter name aliases for convenience
- ✅ **Modern**: camelCase alias follows contemporary API design patterns

## See Also

- [`torch::tensor_reshape`](tensor_reshape.md) - Change tensor shape
- [`torch::tensor_repeat`](tensor_repeat.md) - Repeat tensor along dimensions
- [`torch::tensor_view`](tensor_view.md) - Create views of tensors
- [`torch::tensor_squeeze`](tensor_squeeze.md) - Remove singleton dimensions

## Technical Notes

- Uses PyTorch's `tensor.expand()` method internally
- Returns a view, not a copy of the data
- Broadcasting follows standard PyTorch broadcasting semantics
- Compatible with autograd if input tensor requires gradients 