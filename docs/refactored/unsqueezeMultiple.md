# torch::unsqueezeMultiple

## Overview

The `torch::unsqueezeMultiple` command adds multiple singleton dimensions to a tensor at specified positions. This is useful for broadcasting operations and reshaping tensors for neural network layers.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::unsqueeze_multiple tensor dims
```

### Named Parameter Syntax (New)
```tcl
torch::unsqueeze_multiple -tensor TENSOR -dims DIMS
```

### CamelCase Alias
```tcl
torch::unsqueezeMultiple tensor dims
torch::unsqueezeMultiple -tensor TENSOR -dims DIMS
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-tensor` | string | Yes | Name of the input tensor |
| `dims` / `-dims` | list | Yes | List of dimensions where to add singleton dimensions |

## Return Value

Returns the name of the resulting tensor with added singleton dimensions.

## Examples

### Basic Usage

```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu true]

# Add singleton dimensions at positions 0 and 2
set result [torch::unsqueeze_multiple $t {0 2}]
puts [torch::tensor_shape $result]
# Output: 1 2 3 1

# Using named parameters
set result2 [torch::unsqueeze_multiple -tensor $t -dims {0 2}]
puts [torch::tensor_shape $result2]
# Output: 1 2 3 1

# Using camelCase alias
set result3 [torch::unsqueezeMultiple $t {0 2}]
puts [torch::tensor_shape $result3]
# Output: 1 2 3 1
```

### Single Dimension

```tcl
set t [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu true]

# Add singleton dimension at position 1
set result [torch::unsqueeze_multiple $t {1}]
puts [torch::tensor_shape $result]
# Output: 2 1 3
```

### Negative Dimensions

```tcl
set t [torch::tensor_create {{1 2 3} {4 5 6}} float32 cpu true]

# Use negative indexing
set result [torch::unsqueeze_multiple $t {-1 -3}]
puts [torch::tensor_shape $result]
# Output: 2 1 3 1
```

### Broadcasting Preparation

```tcl
# Create tensors for broadcasting
set a [torch::tensor_create {1 2 3} float32 cpu true]
set b [torch::tensor_create {{10} {20} {30}} float32 cpu true]

# Add dimensions to make them compatible for broadcasting
set a_expanded [torch::unsqueeze_multiple $a {0}]
set b_expanded [torch::unsqueeze_multiple $b {2}]

# Now they can be added together
set result [torch::tensor_add $a_expanded $b_expanded]
puts [torch::tensor_shape $result]
# Output: 1 3 1
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
torch::unsqueeze_multiple tensor_name {0 1 2}
```

**New (Named Parameters):**
```tcl
torch::unsqueeze_multiple -tensor tensor_name -dims {0 1 2}
```

**New (CamelCase Alias):**
```tcl
torch::unsqueezeMultiple tensor_name {0 1 2}
torch::unsqueezeMultiple -tensor tensor_name -dims {0 1 2}
```

### Benefits of New Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Backward Compatibility**: Old syntax continues to work

## Error Handling

The command provides clear error messages for various error conditions:

- **Missing parameters**: "Required parameters missing: tensor, dims"
- **Invalid tensor name**: "Invalid tensor"
- **Empty dimensions list**: Error for empty dims parameter
- **Out of bounds dimensions**: Error for dimensions beyond tensor rank
- **Unknown named parameters**: Error for unrecognized parameter names

## Technical Details

### Dimension Ordering

The command sorts dimensions in descending order before applying unsqueeze operations to avoid index shifting issues. This ensures consistent behavior regardless of the order in which dimensions are specified.

### Memory Management

The resulting tensor is automatically managed by the tensor storage system. The original tensor remains unchanged.

### Performance Considerations

- Multiple unsqueeze operations are applied sequentially
- The operation is memory-efficient as it only adds singleton dimensions
- No data copying occurs during the operation

## Related Commands

- `torch::unsqueeze` - Add a single singleton dimension
- `torch::squeeze` - Remove singleton dimensions
- `torch::squeezeMultiple` - Remove multiple singleton dimensions
- `torch::reshape` - Reshape tensor to different dimensions
- `torch::view` - Create a new view of the tensor

## Notes

- The command preserves the data type and device of the input tensor
- Singleton dimensions are added at the specified positions
- The operation is reversible using `torch::squeezeMultiple`
- Both positive and negative dimension indices are supported 