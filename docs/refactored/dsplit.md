# torch::dsplit

Splits a tensor along the depth dimension (dimension 2). This operation divides the input tensor into multiple smaller tensors along the third dimension (depth).

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::dsplit tensor sections_or_indices
```

### Named Parameter Syntax (Recommended)
```tcl
torch::dsplit -tensor tensor -sections sections
torch::dsplit -tensor tensor -indices indices_list
torch::dsplit -input tensor -sections sections
torch::dsplit -input tensor -indices indices_list
```

### CamelCase Alias
```tcl
torch::dSplit -tensor tensor -sections sections
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| tensor/-tensor/-input | string | Handle to input tensor (minimum 3D) | Yes |
| sections_or_indices/-sections/-indices | int/list | Number of sections or list of indices | Yes |

## Description

The `torch::dsplit` command splits a tensor along dimension 2 (depth dimension). This is equivalent to calling `torch::tensor_split` with `dim=2`. The input tensor must have at least 3 dimensions.

**Splitting modes:**
- **Sections**: Integer number - splits into equal-sized sections
- **Indices**: List of integers - splits at specified indices

## Mathematical Details

For a tensor with shape `[N, H, W, D, ...]`:
- Splits along dimension 2 (depth/width dimension)
- Returns multiple tensors with shapes `[N, H, W_i, D, ...]`
- Each output tensor maintains all other dimensions

## Examples

### Basic Usage - Sections

```tcl
# Create a 3D tensor (2x3x6)
set tensor [torch::zeros -size {2 3 6}]

# Positional syntax - split into 3 sections
set result [torch::dsplit $tensor 3]
# Returns list of 3 tensors, each with shape [2, 3, 2]

# Named parameter syntax - split into 2 sections  
set result [torch::dsplit -tensor $tensor -sections 2]
# Returns list of 2 tensors, each with shape [2, 3, 3]

# Using input alias
set result [torch::dsplit -input $tensor -sections 3]
```

### Advanced Usage - Indices

```tcl
# Create a 4D tensor (2x4x8x3)
set tensor [torch::ones -size {2 4 8 3}]

# Split at specific indices
set result [torch::dsplit -tensor $tensor -indices {2 5 7}]
# Returns 4 tensors with shapes:
# [2, 4, 2, 3], [2, 4, 3, 3], [2, 4, 2, 3], [2, 4, 1, 3]

# Named parameter with indices
set result [torch::dsplit -input $tensor -indices {3 6}]
# Returns 3 tensors with shapes:
# [2, 4, 3, 3], [2, 4, 3, 3], [2, 4, 2, 3]
```

### CamelCase Syntax

```tcl
# Using camelCase alias
set tensor [torch::randn -size {3 5 10}]
set result [torch::dSplit -tensor $tensor -sections 5]
# Returns 5 tensors, each with shape [3, 5, 2]
```

## Applications

### Computer Vision
```tcl
# Split RGB channels along width (assuming HWC format)
set image [torch::randn -size {224 224 3}]
set channels [torch::dsplit -tensor $image -sections 3]
# Splits into separate color components
```

### 3D Data Processing
```tcl
# Split 3D volume along depth
set volume [torch::zeros -size {64 64 32}]
set slices [torch::dsplit -tensor $volume -indices {8 16 24}]
# Creates depth slices for processing
```

### Batch Processing
```tcl
# Split batched 3D data
set batch [torch::randn -size {16 28 28 10}]
set splits [torch::dsplit -input $batch -sections 2]
# Splits along depth for parallel processing
```

## Error Handling

```tcl
# Missing tensor parameter
catch {torch::dsplit -sections 3} error
puts "Error: $error"
# Error: Missing required parameter: tensor

# Invalid tensor dimensions (less than 3D)
set tensor_2d [torch::zeros -size {3 4}]
catch {torch::dsplit -tensor $tensor_2d -sections 2} error
puts "Error: $error"
# Error: Input tensor must have at least 3 dimensions

# Invalid section number
set tensor [torch::zeros -size {2 3 6}]
catch {torch::dsplit -tensor $tensor -sections 0} error
puts "Error: $error"
# Error: Number of sections must be positive

# Out of bounds indices
catch {torch::dsplit -tensor $tensor -indices {10}} error
puts "Error: $error"
# Error: Split indices out of bounds
```

## Performance Considerations

- **Memory Efficient**: Returns views when possible, avoiding data copying
- **Dimension Requirements**: Input must be at least 3-dimensional
- **Index Validation**: Indices are validated to be within tensor bounds
- **Equal Sections**: When using sections, tensor size along dim 2 should be divisible

## Notes

1. **Dimension Order**: Operates on dimension 2 (zero-indexed), typically the depth/width dimension
2. **View vs Copy**: Returns views of the original tensor when possible for efficiency
3. **Index Sorting**: Indices are processed in ascending order automatically
4. **Boundary Handling**: Split points create non-overlapping segments

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::dsplit $tensor 3]

# New named parameter syntax
set result [torch::dsplit -tensor $tensor -sections 3]

# Alternative with input alias
set result [torch::dsplit -input $tensor -sections 3]
```

### Parameter Aliases

- `-tensor` and `-input` are interchangeable for the input tensor
- `-sections` for equal-sized splits
- `-indices` for custom split points

## Return Value

Returns a Tcl list of tensor handles, each representing a split portion of the input tensor along dimension 2.

## See Also

- `torch::hsplit` - Split along dimension 1 (height)
- `torch::vsplit` - Split along dimension 0 (vertical)
- `torch::tensor_split` - General tensor splitting
- `torch::chunk` - Split into equal chunks
- `torch::split` - Split with specific sizes 