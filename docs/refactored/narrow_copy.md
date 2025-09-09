# torch::narrow_copy / torch::narrowCopy

Creates a new tensor that is a narrowed copy of the input tensor along a specified dimension.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::narrow_copy input dim start length
```

### Named Parameters (New Syntax)
```tcl
torch::narrow_copy -input tensor -dim dimension -start startIndex -length numElements
```

### CamelCase Alias
```tcl
torch::narrowCopy -input tensor -dim dimension -start startIndex -length numElements
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` / `-input` | tensor | Input tensor to narrow | Required |
| `dim` / `-dim` | int | Dimension along which to narrow | Required |
| `start` / `-start` | int | Starting index on the specified dimension | Required |
| `length` / `-length` | int | Number of elements to select | Required |

## Returns

Returns a new tensor that is a narrowed copy of the input tensor.

## Description

The `torch::narrow_copy` command returns a new tensor that is a narrowed copy of the input tensor along a specified dimension. The narrowed tensor has the same number of dimensions as the input tensor, but the size of the specified dimension is reduced to the specified length.

This is similar to the `torch::narrow` command, but `narrow_copy` returns a copy of the narrowed tensor rather than a view. This means that the returned tensor does not share storage with the original tensor.

## Examples

### Basic Usage

```tcl
# Create a 3x4 tensor
set tensor [torch::tensor_create {0 1 2 3 4 5 6 7 8 9 10 11} float32]
torch::reshape $tensor {3 4}
puts [torch::tensor_to_string $tensor]
# Output:
# tensor([[0, 1, 2, 3],
#         [4, 5, 6, 7],
#         [8, 9, 10, 11]])

# Narrow along dimension 0 (rows) - positional syntax
set narrowed [torch::narrow_copy $tensor 0 1 2]
puts [torch::tensor_to_string $narrowed]
# Output:
# tensor([[4, 5, 6, 7],
#         [8, 9, 10, 11]])

# Narrow along dimension 1 (columns) - named parameter syntax
set narrowed [torch::narrow_copy -input $tensor -dim 1 -start 1 -length 2]
puts [torch::tensor_to_string $narrowed]
# Output:
# tensor([[1, 2],
#         [5, 6],
#         [9, 10]])

# Using camelCase alias
set narrowed [torch::narrowCopy -input $tensor -dim 0 -start 0 -length 2]
puts [torch::tensor_to_string $narrowed]
# Output:
# tensor([[0, 1, 2, 3],
#         [4, 5, 6, 7]])
```

## Error Handling

The command will raise an error in the following cases:
- The input tensor is invalid or does not exist
- The dimension is out of range
- The start index is out of range
- The length is invalid or would exceed the tensor's size
- Required parameters are missing
- Unknown parameters are provided

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old syntax (still supported)
torch::narrow_copy $tensor 0 1 2

# New syntax
torch::narrow_copy -input $tensor -dim 0 -start 1 -length 2

# CamelCase alias
torch::narrowCopy -input $tensor -dim 0 -start 1 -length 2
```

## See Also

- `torch::narrow` - Returns a view of the input tensor that is narrowed along a dimension
- `torch::slice` - Returns a slice of the input tensor along a dimension
- `torch::select` - Returns a slice of the input tensor with a dimension removed
