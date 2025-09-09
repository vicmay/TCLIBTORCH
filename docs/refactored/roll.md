# torch::roll

Roll the tensor along the specified dimension(s). Elements that are shifted beyond the last position are re-introduced at the first position.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::roll -input tensor -shifts {shift1 ?shift2 ...?} ?-dims {dim1 ?dim2 ...?}?
```

### Positional Syntax (Legacy)
```tcl
torch::roll input shifts ?dims?
```

### CamelCase Alias
```tcl
torch::roll ...  ;# Same syntax options as above
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input`/`-input` | tensor | Yes | - | The input tensor to roll |
| `shifts`/`-shifts` | list of ints | Yes | - | The number of places by which the elements of the tensor are shifted |
| `dims`/`-dims` | list of ints | No | - | The dimensions along which to roll. If not specified, the tensor is flattened before rolling and then restored to the original shape |

## Return Value

Returns a new tensor with the same shape as input, where the values are cyclically shifted.

## Examples

### Basic Roll
```tcl
# Create a tensor
set t [torch::arange 10]  ;# tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

# Roll 2 positions forward (positional syntax)
torch::roll $t {2}  ;# tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

# Roll 2 positions forward (named syntax)
torch::roll -input $t -shifts {2}  ;# tensor([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])
```

### Roll Along Specific Dimension
```tcl
# Create a 2D tensor
set t [torch::reshape [torch::arange 6] {2 3}]
# tensor([[0, 1, 2],
#         [3, 4, 5]])

# Roll along rows (dim 0)
torch::roll -input $t -shifts {1} -dims {0}
# tensor([[3, 4, 5],
#         [0, 1, 2]])

# Roll along columns (dim 1)
torch::roll -input $t -shifts {1} -dims {1}
# tensor([[2, 0, 1],
#         [5, 3, 4]])
```

### Multiple Dimensions
```tcl
# Roll different amounts in different dimensions
torch::roll -input $t -shifts {1 1} -dims {0 1}
# tensor([[5, 3, 4],
#         [2, 0, 1]])
```

## Error Handling

The command will return an error in the following cases:
- Input tensor is invalid or not found
- Shifts list is invalid or empty
- Dimensions list is invalid
- Number of shifts does not match number of dimensions (when dims is provided)
- Dimension values are out of range for the input tensor

## See Also

- `torch::reshape` - Reshape a tensor
- `torch::transpose` - Transpose dimensions of a tensor
- `torch::permute` - Permute dimensions of a tensor 