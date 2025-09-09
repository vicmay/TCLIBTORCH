# torch::scatter_nd / torch::scatterNd

Performs N-dimensional scatter operation on a tensor.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::scatter_nd input indices updates
```

### Named Parameter Syntax
```tcl
torch::scatter_nd -input tensor -indices tensor -updates tensor
torch::scatterNd -input tensor -indices tensor -updates tensor  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| input | tensor | The source tensor to scatter values into |
| indices | tensor | The indices where elements in updates will be scattered to |
| updates | tensor | The values to scatter into input |

## Return Value

Returns a new tensor with the scattered values.

## Description

The `scatter_nd` operation scatters values from the `updates` tensor into a new tensor based on the indices specified in the `indices` tensor. The operation is performed along dimension 0.

- The input tensor is not modified; a new tensor is returned
- The indices tensor must be of type int64
- The updates tensor must have compatible shape with the indices tensor

## Examples

### Using Positional Syntax
```tcl
set input [torch::tensor_create {1 2 3 4 5 6 7 8} float32]
set indices [torch::tensor_create {0 4 2 1} int64]
set updates [torch::tensor_create {10 40 30 20} float32]
set result [torch::scatter_nd $input $indices $updates]
;# Result: tensor([10, 20, 30, 4, 40, 6, 7, 8])
```

### Using Named Parameters
```tcl
set input [torch::tensor_create {1 2 3 4 5 6 7 8} float32]
set indices [torch::tensor_create {0 4 2 1} int64]
set updates [torch::tensor_create {10 40 30 20} float32]
set result [torch::scatter_nd -input $input -indices $indices -updates $updates]
;# Result: tensor([10, 20, 30, 4, 40, 6, 7, 8])
```

### Using camelCase Alias
```tcl
set result [torch::scatterNd -input $input -indices $indices -updates $updates]
```

## Error Conditions

- Returns error if any required parameter is missing
- Returns error if any tensor handle is invalid
- Returns error if indices tensor is not of type int64
- Returns error if tensor shapes are incompatible

## Migration Guide

To migrate from the old positional syntax to the new named parameter syntax:

```tcl
# Old syntax
torch::scatter_nd $input $indices $updates

# New syntax
torch::scatterNd -input $input -indices $indices -updates $updates
```

The new syntax is more explicit and self-documenting, making code easier to read and maintain. 