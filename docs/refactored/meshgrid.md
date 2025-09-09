# torch::meshgrid / torch::meshGrid

Creates coordinate grids from multiple tensors.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::meshgrid tensor1 tensor2 ?tensor3 ...?
```

### Named Parameters (New Syntax)
```tcl
torch::meshgrid -tensors {tensor1 tensor2 ?tensor3 ...?}
torch::meshGrid -tensors {tensor1 tensor2 ?tensor3 ...?}
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor1, tensor2, ...` / `-tensors` | list | Names of input tensors | Required |

## Returns

Returns a list of tensor handles. The number of output tensors equals the number of input tensors. Each output tensor has a shape determined by the sizes of all input tensors.

## Description

The `torch::meshgrid` command takes one or more 1D tensors and creates a grid of coordinates. The resulting tensors can be used to evaluate functions on a grid.

For example, if you have two 1D tensors of sizes (n) and (m), the output will be two 2D tensors of shape (n, m) where:
- The first output tensor has each row filled with the values from the first input tensor
- The second output tensor has each column filled with the values from the second input tensor

## Examples

### Basic Usage

```tcl
# Create two 1D tensors
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set y [torch::tensor_create -data {4.0 5.0} -dtype float32 -device cpu]

# Create coordinate grid (positional syntax)
set grid_list [torch::meshgrid $x $y]
# grid_list contains two tensors:
# - First tensor is [[1, 1], [2, 2], [3, 3]]
# - Second tensor is [[4, 5], [4, 5], [4, 5]]

# Create coordinate grid (named syntax)
set grid_list2 [torch::meshgrid -tensors [list $x $y]]

# Create coordinate grid (camelCase alias)
set grid_list3 [torch::meshGrid -tensors [list $x $y]]
```

### Three-Dimensional Grid

```tcl
# Create three 1D tensors
set x [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
set y [torch::tensor_create -data {3.0 4.0 5.0} -dtype float32 -device cpu]
set z [torch::tensor_create -data {6.0 7.0} -dtype float32 -device cpu]

# Create 3D coordinate grid
set grid_list [torch::meshgrid $x $y $z]
# grid_list contains three tensors of shape (2, 3, 2)
```

## Error Handling

The command will raise an error in the following cases:
- Any of the input tensor names are invalid
- No tensors are provided
- Invalid parameter name is used

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old syntax (still supported)
set grid_list [torch::meshgrid $x $y]

# New syntax
set grid_list [torch::meshgrid -tensors [list $x $y]]
```

## See Also

- `torch::tensor_create` - Create a new tensor
- `torch::cartesian_prod` - Compute the Cartesian product of tensors
