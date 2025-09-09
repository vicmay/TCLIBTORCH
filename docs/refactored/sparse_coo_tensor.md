# torch::sparse_coo_tensor / torch::sparseCooTensor

Creates a sparse COO (Coordinate) format tensor.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::sparse_coo_tensor indices values size ?dtype? ?device? ?requires_grad?

# Named parameter syntax
torch::sparse_coo_tensor -indices indices -values values -size size ?-dtype dtype? ?-device device? ?-requires_grad requires_grad?

# CamelCase alias
torch::sparseCooTensor -indices indices -values values -size size ?-dtype dtype? ?-device device? ?-requires_grad requires_grad?
```

## Parameters

* `indices` (tensor): A 2D tensor of indices with shape `[sparse_dims, num_points]`. Each column specifies the coordinates of a non-zero value.
* `values` (tensor): A 1D tensor containing the values at the coordinates specified by `indices`.
* `size` (list): The size of the resulting tensor as a list of dimensions.
* `dtype` (string, optional): The desired data type for the tensor. Defaults to `float32`.
* `device` (string, optional): The device to place the tensor on (`cpu` or `cuda`). Defaults to `cpu`.
* `requires_grad` (boolean, optional): Whether to enable gradient computation. Defaults to `0`.

## Return Value

Returns a handle to the newly created sparse tensor.

## Examples

```tcl
# Create indices tensor (2x3 matrix)
set indices [torch::tensor_create {{0 1 1} {1 0 2}} int64]

# Create values tensor (3 values)
set values [torch::tensor_create {1.0 2.0 3.0} float32]

# Using positional syntax
set sparse1 [torch::sparse_coo_tensor $indices $values {2 3}]

# Using named parameter syntax
set sparse2 [torch::sparse_coo_tensor \
    -indices $indices \
    -values $values \
    -size {2 3} \
    -dtype float32 \
    -device cpu \
    -requires_grad 1]

# Using camelCase alias
set sparse3 [torch::sparseCooTensor \
    -indices $indices \
    -values $values \
    -size {2 3}]
```

## Error Conditions

* Returns an error if `indices` or `values` tensors are invalid or not found.
* Returns an error if the number of arguments is incorrect.
* Returns an error if `dtype` or `device` values are invalid.
* Returns an error if `requires_grad` is not a valid boolean value.

## See Also

* `torch::sparse_csr_tensor` - Creates a sparse CSR format tensor
* `torch::sparse_csc_tensor` - Creates a sparse CSC format tensor
* `torch::sparse_to_dense` - Converts a sparse tensor to dense format 