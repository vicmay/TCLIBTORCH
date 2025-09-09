# torch::sparse_csc_tensor

Creates a sparse tensor in Compressed Sparse Column (CSC) format.

## Syntax

```tcl
# Positional syntax
torch::sparse_csc_tensor ccol_indices row_indices values size ?dtype? ?device? ?requires_grad?

# Named parameter syntax
torch::sparse_csc_tensor -ccol_indices TENSOR -row_indices TENSOR -values TENSOR -size LIST \
                        ?-dtype STRING? ?-device STRING? ?-requires_grad BOOLEAN?

# CamelCase alias
torch::sparseCscTensor ...  ;# Same syntax options as above
```

## Description

Creates a sparse tensor in CSC format using the provided column indices, row indices, and values. The CSC format is particularly efficient for column-wise operations and is commonly used in scientific computing and numerical analysis.

## Parameters

* `ccol_indices` (tensor) - A 1-D tensor containing the compressed column indices
* `row_indices` (tensor) - A 1-D tensor containing the row indices
* `values` (tensor) - A 1-D tensor containing the non-zero values
* `size` (list) - A list of integers specifying the dimensions of the sparse tensor
* `dtype` (string, optional) - The data type of the tensor. Default: "float32"
* `device` (string, optional) - The device to place the tensor on. Default: "cpu"
* `requires_grad` (boolean, optional) - Whether to enable gradient computation. Default: 0

## Return Value

Returns a handle to the newly created sparse tensor.

## Examples

```tcl
# Create test tensors
set ccol_indices [torch::tensor_create -data {0 1 2} -dtype "int64"]
set row_indices [torch::tensor_create -data {0 1 0} -dtype "int64"]
set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]

# Using positional syntax
set sparse1 [torch::sparse_csc_tensor $ccol_indices $row_indices $values {2 2}]

# Using named parameter syntax
set sparse2 [torch::sparse_csc_tensor \
    -ccol_indices $ccol_indices \
    -row_indices $row_indices \
    -values $values \
    -size {2 2} \
    -dtype "float64" \
    -device "cpu" \
    -requires_grad 1]

# Using camelCase alias
set sparse3 [torch::sparseCscTensor $ccol_indices $row_indices $values {2 2}]
```

## Error Conditions

* Invalid tensor handles for `ccol_indices`, `row_indices`, or `values`
* Invalid size format (must be a valid list of integers)
* Invalid dtype value
* Invalid device value
* Invalid requires_grad value (must be 0 or 1)
* Mismatched dimensions between indices and values
* Indices out of bounds for the specified size

## See Also

* [torch::sparse_coo_tensor](sparse_coo_tensor.md) - Create a sparse tensor in COO format
* [torch::sparse_csr_tensor](sparse_csr_tensor.md) - Create a sparse tensor in CSR format
* [torch::sparse_to_dense](sparse_to_dense.md) - Convert a sparse tensor to dense format 