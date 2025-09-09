# torch::sparse_csr_tensor

Creates a sparse tensor in Compressed Sparse Row (CSR) format.

## Syntax

```tcl
# Positional syntax
torch::sparse_csr_tensor crow_indices col_indices values size ?dtype? ?device? ?requires_grad?

# Named parameter syntax
torch::sparse_csr_tensor -crow_indices TENSOR -col_indices TENSOR -values TENSOR -size LIST \
                        ?-dtype STRING? ?-device STRING? ?-requires_grad BOOLEAN?

# CamelCase alias
torch::sparseCsrTensor ...  ;# Same syntax options as above
```

## Description

Creates a sparse tensor in CSR format using the provided row indices, column indices, and values. The CSR format is particularly efficient for row-wise operations and is commonly used in scientific computing and numerical analysis.

## Parameters

* `crow_indices` (tensor) - A 1-D tensor containing the compressed row indices. For a matrix with `m` rows, this tensor has length `m + 1`. Each element represents the starting position of a row in the `col_indices` and `values` arrays.
* `col_indices` (tensor) - A 1-D tensor containing the column indices for each non-zero element.
* `values` (tensor) - A 1-D tensor containing the values of the non-zero elements.
* `size` (list) - A list of integers specifying the dimensions of the sparse tensor.
* `dtype` (string, optional) - The data type for the tensor. Default: "float32". Valid values:
  * "uint8", "int8", "int16", "int32", "int64"
  * "float16", "float32", "float64"
  * "complex32", "complex64", "complex128"
  * "bool"
* `device` (string, optional) - The device to store the tensor on. Default: "cpu". Valid values:
  * "cpu" - Store on CPU memory
  * "cuda" - Store on GPU memory (if CUDA is available)
  * "cuda:N" - Store on specific GPU device N
* `requires_grad` (boolean, optional) - Whether to enable gradient computation. Default: 0 (false)

## Return Value

Returns a handle to the newly created sparse tensor.

## Examples

```tcl
# Create test tensors
set crow_indices [torch::tensor_create -data {0 2 3} -dtype "int64"]
set col_indices [torch::tensor_create -data {1 2 0} -dtype "int64"]
set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]

# Create sparse tensor using positional syntax
set sparse1 [torch::sparse_csr_tensor $crow_indices $col_indices $values {2 3}]

# Create sparse tensor using named parameter syntax
set sparse2 [torch::sparse_csr_tensor \
    -crow_indices $crow_indices \
    -col_indices $col_indices \
    -values $values \
    -size {2 3} \
    -dtype "float64" \
    -device "cpu" \
    -requires_grad 1]

# Using camelCase alias
set sparse3 [torch::sparseCsrTensor $crow_indices $col_indices $values {2 3}]
```

## Error Conditions

* Invalid crow_indices tensor - The provided crow_indices tensor handle doesn't exist
* Invalid col_indices tensor - The provided col_indices tensor handle doesn't exist
* Invalid values tensor - The provided values tensor handle doesn't exist
* Invalid dtype - The specified dtype is not supported
* Invalid device - The specified device is not available
* Invalid requires_grad value - The requires_grad value must be a boolean (0 or 1)
* Missing required parameters - One or more required parameters are missing
* Invalid size list - The size list must contain valid integer dimensions
* Shape mismatch - The shapes of crow_indices, col_indices, and values tensors are incompatible

## See Also

* [torch::sparse_coo_tensor](sparse_coo_tensor.md) - Create sparse tensor in COO format
* [torch::sparse_csc_tensor](sparse_csc_tensor.md) - Create sparse tensor in CSC format
* [torch::sparse_to_dense](sparse_to_dense.md) - Convert sparse tensor to dense format 