# torch::sparse_tensor_create / torch::sparseTensorCreate

Creates a sparse COO (Coordinate Format) tensor from indices, values, and size specification.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_tensor_create indices values size
```

### Named Parameter Syntax
```tcl
torch::sparse_tensor_create -indices indices_tensor -values values_tensor -size size_list
torch::sparseTensorCreate -indices indices_tensor -values values_tensor -size size_list  ;# camelCase alias
```

## Parameters

- **indices** (tensor, required): A 2D tensor of type int64 containing the indices of non-zero elements. Each row represents a coordinate in the sparse tensor.
- **values** (tensor, required): A 1D tensor containing the values at the corresponding indices.
- **size** (list, required): A list of integers specifying the dimensions of the sparse tensor.

## Return Value

Returns a handle to the newly created sparse tensor. The tensor is stored in COO format, which is efficient for representing sparse data.

## Examples

```tcl
# Create indices tensor (2D) - each row is a coordinate
set indices [torch::tensor_create {{0 0} {1 1} {2 2}} -dtype "int64"]

# Create values tensor (1D) - values at those coordinates
set values [torch::tensor_create {1.0 2.0 3.0}]

# Create a 3x3 sparse tensor (both syntaxes)
set sparse_tensor1 [torch::sparse_tensor_create $indices $values {3 3}]
set sparse_tensor2 [torch::sparseTensorCreate -indices $indices -values $values -size {3 3}]

# Convert to dense to visualize
# Result will be:
# [[1.0 0.0 0.0]
#  [0.0 2.0 0.0]
#  [0.0 0.0 3.0]]
set dense_tensor [torch::to_dense $sparse_tensor1]
```

## Error Conditions

- Invalid tensor handle: Returns error "Invalid tensor handle"
- Invalid size format: Returns error "expected list but got ..."
- Missing parameters: Returns error "Required parameters missing: ..."
- Missing parameter value: Returns error "Missing value for parameter"
- Unknown parameter: Returns error "Unknown parameter: ..."

## Related Commands

- `torch::sparse_coo_tensor` - Alternative way to create sparse tensors with more options
- `torch::to_dense` - Convert sparse tensor to dense format
- `torch::sparse_add` - Add two sparse tensors
- `torch::sparse_mm` - Sparse matrix multiplication 