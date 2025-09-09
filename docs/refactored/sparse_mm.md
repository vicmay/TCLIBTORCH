# torch::sparse_mm / torch::sparseMm

Performs matrix multiplication between a sparse tensor and a dense tensor.

## Syntax

### Traditional (Positional Parameters)
```tcl
torch::sparse_mm sparse_tensor dense_tensor
```

### Modern (Named Parameters)
```tcl
torch::sparse_mm -sparse_tensor TENSOR -dense_tensor TENSOR
```

### CamelCase Alias
```tcl
torch::sparseMm -sparseTensor TENSOR -denseTensor TENSOR
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| sparse_tensor | A sparse tensor to use as the first operand in matrix multiplication |
| dense_tensor | A dense tensor to use as the second operand in matrix multiplication |

## Return Value

Returns a new dense tensor containing the result of the matrix multiplication.

## Description

The `torch::sparse_mm` command performs matrix multiplication between a sparse tensor and a dense tensor. The operation is optimized for sparse tensors and follows the standard matrix multiplication rules.

The first tensor must be a sparse tensor, and the second tensor must be a dense tensor. The dimensions of the tensors must be compatible for matrix multiplication.

## Examples

### Using Traditional Syntax
```tcl
# Create a sparse tensor
set sparse [torch::sparse_tensor_create {2 3} {0 1 1 0} {1 2} {1.0 2.0}]
# Create a dense tensor
set dense [torch::tensor_create {{1.0 2.0} {3.0 4.0} {5.0 6.0}}]
# Perform sparse matrix multiplication
set result [torch::sparse_mm $sparse $dense]
```

### Using Modern Syntax
```tcl
set result [torch::sparse_mm \
    -sparse_tensor $sparse \
    -dense_tensor $dense]
```

### Using CamelCase Alias
```tcl
set result [torch::sparseMm \
    -sparseTensor $sparse \
    -denseTensor $dense]
```

## Error Conditions

The command will raise an error if:
- Either tensor handle is invalid
- The first tensor is not a sparse tensor
- The second tensor is not a dense tensor
- The tensor dimensions are incompatible for matrix multiplication
- Memory allocation fails

## See Also

- `torch::sparse_tensor_create` - Create a sparse tensor
- `torch::sparse_tensor_dense` - Convert sparse tensor to dense tensor
- `torch::sparse_mask` - Apply sparse mask to tensor 