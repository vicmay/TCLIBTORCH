# torch::sparse_mask / torch::sparseMask

Applies a sparse mask to a tensor, zeroing out elements where the mask is zero. The mask must be a sparse tensor with the same shape as the input tensor.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_mask tensor mask
```

### Named Parameter Syntax
```tcl
torch::sparse_mask -tensor tensor -mask mask
```

### CamelCase Alias
```tcl
torch::sparseMask -tensor tensor -mask mask
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor | tensor | The input tensor to be masked |
| mask | tensor | The sparse tensor to use as a mask |

## Return Value

Returns a new sparse tensor with the same shape as the input tensor, where elements are zeroed out where the mask is zero.

## Examples

### Using Positional Syntax
```tcl
# Create a sparse tensor
set indices [torch::tensor_create {0 1 2 0 1 2} int64 cpu]
set indices [torch::tensor_reshape -input $indices -shape {2 3}]
set values [torch::tensor_create {1.0 2.0 3.0} float32 cpu]
set tensor [torch::sparse_coo_tensor $indices $values {3 3}]

# Create a mask tensor
set mask_indices [torch::tensor_create {0 1} int64 cpu]
set mask_indices [torch::tensor_reshape -input $mask_indices -shape {2 1}]
set mask_values [torch::tensor_create {1.0} float32 cpu]
set mask [torch::sparse_coo_tensor $mask_indices $mask_values {3 3}]

# Apply mask
set result [torch::sparse_mask $tensor $mask]
```

### Using Named Parameter Syntax
```tcl
set result [torch::sparse_mask -tensor $tensor -mask $mask]
```

### Using CamelCase Alias
```tcl
set result [torch::sparseMask -tensor $tensor -mask $mask]
```

## Error Handling

The command will return an error in the following cases:
- If either the tensor or mask parameter is missing
- If either the tensor or mask parameter is an invalid tensor handle
- If an unknown parameter is provided
- If the mask tensor has a different shape than the input tensor

## See Also

- [torch::sparse_coo_tensor](sparse_coo_tensor.md) - Create a sparse COO tensor
- [torch::sparse_csr_tensor](sparse_csr_tensor.md) - Create a sparse CSR tensor
- [torch::sparse_csc_tensor](sparse_csc_tensor.md) - Create a sparse CSC tensor 