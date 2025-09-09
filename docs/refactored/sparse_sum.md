# torch::sparse_sum / torch::sparseSum

Computes the sum of elements in a sparse tensor, either across all dimensions or along a specified dimension.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_sum sparse_tensor ?dim?
```

### Named Parameter Syntax
```tcl
torch::sparse_sum -input sparse_tensor ?-dim dimension?
torch::sparseSum -input sparse_tensor ?-dim dimension?  ;# camelCase alias
```

## Parameters

- **sparse_tensor** (tensor, required): The input sparse tensor to sum.
- **dim** (integer, optional): The dimension along which to compute the sum. If not provided, the sum is computed across all dimensions.

## Return Value

Returns a new tensor containing the sum of elements:
- If `dim` is not specified: Returns a scalar tensor containing the sum of all elements.
- If `dim` is specified: Returns a tensor with the specified dimension reduced by summing.

## Examples

```tcl
# Create a sparse tensor for demonstration
set indices [torch::tensor_create {{0 0} {1 1} {2 2}} -dtype "int64"]
set values [torch::tensor_create {1.0 2.0 3.0}]
set sparse_tensor [torch::sparse_coo_tensor $indices $values {3 3}]

# Sum all elements (both syntaxes)
set total_sum [torch::sparse_sum $sparse_tensor]
set total_sum_named [torch::sparseSum -input $sparse_tensor]

# Sum along dimension 0
set dim0_sum [torch::sparse_sum $sparse_tensor 0]
set dim0_sum_named [torch::sparseSum -input $sparse_tensor -dim 0]

# Sum along dimension 1
set dim1_sum [torch::sparse_sum $sparse_tensor 1]
set dim1_sum_named [torch::sparseSum -input $sparse_tensor -dim 1]
```

## Error Conditions

- Invalid tensor handle: Returns error "Invalid sparse tensor"
- Invalid dimension type: Returns error "expected integer but got ..."
- Too many arguments: Returns error "wrong # args: should be ..."

## Related Commands

- `torch::sparse_add` - Add two sparse tensors
- `torch::sparse_to_dense` - Convert sparse tensor to dense format
- `torch::sparse_coo_tensor` - Create a sparse COO tensor
- `torch::sparse_transpose` - Transpose a sparse tensor 