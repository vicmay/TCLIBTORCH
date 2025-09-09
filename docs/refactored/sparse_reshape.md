# torch::sparse_reshape

Reshapes a sparse tensor to a new shape while preserving the number of elements and sparsity pattern.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_reshape sparse_tensor shape
```

### Named Parameter Syntax
```tcl
torch::sparse_reshape -input sparse_tensor -shape shape
```

### CamelCase Alias
```tcl
torch::sparseReshape -input sparse_tensor -shape shape
```

## Parameters

- `sparse_tensor` (tensor): The input sparse tensor to reshape
- `shape` (list): The new shape as a list of integers. The product of dimensions must match the original tensor's total number of elements.

## Return Value

Returns a new sparse tensor with the specified shape.

## Description

The `sparse_reshape` command reshapes a sparse tensor to a new shape while preserving its values and sparsity pattern. The total number of elements (product of dimensions) must remain the same.

Important constraints:
1. The number of dimensions must be sparse_dim + dense_dim
2. Cannot shrink sparse dimensions on a non-empty sparse tensor
3. All dimensions must be positive integers

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create a 2x2 sparse tensor
set indices [torch::tensor_create {
    {0 1}
    {0 1}
} -dtype "int64"]
set values [torch::tensor_create {1.0 2.0}]
set sparse [torch::sparse_coo_tensor $indices $values {2 2}]

# Reshape to 1x4
set reshaped [torch::sparse_reshape $sparse {1 4}]
```

### Using Named Parameters
```tcl
# Create a 2x2 sparse tensor
set indices [torch::tensor_create {
    {0 1}
    {0 1}
} -dtype "int64"]
set values [torch::tensor_create {1.0 2.0}]
set sparse [torch::sparse_coo_tensor $indices $values {2 2}]

# Reshape to 4x1 using named parameters
set reshaped [torch::sparse_reshape -input $sparse -shape {4 1}]
```

### Using CamelCase Alias
```tcl
# Reshape using camelCase alias
set reshaped [torch::sparseReshape -input $sparse -shape {1 4}]
```

## Error Cases

The command will return an error in the following cases:

1. Invalid tensor:
```tcl
torch::sparse_reshape invalid_tensor {1 4}
# Error: Invalid sparse tensor
```

2. Missing shape parameter:
```tcl
torch::sparse_reshape -input $sparse
# Error: Missing value for parameter
```

3. Invalid shape list:
```tcl
torch::sparse_reshape -input $sparse -shape {1 -1}
# Error: Invalid integer in shape list
```

4. Invalid parameter:
```tcl
torch::sparse_reshape -input $sparse -invalid value
# Error: Unknown parameter: -invalid
```

## See Also

- [torch::sparse_coo_tensor](sparse_coo_tensor.md) - Create a sparse COO tensor
- [torch::tensor_reshape](tensor_reshape.md) - Reshape a dense tensor 