# torch::matrix_power

Computes the matrix power of a square matrix.

## Syntax

### Positional Arguments (Backward Compatible)
```tcl
torch::matrix_power input n
```

### Named Parameters (New)
```tcl
torch::matrix_power -input tensor -n integer
```

### camelCase Alias
```tcl
torch::matrixPower -input tensor -n integer
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` | tensor | Input square matrix tensor | Required |
| `n` | integer | Power exponent (can be negative) | Required |

## Description

The `torch::matrix_power` command computes the matrix power of a square matrix. This operation is equivalent to multiplying the matrix by itself `n` times.

For `n = 0`, returns the identity matrix.
For `n = 1`, returns the original matrix.
For `n > 1`, computes the matrix multiplied by itself `n` times.
For `n < 0`, computes the inverse matrix raised to the power of `|n|`.

## Examples

### Basic Usage
```tcl
# Create a 2x2 matrix
set data [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
set A [torch::tensor_reshape $data {2 2}]

# Compute A^2 using positional syntax
set result1 [torch::matrix_power $A 2]

# Compute A^2 using named parameters
set result2 [torch::matrix_power -input $A -n 2]

# Compute A^2 using camelCase alias
set result3 [torch::matrixPower -input $A -n 2]
```

### Special Cases
```tcl
# Identity matrix (A^0 = I)
set identity [torch::matrix_power $A 0]

# Original matrix (A^1 = A)
set original [torch::matrix_power $A 1]

# Matrix inverse (A^-1)
set inverse [torch::matrix_power $A -1]
```

### Batch Operations
```tcl
# Create batch of matrices
set batch_data [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
set batch_matrices [torch::tensor_reshape $batch_data {2 2 2}]

# Compute power for each matrix in the batch
set batch_result [torch::matrix_power $batch_matrices 2]
```

### Mixed Syntax
```tcl
# Positional input with named power parameter
set result [torch::matrix_power $A -n 3]
```

## Return Value

Returns a tensor handle representing the matrix power result. The output tensor has the same shape as the input tensor.

## Error Handling

The command will throw an error if:
- Input tensor is not provided
- Power parameter `n` is not provided
- Input tensor is not a square matrix
- Power parameter `n` is not an integer
- Invalid tensor handle is provided

## Notes

- The input tensor must be a square matrix (same number of rows and columns)
- For negative powers, the matrix must be invertible
- The operation preserves the batch dimension if the input is a batch of matrices
- All three syntaxes (positional, named, camelCase) produce identical results

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::matrix_power $matrix 2

# New named parameter syntax
torch::matrix_power -input $matrix -n 2

# Or using camelCase alias
torch::matrixPower -input $matrix -n 2
```

### Mixed Syntax Support
```tcl
# Mixed syntax is also supported
torch::matrix_power $matrix -n 2
```

## See Also

- [torch::matmul](matmul.md) - Matrix multiplication
- [torch::matrix_norm](matrix_norm.md) - Matrix norms
- [torch::matrix_rank](matrix_rank.md) - Matrix rank
- [torch::linalg_inv](linalg_inv.md) - Matrix inverse 