# torch::matrix_rank

Computes the numerical rank of a matrix or batch of matrices.

## Syntax

### Positional Arguments (Backward Compatible)
```tcl
torch::matrix_rank input ?tol? ?hermitian?
```

### Named Parameters (New)
```tcl
torch::matrix_rank -input tensor ?-tol double? ?-hermitian bool?
```

### camelCase Alias
```tcl
torch::matrixRank -input tensor ?-tol double? ?-hermitian bool?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `input` | tensor | Input matrix or batch of matrices | Required |
| `tol` (or `tolerance`) | double | Tolerance for determining rank | 1e-12 |
| `hermitian` | bool | Whether to treat matrix as Hermitian (0/1) | false (0) |

## Description

The `torch::matrix_rank` command computes the numerical rank of a matrix using singular value decomposition (SVD). The rank is determined by counting the number of singular values that are greater than the specified tolerance.

For Hermitian matrices (complex matrices that are equal to their conjugate transpose), setting the `hermitian` parameter to true can improve computational efficiency and numerical stability.

## Examples

### Basic Usage
```tcl
# Create a 3x3 identity matrix
set A [torch::eye 3 3 float32 cpu false]

# Compute rank using positional syntax
set rank1 [torch::matrix_rank $A]

# Compute rank using named parameters
set rank2 [torch::matrix_rank -input $A]

# Compute rank using camelCase alias
set rank3 [torch::matrixRank -input $A]
```

### With Tolerance Parameter
```tcl
# Create a matrix
set A [torch::eye 3 3 float32 cpu false]

# Compute rank with custom tolerance (positional)
set rank1 [torch::matrix_rank $A 1e-6]

# Compute rank with custom tolerance (named)
set rank2 [torch::matrix_rank -input $A -tol 1e-6]

# Using tolerance alias
set rank3 [torch::matrix_rank -input $A -tolerance 1e-6]
```

### With Hermitian Flag
```tcl
# Create a Hermitian matrix
set A [torch::eye 3 3 float32 cpu false]

# Compute rank with Hermitian flag (positional)
set rank1 [torch::matrix_rank $A 1e-12 1]

# Compute rank with Hermitian flag (named)
set rank2 [torch::matrix_rank -input $A -hermitian 1]

# Mixed syntax
set rank3 [torch::matrix_rank $A -hermitian 1]
```

### Rank-Deficient Matrices
```tcl
# Create a singular (rank-deficient) matrix
set data [torch::tensor_create {1.0 2.0 3.0 2.0 4.0 6.0 3.0 6.0 9.0} float32 cpu false]
set A [torch::tensor_reshape $data {3 3}]

# Compute rank (should be less than 3)
set rank [torch::matrix_rank $A]
```

### Batch Processing
```tcl
# Create batch of matrices
set data [torch::tensor_create {1.0 0.0 0.0 1.0 2.0 0.0 0.0 2.0} float32 cpu false]
set batch_matrices [torch::tensor_reshape $data {2 2 2}]

# Compute rank for each matrix in the batch
set ranks [torch::matrix_rank $batch_matrices]
```

### Mixed Syntax Examples
```tcl
# Positional input with named parameters
set rank1 [torch::matrix_rank $A -tol 1e-6]
set rank2 [torch::matrix_rank $A -hermitian 1]
set rank3 [torch::matrix_rank $A -tol 1e-6 -hermitian 1]
```

## Return Value

Returns a tensor handle representing the rank(s) of the input matrix/matrices. For a single matrix, returns a scalar tensor. For a batch of matrices, returns a tensor with the same batch dimensions containing the rank of each matrix.

## Error Handling

The command will throw an error if:
- Input tensor is not provided
- Invalid tensor handle is provided
- Tolerance parameter is not a valid number
- Hermitian parameter is not a boolean value (0 or 1)
- Unknown parameter is provided
- Parameter value is missing

## Notes

- The numerical rank is computed using SVD decomposition
- Singular values smaller than the tolerance are considered zero
- For Hermitian matrices, set the `hermitian` flag to true for better performance
- The tolerance should be chosen based on the expected numerical precision of your data
- Works with both square and rectangular matrices
- Supports batched operations for multiple matrices

## Mathematical Background

The matrix rank is computed as:
```
rank(A) = #{σᵢ : σᵢ > tol}
```
where σᵢ are the singular values of matrix A, and #{} denotes the count of elements satisfying the condition.

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::matrix_rank $matrix
torch::matrix_rank $matrix 1e-6
torch::matrix_rank $matrix 1e-6 1

# New named parameter syntax
torch::matrix_rank -input $matrix
torch::matrix_rank -input $matrix -tol 1e-6
torch::matrix_rank -input $matrix -tol 1e-6 -hermitian 1

# Or using camelCase alias
torch::matrixRank -input $matrix
torch::matrixRank -input $matrix -tol 1e-6
torch::matrixRank -input $matrix -tol 1e-6 -hermitian 1
```

### Mixed Syntax Support
```tcl
# Mixed syntax is also supported
torch::matrix_rank $matrix -tol 1e-6
torch::matrix_rank $matrix -hermitian 1
torch::matrix_rank $matrix -tol 1e-6 -hermitian 1
```

## See Also

- [torch::matrix_power](matrix_power.md) - Matrix power computation
- [torch::matrix_norm](matrix_norm.md) - Matrix norms
- [torch::linalg_cond](linalg_cond.md) - Condition number
- [torch::linalg_det](linalg_det.md) - Matrix determinant
- [torch::linalg_pinv](linalg_pinv.md) - Pseudo-inverse 