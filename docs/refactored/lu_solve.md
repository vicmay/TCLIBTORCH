# torch::lu_solve

Solves a system of linear equations given LU decomposition results.

## Syntax

### Current Syntax
```tcl
torch::lu_solve B LU_data LU_pivots
```

### Named Parameter Syntax  
```tcl
torch::lu_solve -B tensor -LU_data tensor -LU_pivots tensor
```

### camelCase Alias
```tcl
torch::luSolve -B tensor -LU_data tensor -LU_pivots tensor
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-B` (required): Right-hand side tensor (the B matrix in AX = B)
- `-LU_data` (required): LU decomposition matrix data
- `-LU_pivots` (required): Pivot indices from LU decomposition

### Alternative Parameter Names
- `-b` (alternative to `-B`): Right-hand side tensor
- `-luData` (alternative to `-LU_data`): LU decomposition matrix data
- `-luPivots` (alternative to `-LU_pivots`): Pivot indices

### Positional Parameters
1. `B` (required): Right-hand side tensor name
2. `LU_data` (required): LU decomposition matrix data tensor name
3. `LU_pivots` (required): Pivot indices tensor name

## Description

The `torch::lu_solve` function solves a system of linear equations `AX = B` using the LU decomposition of matrix `A`. This function requires the LU decomposition data and pivot indices to be provided as inputs, typically obtained from a previous LU decomposition operation.

The function efficiently solves the system using forward and backward substitution on the LU decomposition, avoiding the need to invert the matrix directly.

## Mathematical Details

For a system `AX = B` where:
- `A` is an (n × n) coefficient matrix (represented by its LU decomposition)
- `B` is an (n × k) right-hand side matrix
- `X` is the (n × k) solution matrix

Given the LU decomposition `A = PLU` where:
- `P` is a permutation matrix (represented by pivot indices)
- `L` is a lower triangular matrix with unit diagonal
- `U` is an upper triangular matrix

The solution process involves:
1. Applying permutations based on pivot indices
2. Forward substitution to solve `Ly = Pb`
3. Backward substitution to solve `Ux = y`

## Examples

### Basic Usage
```tcl
# Create right-hand side vector
set B [torch::tensor_create -data {1.0 2.0} -shape {2 1} -dtype float32]

# Assume we have LU decomposition data from a previous operation
set LU_data [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]

# Solve using positional syntax
set solution [torch::lu_solve $B $LU_data $LU_pivots]

# Solve using named parameters
set solution [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]

# Solve using camelCase alias
set solution [torch::luSolve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
```

### Multiple Right-hand Sides
```tcl
# Create matrix with multiple right-hand sides
set B [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set LU_data [torch::tensor_create -data {2.0 1.0 1.0 2.0} -shape {2 2} -dtype float32]
set LU_pivots [torch::tensor_create -data {1 2} -shape {2} -dtype int32]

# Solve multiple systems simultaneously
set solutions [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
```

### Parameter Order Independence
```tcl
# Named parameters can be specified in any order
set solution1 [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
set solution2 [torch::lu_solve -LU_pivots $LU_pivots -B $B -LU_data $LU_data]
# Both produce identical results
```

### Alternative Parameter Names
```tcl
# Using alternative camelCase parameter names
set solution [torch::lu_solve -b $B -luData $LU_data -luPivots $LU_pivots]
```

## Data Types

The function supports various data types:
- `float32` - Single precision floating point
- `float64` - Double precision floating point
- `int32` - 32-bit integers
- `int64` - 64-bit integers

All input tensors should have compatible data types.

## Error Handling

The function provides clear error messages for common issues:

```tcl
# Missing required parameters
catch {torch::lu_solve -B $B -LU_data $LU_data} result
# Error: "Required parameters missing"

# Invalid tensor names
catch {torch::lu_solve -B $B -LU_data invalid_tensor -LU_pivots $LU_pivots} result
# Error: "Invalid tensor name"

# Incompatible tensor shapes
catch {torch::lu_solve -B $wrong_shape_B -LU_data $LU_data -LU_pivots $LU_pivots} result
# Error: Shape mismatch information
```

## Performance Notes

- The function uses optimized LAPACK routines for numerical stability
- For large systems, ensure sufficient memory is available
- The function is most efficient when the LU decomposition has already been computed
- Batch operations (multiple right-hand sides) are more efficient than individual solves

## See Also

- `torch::linalg_solve` - General linear system solver
- `torch::cholesky_solve` - Solve using Cholesky decomposition
- `torch::solve_triangular` - Solve triangular systems
- `torch::lstsq` - Least squares solution

## Migration Guide

When migrating from positional to named parameter syntax:

```tcl
# Old positional syntax
set result [torch::lu_solve $B $LU_data $LU_pivots]

# New named parameter syntax
set result [torch::lu_solve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]

# Or using camelCase alias
set result [torch::luSolve -B $B -LU_data $LU_data -LU_pivots $LU_pivots]
```

The positional syntax remains fully supported for backward compatibility.

## Notes

- This function wraps PyTorch's `torch.lu_solve` function
- PyTorch recommends using `torch.linalg.lu_solve` in newer versions
- The function maintains numerical stability through careful implementation
- All input tensors must be on the same device (CPU or CUDA)
- The function supports both real and complex number types where applicable 