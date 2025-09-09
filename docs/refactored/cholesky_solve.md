# torch::cholesky_solve

## Overview

Solves linear systems using Cholesky decomposition. Given a Cholesky decomposition L of a positive definite matrix A (where A = L * L^T for lower triangular L or A = U^T * U for upper triangular U), this function solves the system A * X = B for X.

**Status**: ✅ **REFACTORED** - Supports both snake_case and camelCase syntax with named parameters

## Syntax

### Current Syntax (Recommended)
```tcl
# Named parameters (recommended)
torch::cholesky_solve -b tensor_handle -l tensor_handle -upper bool
torch::choleskySolve -b tensor_handle -l tensor_handle -upper bool

# Alternative parameter names (case variations)
torch::cholesky_solve -B tensor_handle -L tensor_handle -upper bool
```

### Legacy Syntax (Backward Compatible)
```tcl
# Positional parameters (still supported)
torch::cholesky_solve tensor_B tensor_L ?upper?
torch::choleskySolve tensor_B tensor_L ?upper?
```

## Parameters

### Named Parameters
- **`-b tensor_handle`** (required): Right-hand side tensor B
  - Alternative: **`-B tensor_handle`** (uppercase)
  - Must be a valid tensor handle
  - Shape: (..., N, K) where N matches the size of the Cholesky factor
  
- **`-l tensor_handle`** (required): Cholesky factor tensor L 
  - Alternative: **`-L tensor_handle`** (uppercase)
  - Must be a valid tensor handle containing the Cholesky decomposition
  - Shape: (..., N, N) - square matrix
  
- **`-upper bool`** (optional): Whether L is upper triangular
  - Type: Boolean (0/1)
  - Default: false (0) - assumes L is lower triangular
  - true (1): L is upper triangular (U)
  - false (0): L is lower triangular

### Legacy Positional Parameters
1. **`tensor_B`**: Right-hand side tensor B
2. **`tensor_L`**: Cholesky factor tensor L
3. **`upper`** (optional): Boolean indicating if L is upper triangular (default: false)

## Return Value

Returns a handle to the solution tensor X with the same shape as B, where A * X = B.

## Examples

### Basic Usage
```tcl
# Create a positive definite matrix A and its Cholesky factor L
set A [torch::tensor_create {4.0 2.0 2.0 2.0} float32]
set A [torch::tensor_reshape $A {2 2}]

# Cholesky factor: L = {{2.0 0.0} {1.0 1.0}} where A = L * L^T
set L [torch::tensor_create {2.0 0.0 1.0 1.0} float32]
set L [torch::tensor_reshape $L {2 2}]

# Right-hand side
set B [torch::tensor_create {1.0 1.0} float32]
set B [torch::tensor_reshape $B {2 1}]

# Named parameter syntax (recommended)
set X [torch::cholesky_solve -b $B -l $L]

# Legacy syntax (still works)
set X [torch::cholesky_solve $B $L]

# CamelCase alias
set X [torch::choleskySolve -b $B -l $L]
```

### Advanced Examples
```tcl
# Using upper triangular Cholesky factor
set U [torch::tensor_create {2.0 1.0 0.0 1.0} float32]
set U [torch::tensor_reshape $U {2 2}]
set B [torch::tensor_create {1.0 2.0} float32]
set B [torch::tensor_reshape $B {2 1}]

# Solve with upper triangular factor
set X [torch::cholesky_solve -b $B -l $U -upper 1]

# Parameter order flexibility
set X [torch::cholesky_solve -upper 0 -l $L -b $B]

# Using uppercase parameter aliases
set X [torch::cholesky_solve -B $B -L $L -upper 0]
```

### Batch Operations
```tcl
# Solve multiple systems simultaneously
set B_batch [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
set B_batch [torch::tensor_reshape $B_batch {2 3}]  # 2x3 matrix (multiple RHS)

set L [torch::tensor_create {2.0 0.0 1.0 1.0} float32]
set L [torch::tensor_reshape $L {2 2}]

set X_batch [torch::cholesky_solve -b $B_batch -l $L]
```

### Integration with Cholesky Decomposition
```tcl
# Complete workflow: decomposition and solving
proc solve_positive_definite_system {A B} {
    # Compute Cholesky decomposition
    set L [torch::cholesky $A]
    
    # Solve the system
    set X [torch::cholesky_solve -b $B -l $L -upper 0]
    
    return $X
}

# Example usage
set A [torch::tensor_create {9.0 3.0 3.0 5.0} float32]
set A [torch::tensor_reshape $A {2 2}]
set B [torch::tensor_create {1.0 1.0} float32]
set B [torch::tensor_reshape $B {2 1}]

set solution [solve_positive_definite_system $A $B]
```

## Mathematical Description

The Cholesky solve operation solves the linear system:

**A * X = B**

Where:
- **A** is a positive definite matrix (not directly provided)
- **L** is the Cholesky factor of A, such that:
  - For lower triangular: **A = L * L^T**
  - For upper triangular: **A = U^T * U** (where U is provided as L)
- **B** is the right-hand side tensor
- **X** is the solution tensor (returned)

The algorithm efficiently solves this by:
1. Forward substitution: **L * Y = B** (solve for Y)
2. Backward substitution: **L^T * X = Y** (solve for X)

For upper triangular factors, the process is adapted accordingly.

## Error Handling

### Common Errors
```tcl
# Missing required parameters
torch::cholesky_solve
# Error: Usage: torch::cholesky_solve B L ?upper? | torch::choleskySolve -b tensor -l tensor -upper bool

# Invalid tensor handles
torch::cholesky_solve invalid_B valid_L
# Error: Invalid B tensor

torch::cholesky_solve valid_B invalid_L
# Error: Invalid L tensor

# Unknown parameter
torch::cholesky_solve -b $B -l $L -invalid_param value
# Error: Unknown parameter: -invalid_param. Valid parameters are: -b, -B, -l, -L, -upper

# Missing parameter value
torch::cholesky_solve -b $B -l
# Error: Missing value for parameter

# Invalid upper parameter
torch::cholesky_solve -b $B -l $L -upper invalid_value
# Error: Invalid upper parameter value
```

## Performance Notes

- The new named parameter syntax has equivalent performance to the legacy syntax
- Cholesky solve is numerically stable and efficient for positive definite systems
- Memory efficient - operates in-place where possible
- CUDA tensors are supported for GPU acceleration
- Batch operations are optimized for multiple right-hand sides

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Before (legacy - still works)
set X [torch::cholesky_solve $B $L]
set X [torch::cholesky_solve $B $L 1]

# After (recommended)
set X [torch::cholesky_solve -b $B -l $L]
set X [torch::cholesky_solve -b $B -l $L -upper 1]

# CamelCase alternative
set X [torch::choleskySolve -b $B -l $L]
set X [torch::choleskySolve -b $B -l $L -upper 1]
```

### Parameter Mapping
| Legacy Position | Named Parameter | Alternative |
|----------------|-----------------|-------------|
| 1st argument   | `-b`            | `-B`        |
| 2nd argument   | `-l`            | `-L`        |
| 3rd argument   | `-upper`        | N/A         |

## Use Cases

1. **Linear System Solving**: Efficient solution of positive definite linear systems
2. **Least Squares Problems**: Solving normal equations in optimization
3. **Kalman Filtering**: Covariance matrix computations in state estimation
4. **Machine Learning**: Solving regularized linear regression problems
5. **Scientific Computing**: Solving PDEs with positive definite operators
6. **Statistics**: Computing confidence intervals and hypothesis tests

## Implementation Details

- **Backward Compatible**: Legacy positional syntax fully supported
- **Input Validation**: Comprehensive parameter and tensor validation
- **Error Messages**: Clear, helpful error messages for both syntaxes
- **Numerical Stability**: Leverages optimized LAPACK routines
- **Thread Safe**: Safe for concurrent execution
- **GPU Support**: CUDA acceleration when available

## Related Commands

- [`torch::cholesky`](cholesky.md) - Cholesky decomposition
- [`torch::solve`](solve.md) - General linear system solver
- [`torch::lu_solve`](lu_solve.md) - LU decomposition solver
- [`torch::lstsq`](lstsq.md) - Least squares solver
- [`torch::solve_triangular`](solve_triangular.md) - Triangular system solver

## Mathematical Properties

- **Complexity**: O(N²K) where N is matrix size and K is number of right-hand sides
- **Stability**: Numerically stable for well-conditioned positive definite matrices
- **Accuracy**: High precision due to specialized algorithms
- **Scalability**: Efficient for batch operations

## Version History

- **v1.0**: Original implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase alias
- **Current**: Full backward compatibility maintained

---

**Note**: This command is part of the LibTorch TCL Extension refactoring initiative, providing modern, user-friendly APIs while maintaining full backward compatibility. 