# torch::solve_triangular

Solves a triangular system of equations Ax = B where A is a triangular matrix.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::solve_triangular -B tensor -A tensor ?-upper bool? ?-left bool? ?-unitriangular bool?
torch::solveTriangular -B tensor -A tensor ?-upper bool? ?-left bool? ?-unitriangular bool?
```

### Positional Parameters (Legacy)
```tcl
torch::solve_triangular B A ?upper? ?left? ?unitriangular?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `B` | tensor | required | Right-hand side tensor of shape (..., n, k) |
| `A` | tensor | required | Triangular matrix tensor of shape (..., n, n) |
| `upper` | bool | true | Whether A is upper triangular (true) or lower triangular (false) |
| `left` | bool | true | Whether to solve Ax = B (true) or xA = B (false) |
| `unitriangular` | bool | false | Whether A is unit triangular (diagonal elements are all 1) |

## Returns

A tensor containing the solution to the triangular system.

## Description

This function solves triangular systems of linear equations. It is particularly useful for:
- Forward substitution (lower triangular matrices)
- Back substitution (upper triangular matrices)
- Solving systems after LU decomposition
- Solving systems after Cholesky decomposition

The function leverages the triangular structure of the matrix A for efficient computation.

## Examples

### Basic Usage with Named Parameters

```tcl
# Create an upper triangular matrix A (3x3)
set A [torch::tensorCreate -data {3.0 2.0 1.0 0.0 4.0 2.0 0.0 0.0 5.0} -shape {3 3} -dtype float32]

# Create right-hand side B (3x1)
set B [torch::tensorCreate -data {14.0 18.0 15.0} -shape {3 1} -dtype float32]

# Solve the system Ax = B (upper triangular)
set x [torch::solve_triangular -B $B -A $A -upper 1]
puts "Solution: [torch::tensorToList $x]"
```

### Lower Triangular System

```tcl
# Create a lower triangular matrix
set A [torch::tensorCreate -data {2.0 0.0 0.0 1.0 3.0 0.0 4.0 2.0 1.0} -shape {3 3} -dtype float32]
set B [torch::tensorCreate -data {4.0 8.0 18.0} -shape {3 1} -dtype float32]

# Solve lower triangular system
set x [torch::solve_triangular -B $B -A $A -upper 0]
```

### Using camelCase Alias

```tcl
# Same functionality with camelCase name
set result [torch::solveTriangular -B $B -A $A -upper 1 -left 1]
```

### Batch Processing

```tcl
# Solve multiple systems simultaneously
set A [torch::tensorCreate -data {3.0 2.0 1.0 0.0 4.0 2.0 0.0 0.0 5.0} -shape {3 3} -dtype float32]
set B [torch::tensorCreate -data {14.0 21.0 18.0 26.0 15.0 20.0} -shape {3 2} -dtype float32]

# Solve for multiple right-hand sides
set solutions [torch::solve_triangular -B $B -A $A]
```

### Right-side Multiplication (xA = B)

```tcl
# Solve xA = B instead of Ax = B
set result [torch::solve_triangular -B $B -A $A -left 0]
```

### Unit Triangular Matrix

```tcl
# For unit triangular matrices (diagonal elements are 1)
set result [torch::solve_triangular -B $B -A $A -unitriangular 1]
```

## Legacy Syntax Examples

```tcl
# Basic positional syntax
set result [torch::solve_triangular $B $A 1 1 0]

# With optional parameters
set result [torch::solve_triangular $B $A 0 1 0]  # lower triangular
```

## Mathematical Background

For an upper triangular system Ax = b:
```
[a₁₁ a₁₂ a₁₃] [x₁]   [b₁]
[  0 a₂₂ a₂₃] [x₂] = [b₂]
[  0   0 a₃₃] [x₃]   [b₃]
```

The solution is computed via back substitution:
- x₃ = b₃ / a₃₃
- x₂ = (b₂ - a₂₃×x₃) / a₂₂
- x₁ = (b₁ - a₁₂×x₂ - a₁₃×x₃) / a₁₁

For lower triangular systems, forward substitution is used.

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::solve_triangular -A $A} msg
# Error: Required parameters missing: B and A tensors required

# Invalid tensor handle
catch {torch::solve_triangular -B $B -A "invalid"} msg
# Error: Invalid A tensor

# Invalid parameter values
catch {torch::solve_triangular -B $B -A $A -upper "invalid"} msg
# Error: Invalid upper parameter value
```

## Performance Notes

- The function is optimized for triangular matrices and is much faster than general linear solvers
- Batch processing is supported for solving multiple systems efficiently
- Memory usage is minimal as no matrix factorization is required

## Related Functions

- `torch::lstsq` - General least squares solver
- `torch::cholesky_solve` - Solve using Cholesky decomposition
- `torch::lu_solve` - Solve using LU decomposition

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::solve_triangular $B $A 1 1 0]

# New named parameter syntax
set result [torch::solve_triangular -B $B -A $A -upper 1 -left 1 -unitriangular 0]

# Simplified with defaults
set result [torch::solve_triangular -B $B -A $A]  # uses defaults: upper=1, left=1, unitriangular=0
```

### Common Migration Patterns

```tcl
# Lower triangular system
# Old: torch::solve_triangular $B $A 0 1 0
# New: torch::solve_triangular -B $B -A $A -upper 0

# Right multiplication
# Old: torch::solve_triangular $B $A 1 0 0  
# New: torch::solve_triangular -B $B -A $A -left 0

# Unit triangular
# Old: torch::solve_triangular $B $A 1 1 1
# New: torch::solve_triangular -B $B -A $A -unitriangular 1
```

## Version History

- **Current**: Added named parameter support and camelCase alias
- **Legacy**: Positional parameter syntax (still supported for backward compatibility) 