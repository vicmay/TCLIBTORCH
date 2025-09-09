# torch::lstsq

Computes the least squares solution to a linear system of equations using SVD decomposition.

## Syntax

### Current Syntax
```tcl
torch::lstsq B A ?rcond?
```

### Named Parameter Syntax  
```tcl
torch::lstsq -b tensor -a tensor ?-rcond double?
```

### camelCase Alias
```tcl
torch::leastSquares -b tensor -a tensor ?-rcond double?
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-b` (required): Right-hand side tensor (observations)
- `-a` (required): Coefficient matrix tensor  
- `-rcond` (optional): Cutoff for small singular values (default: machine precision)

### Alternative Parameter Names
- `-B` (alternative to `-b`): Right-hand side tensor
- `-A` (alternative to `-a`): Coefficient matrix tensor

### Positional Parameters
1. `B` (required): Right-hand side tensor name
2. `A` (required): Coefficient matrix tensor name  
3. `rcond` (optional): Cutoff for small singular values

## Description

The `torch::lstsq` function solves the linear system `Ax = B` using the least squares method. It computes the solution that minimizes the sum of squared residuals `||Ax - B||²`. This is particularly useful for overdetermined systems (more equations than unknowns) where an exact solution may not exist.

The function uses Singular Value Decomposition (SVD) to provide a robust solution even when the coefficient matrix A is singular or near-singular.

## Mathematical Details

For a system `Ax = B` where:
- `A` is an (m × n) coefficient matrix
- `B` is an (m × k) right-hand side matrix
- `x` is the (n × k) solution matrix

The least squares solution minimizes:
```
||Ax - B||²
```

Key properties:
- For overdetermined systems (m > n): Finds the best-fit solution
- For underdetermined systems (m < n): Returns the minimum-norm solution
- For square systems (m = n): Equivalent to solving the linear system directly
- Uses SVD for numerical stability

The `rcond` parameter controls the effective rank of the matrix by treating singular values smaller than `rcond * largest_singular_value` as zero.

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Solve overdetermined system Ax = b
set A [torch::tensor_create {{1.0 1.0} {1.0 2.0} {1.0 3.0}} float32]
set b [torch::tensor_create {{6.0} {8.0} {10.0}} float32]
set solution [torch::lstsq $b $A]
```

#### Named Parameter Syntax
```tcl
# Same operation using named parameters
set A [torch::tensor_create {{1.0 1.0} {1.0 2.0} {1.0 3.0}} float32]
set b [torch::tensor_create {{6.0} {8.0} {10.0}} float32]
set solution [torch::lstsq -b $b -a $A]
```

#### camelCase Alias
```tcl
# Using camelCase alias
set solution [torch::leastSquares -b $b -a $A]
```

### Linear Regression Example

```tcl
# Fit a line y = mx + c to data points
# A matrix: [[x1, 1], [x2, 1], [x3, 1], ...] 
# b vector: [y1, y2, y3, ...]

set x_data [torch::tensor_create {{1.0} {2.0} {3.0} {4.0} {5.0}} float32]
set y_data [torch::tensor_create {{2.1} {3.9} {6.1} {7.8} {10.2}} float32]

# Create coefficient matrix [x, 1] for y = mx + c
set ones [torch::ones {5 1} float32]
set A [torch::cat $x_data $ones -dim 1]

# Solve for [m, c] parameters
set params [torch::lstsq -b $y_data -a $A]
puts "Slope and intercept: [torch::tensor_to_list $params]"
```

### Polynomial Fitting

```tcl
# Fit quadratic polynomial y = ax² + bx + c
set x [torch::tensor_create {{1.0} {2.0} {3.0} {4.0} {5.0}} float32]
set y [torch::tensor_create {{1.2} {3.8} {8.9} {16.1} {25.2}} float32]

# Create Vandermonde matrix [x², x, 1]
set x_squared [torch::tensor_mul $x $x]
set ones [torch::ones {5 1} float32]
set A [torch::cat $x_squared $x -dim 1]
set A [torch::cat $A $ones -dim 1]

# Solve for [a, b, c] coefficients
set coeffs [torch::lstsq -b $y -a $A -rcond 1e-12]
```

### Multiple Right-Hand Sides

```tcl
# Solve multiple systems simultaneously
set A [torch::tensor_create {{1.0 2.0} {3.0 4.0} {5.0 6.0}} float32]
set B [torch::tensor_create {{1.0 7.0} {2.0 8.0} {3.0 9.0}} float32]
set solutions [torch::lstsq -b $B -a $A]
# Each column of solutions corresponds to one right-hand side
```

### Controlling Numerical Precision

```tcl
# Use stricter tolerance for singular values
set A [torch::tensor_create {{1.0 1.0} {1.0 1.0001}} float32]
set b [torch::tensor_create {{2.0} {2.0001}} float32]

# Default tolerance (may be sensitive to noise)
set sol1 [torch::lstsq $b $A]

# Stricter tolerance
set sol2 [torch::lstsq -b $b -a $A -rcond 1e-10]
```

## Input Requirements

### Matrix A (coefficient matrix)
- **Shape**: (m × n) where m ≥ n typically
- **Data Type**: Floating-point (float32, float64) or complex (complex64, complex128)
- **Rank**: Can be singular or near-singular

### Matrix B (right-hand side)
- **Shape**: (m × k) where m matches A's first dimension
- **Data Type**: Must match A's data type
- **Multiple systems**: k can be > 1 for multiple right-hand sides

### rcond parameter
- **Type**: Double precision floating-point
- **Range**: Positive values, typically 1e-15 to 1e-6
- **Default**: Machine precision for the data type

## Output

Returns a tensor containing the least squares solution with shape (n × k), where:
- n is the number of columns in A
- k is the number of columns in B

## Error Handling

The function will raise an error if:
- Tensor names are invalid or don't exist
- Required parameters (A and B) are missing
- Matrix dimensions are incompatible (A.shape[0] != B.shape[0])
- rcond parameter is invalid (negative or non-numeric)
- Unknown parameters are provided
- Data types of A and B don't match

## Performance Considerations

- **Computational Complexity**: O(mn² + n³) for SVD decomposition
- **Memory Usage**: Creates temporary matrices for SVD computation
- **GPU Acceleration**: Available for CUDA tensors
- **Numerical Stability**: SVD provides robust solutions for ill-conditioned systems
- **Large Systems**: Consider iterative methods for very large problems

## Common Use Cases

1. **Linear Regression**: Fitting lines and curves to data
2. **Polynomial Fitting**: Finding polynomial coefficients
3. **System Identification**: Estimating model parameters
4. **Signal Processing**: Parameter estimation in signal models
5. **Computer Vision**: Solving homography and camera calibration problems
6. **Machine Learning**: Training linear models and neural network layers

## Related Functions

- `torch::solve` - Exact solution for square systems
- `torch::cholesky_solve` - Efficient solution for symmetric positive definite systems
- `torch::svd` - Singular Value Decomposition
- `torch::pinverse` - Moore-Penrose pseudoinverse
- `torch::matrix_rank` - Rank computation
- `torch::cond` - Condition number estimation

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax (still supported)
set solution [torch::lstsq $B $A]
set solution [torch::lstsq $B $A 1e-12]

# New named parameter syntax
set solution [torch::lstsq -b $B -a $A]
set solution [torch::lstsq -b $B -a $A -rcond 1e-12]

# camelCase alias
set solution [torch::leastSquares -b $B -a $A]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter roles are explicit (-b for RHS, -a for coefficient matrix)
2. **Flexibility**: Parameters can be provided in any order
3. **Error Prevention**: Reduces chance of swapping A and B matrices
4. **Extensibility**: Easy to add optional parameters in the future
5. **Consistency**: Matches modern TCL conventions

## Technical Notes

- Implements PyTorch's `torch.linalg.lstsq()` function
- Uses SVD decomposition for numerical stability
- Supports automatic differentiation when `requires_grad=true`
- Thread-safe operation
- Handles both real and complex matrices
- Preserves tensor properties (device, dtype, etc.)

## Numerical Considerations

- **Conditioning**: Well-conditioned problems (low condition number) give more accurate results
- **Rank Deficiency**: Rank-deficient matrices are handled gracefully via SVD
- **Regularization**: Consider adding regularization for ill-posed problems
- **Scaling**: Proper scaling of variables can improve numerical stability

## Version History

- Added dual syntax support in refactoring phase
- Original positional syntax maintained for backward compatibility
- Named parameter syntax added for modern TCL conventions
- camelCase alias `torch::leastSquares` added for improved readability 