# torch::tensor_matrix_exp

Computes the matrix exponential of a square matrix.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::tensor_matrix_exp -input tensor
torch::tensorMatrixExp -input tensor
```

### Positional Parameters (Legacy)
```tcl
torch::tensor_matrix_exp tensor
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input`, `-tensor` | string | Input square matrix tensor handle | Yes |

## Description

The `tensor_matrix_exp` operation computes the matrix exponential of a square matrix using the formula:

```
exp(A) = I + A + A²/2! + A³/3! + A⁴/4! + ...
```

Where:
- `A` is the input square matrix
- `I` is the identity matrix of the same size
- The series converges for any square matrix

This operation is fundamental in many areas of mathematics and physics, including:
- Solving systems of linear differential equations
- Computing matrix functions
- Quantum mechanics (time evolution operators)
- Control theory and system analysis

## Examples

### Named Parameter Syntax
```tcl
# Create a 2x2 identity matrix
set identity [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
set matrix [torch::tensor_reshape $identity {2 2}]

# Compute matrix exponential
set result [torch::tensor_matrix_exp -input $matrix]

# Alternative parameter name
set result2 [torch::tensor_matrix_exp -tensor $matrix]

# Using camelCase alias
set result3 [torch::tensorMatrixExp -input $matrix]
```

### Positional Parameter Syntax (Legacy)
```tcl
# Same operation using legacy syntax
set identity [torch::tensor_create {1.0 0.0 0.0 1.0} float32 cpu false]
set matrix [torch::tensor_reshape $identity {2 2}]
set result [torch::tensor_matrix_exp $matrix]
```

### Matrix Exponential of Zero Matrix
```tcl
# Create a 3x3 zero matrix
set zero_matrix [torch::zeros {3 3} float32 cpu false]

# Matrix exponential of zero matrix equals identity matrix
set identity_result [torch::tensor_matrix_exp -input $zero_matrix]
# Result will be the 3x3 identity matrix
```

### Diagonal Matrix Example
```tcl
# Create a diagonal matrix
set diag_data {2.0 0.0 0.0 3.0}
set tensor [torch::tensor_create $diag_data float32 cpu false]
set diag_matrix [torch::tensor_reshape $tensor {2 2}]

# Matrix exponential of diagonal matrix
set result [torch::tensor_matrix_exp -input $diag_matrix]
# Result will be a diagonal matrix with exp(2.0) and exp(3.0) on the diagonal
```

### System of Differential Equations
```tcl
# Example coefficient matrix for dx/dt = Ax
set coeff_data {-0.5 1.0 -1.0 -0.5}
set A [torch::tensor_create $coeff_data float32 cpu false]
set matrix_A [torch::tensor_reshape $A {2 2}]

# Compute exp(A) for solving the system
set exp_A [torch::tensor_matrix_exp -input $matrix_A]
# exp_A can be used to find the solution x(t) = exp(At) * x(0)
```

## Return Value

Returns a string handle to the new tensor containing the matrix exponential result. The output tensor has the same shape and data type as the input tensor.

## Error Handling

The command will raise an error in the following cases:

- **Missing parameters**: The input tensor must be provided
- **Invalid tensor**: The specified tensor handle does not exist
- **Non-square matrix**: The input tensor must be a square matrix
- **Unknown parameters**: Using parameter names not recognized by the command

### Error Examples
```tcl
# Error: Missing required parameter
catch {torch::tensor_matrix_exp} error
# Returns: "Required parameter missing: -input"

# Error: Invalid tensor handle
catch {torch::tensor_matrix_exp -input "bad_tensor"} error
# Returns: "Invalid tensor name"

# Error: Non-square matrix
set data {1.0 2.0 3.0 4.0 5.0 6.0}
set tensor [torch::tensor_create $data float32 cpu false]
set non_square [torch::tensor_reshape $tensor {2 3}]  # 2x3 matrix
catch {torch::tensor_matrix_exp -input $non_square} error
# Returns PyTorch error about non-square matrix
```

## Mathematical Properties

### Identity Matrix
```tcl
# exp(I) = e * I where e ≈ 2.71828
set I [torch::eye 3 3 float32 cpu false]
set exp_I [torch::tensor_matrix_exp -input $I]
# Each diagonal element will be approximately 2.71828
```

### Zero Matrix
```tcl
# exp(0) = I (identity matrix)
set zero [torch::zeros {3 3} float32 cpu false]
set exp_zero [torch::tensor_matrix_exp -input $zero]
# Result is the 3x3 identity matrix
```

### Diagonal Matrices
For a diagonal matrix with eigenvalues λ₁, λ₂, ..., λₙ, the matrix exponential is also diagonal with elements exp(λ₁), exp(λ₂), ..., exp(λₙ).

## Performance Considerations

- **Computational Complexity**: O(n³) for an n×n matrix
- **Numerical Stability**: Uses PyTorch's numerically stable implementation
- **Memory Usage**: Creates a new tensor; does not modify the input
- **GPU Support**: Fully supports CUDA tensors for GPU acceleration

```tcl
# Example with CUDA tensor (if CUDA is available)
set matrix_cpu [torch::tensor_create {1.0 0.5 0.5 1.0} float32 cpu false]
set matrix_2x2 [torch::tensor_reshape $matrix_cpu {2 2}]
set matrix_gpu [torch::tensor_to $matrix_2x2 cuda]
set result_gpu [torch::tensor_matrix_exp -input $matrix_gpu]
```

## Migration Guide

### From Positional to Named Parameters

**Old Syntax:**
```tcl
set result [torch::tensor_matrix_exp $matrix]
```

**New Syntax:**
```tcl
set result [torch::tensor_matrix_exp -input $matrix]
# or using camelCase
set result [torch::tensorMatrixExp -input $matrix]
```

### Parameter Mapping

| Positional Order | Named Parameter | Alternative Names |
|------------------|-----------------|-------------------|
| 1st argument | `-input` | `-tensor` |

## Compatibility

- ✅ **Backward Compatible**: All existing code using positional syntax continues to work
- ✅ **New Features**: Named parameters provide better readability and maintainability
- ✅ **Flexible**: Multiple parameter name aliases for convenience
- ✅ **Modern**: camelCase alias follows contemporary API design patterns

## See Also

- [`torch::tensor_matrix_power`](tensor_matrix_power.md) - Matrix power operations
- [`torch::tensor_exp`](tensor_exp.md) - Element-wise exponential
- [`torch::tensor_eig`](tensor_eig.md) - Eigenvalue decomposition
- [`torch::tensor_svd`](tensor_svd.md) - Singular value decomposition
- [`torch::tensor_cholesky`](tensor_cholesky.md) - Cholesky decomposition

## Technical Notes

- Uses PyTorch's `torch::linalg::matrix_exp()` function internally
- Supports autograd if input tensor requires gradients
- Works with all floating-point data types (float32, float64)
- Requires square matrices (n×n dimensions)
- Numerically stable implementation using scaling and squaring with Padé approximation

## Applications

### Solving Linear ODEs
Matrix exponentials are the solution to linear systems of differential equations:
```
dx/dt = Ax  →  x(t) = exp(At) * x(0)
```

### Quantum Mechanics
Time evolution operators in quantum mechanics:
```
U(t) = exp(-iHt/ℏ)
```

### Control Theory
State transition matrices in linear control systems:
```
x(t) = exp(At) * x(0) + ∫[0,t] exp(A(t-τ)) * B * u(τ) dτ
``` 