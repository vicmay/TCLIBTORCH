# torch::matrix_norm

Computes the matrix norm of a tensor.

## Syntax

### Positional Syntax
```tcl
torch::matrix_norm input ?ord? ?dim? ?keepdim?
```

### Named Parameter Syntax
```tcl
torch::matrix_norm -input tensor_name ?-ord norm_type? ?-dim dimensions? ?-keepdim boolean?
```

### camelCase Alias
```tcl
torch::matrixNorm -input tensor_name ?-ord norm_type? ?-dim dimensions? ?-keepdim boolean?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | tensor | required | Input tensor |
| `ord` | string/number | "fro" | Order of the norm ("fro", "nuc", or numeric value) |
| `dim` | list | {} | Dimensions to compute norm over |
| `keepdim` | boolean | false | Keep dimensions after reduction |

## Description

The `torch::matrix_norm` command computes the matrix norm of the input tensor. The norm order can be specified using the `ord` parameter:

- `"fro"` (default): Frobenius norm
- `"nuc"`: Nuclear norm (sum of singular values)
- Numeric value: Specific norm order (e.g., 1, 2, -1, -2)

When `dim` is specified, the norm is computed over the specified dimensions. When `keepdim` is true, the output tensor has the same number of dimensions as the input tensor, with the reduced dimensions having size 1.

## Examples

### Basic Usage
```tcl
# Create a 2x2 matrix
set data [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu false]
set A [torch::tensor_reshape $data {2 2}]

# Compute Frobenius norm (default)
set norm [torch::matrix_norm $A]
# Result: approximately 5.477

# Using named syntax
set norm [torch::matrix_norm -input $A]
# Result: approximately 5.477

# Using camelCase alias
set norm [torch::matrixNorm -input $A]
# Result: approximately 5.477
```

### Different Norm Types
```tcl
# Frobenius norm
set fro_norm [torch::matrix_norm $A fro]
# Result: approximately 5.477

# Nuclear norm
set nuc_norm [torch::matrix_norm $A nuc]
# Result: approximately 5.831

# 2-norm (spectral norm)
set spec_norm [torch::matrix_norm $A 2]
# Result: approximately 5.465

# Using named syntax
set fro_norm [torch::matrix_norm -input $A -ord fro]
set nuc_norm [torch::matrix_norm -input $A -ord nuc]
set spec_norm [torch::matrix_norm -input $A -ord 2]
```

### With Dimensions
```tcl
# Create a 3D tensor
set data [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32 cpu false]
set B [torch::tensor_reshape $data {2 2 2}]

# Compute norm over last two dimensions
set norm [torch::matrix_norm $B fro {1 2}]
# Result: tensor with shape [2]

# Using named syntax
set norm [torch::matrix_norm -input $B -ord fro -dim {1 2}]
# Result: tensor with shape [2]
```

### With keepdim
```tcl
# Keep dimensions after reduction
set norm [torch::matrix_norm $B fro {1 2} 1]
# Result: tensor with shape [2 1 1]

# Using named syntax
set norm [torch::matrix_norm -input $B -ord fro -dim {1 2} -keepdim 1]
# Result: tensor with shape [2 1 1]
```

## Return Value

Returns a tensor handle containing the computed matrix norm values.

## Error Handling

The command will throw an error if:
- The input tensor is not provided
- The input tensor is not a valid tensor handle
- The input tensor is not a floating-point tensor
- Unknown parameters are provided

## Backward Compatibility

The original positional syntax is fully supported:
```tcl
# These are equivalent
set norm1 [torch::matrix_norm $A fro {1 2} 1]
set norm2 [torch::matrix_norm -input $A -ord fro -dim {1 2} -keepdim 1]
```

## Mathematical Notes

The matrix norm is computed according to the specified order:
- **Frobenius norm**: `||A||_F = sqrt(sum(A^2))`
- **Nuclear norm**: `||A||_* = sum(σ_i)` where σ_i are singular values
- **p-norm**: Various matrix norms for different values of p

The function requires floating-point tensors and will throw an error for integer tensors.

## See Also

- [torch::norm](norm.md) - Vector and tensor norms
- [torch::linalg_norm](linalg_norm.md) - General linear algebra norms
- [torch::svd](svd.md) - Singular value decomposition 