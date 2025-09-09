# torch::spectral_norm

Applies spectral normalization to a tensor by normalizing it by its largest singular value.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::spectral_norm tensor ?n_power_iterations?
```

### Named Parameter Syntax
```tcl
torch::spectral_norm tensor ?n_power_iterations?
```

### CamelCase Alias
```tcl
torch::spectralNorm tensor ?n_power_iterations?
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor | tensor | The input tensor to normalize (must be at least 2D) |
| n_power_iterations | int | Optional. Number of power iterations to compute spectral norm (default: 1) |

## Return Value

Returns a new tensor normalized by its largest singular value.

## Examples

### Basic Usage
```tcl
# Create a test matrix
set matrix [torch::tensor_create -data {{4.0 0.0 0.0} {0.0 3.0 0.0} {0.0 0.0 2.0}} -dtype "float32"]

# Apply spectral normalization
set result1 [torch::spectral_norm $matrix]

# Apply spectral normalization with more power iterations
set result2 [torch::spectral_norm $matrix 5]

# Using camelCase alias
set result3 [torch::spectralNorm $matrix]
```

## Error Handling

The command will raise an error in the following cases:
- If the input tensor is not valid
- If the input tensor has less than 2 dimensions
- If n_power_iterations is not a valid integer

## Migration Guide

### From Positional to Named Parameter Syntax

The command currently only supports positional syntax. Both old and new code use the same syntax:

```tcl
torch::spectral_norm $tensor 5
# or using camelCase alias
torch::spectralNorm $tensor 5
```

## Technical Details

Spectral normalization is computed using the power iteration method to estimate the largest singular value of the input tensor. The tensor is then normalized by dividing it by this value.

For a matrix A, the spectral norm is equivalent to the largest singular value σ₁:
```
spectral_norm(A) = σ₁ = max{|λ| : λ is an eigenvalue of A^T A}^(1/2)
```

The power iteration method is used to approximate this value efficiently:
1. Start with a random vector u
2. Repeatedly compute:
   - v = A^T u / ||A^T u||
   - u = A v / ||A v||
3. The spectral norm is approximated by ||A v||

The number of power iterations can be controlled with the optional parameter n_power_iterations. 