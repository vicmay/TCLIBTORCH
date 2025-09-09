# torch::trace

Computes the trace of a matrix (sum of the elements on the main diagonal). The trace is a fundamental property of square matrices and is invariant under similarity transformations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::trace input
```

### Named Parameter Syntax (New)
```tcl
torch::trace -input matrix
```

### CamelCase Alias
```tcl
torch::Trace -input matrix
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` / `-input` | string | Yes | Name of the input matrix tensor |

## Return Value

Returns a string handle to the resulting scalar tensor containing the trace value.

## Description

The trace of a matrix is the sum of the elements on its main diagonal (from top-left to bottom-right). For a matrix A with elements aᵢⱼ, the trace is:

```
tr(A) = a₁₁ + a₂₂ + a₃₃ + ... + aₙₙ
```

**Mathematical Properties:**
- **Linearity**: tr(A + B) = tr(A) + tr(B)
- **Scalar multiplication**: tr(cA) = c·tr(A)
- **Cyclic property**: tr(ABC) = tr(BCA) = tr(CAB)
- **Identity matrix**: tr(Iₙ) = n
- **Zero matrix**: tr(0) = 0

**Applications:**
- Linear algebra computations
- Matrix analysis and eigendecomposition
- Quantum mechanics (density matrices)
- Machine learning (gradient computations)
- Control theory

## Examples

### Basic Usage

```tcl
# Create a 3x3 matrix
set matrix [torch::tensor_create -data {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}}]

# Compute trace with positional syntax
set trace [torch::trace $matrix]
puts [torch::tensor_to_list $trace]
# Output: 15.0

# Compute trace with named syntax
set trace2 [torch::trace -input $matrix]
puts [torch::tensor_to_list $trace2]
# Output: 15.0
```

### Identity Matrix

```tcl
# Create 3x3 identity matrix
set identity [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 1.0 0.0} {0.0 0.0 1.0}}]

set trace [torch::trace $identity]
puts [torch::tensor_to_list $trace]
# Output: 3.0
```

### Using CamelCase Alias

```tcl
set matrix [torch::tensor_create -data {{1.0 2.0} {3.0 4.0}}]

# Use camelCase alias
set trace [torch::Trace -input $matrix]
puts [torch::tensor_to_list $trace]
# Output: 5.0
```

### Different Matrix Types

```tcl
# Zero matrix
set zero [torch::tensor_create -data {{0.0 0.0 0.0} {0.0 0.0 0.0} {0.0 0.0 0.0}}]
set trace [torch::trace $zero]
puts [torch::tensor_to_list $trace]
# Output: 0.0

# Diagonal matrix
set diagonal [torch::tensor_create -data {{2.0 0.0 0.0} {0.0 3.0 0.0} {0.0 0.0 4.0}}]
set trace [torch::trace $diagonal]
puts [torch::tensor_to_list $trace]
# Output: 9.0

# Matrix with negative values
set negative [torch::tensor_create -data {{-1.0 -2.0} {-3.0 -4.0}}]
set trace [torch::trace $negative]
puts [torch::tensor_to_list $trace]
# Output: -5.0
```

### Non-Square Matrices

```tcl
# 2x3 matrix (trace = sum of min dimension)
set rect [torch::tensor_create -data {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
set trace [torch::trace $rect]
puts [torch::tensor_to_list $trace]
# Output: 6.0

# 3x2 matrix
set rect2 [torch::tensor_create -data {{1.0 2.0} {3.0 4.0} {5.0 6.0}}]
set trace [torch::trace $rect2]
puts [torch::tensor_to_list $trace]
# Output: 5.0
```

### Large Matrices

```tcl
# Large matrix with known trace
set large [torch::tensor_create -data {{100.0 200.0 300.0} {400.0 500.0 600.0} {700.0 800.0 900.0}}]
set trace [torch::trace $large]
puts [torch::tensor_to_list $trace]
# Output: 1500.0
```

## Migration Guide

### From Positional to Named Syntax

**Old (Positional):**
```tcl
torch::trace $matrix
```

**New (Named Parameters):**
```tcl
torch::trace -input $matrix
```

### Benefits of Named Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Documentation**: Self-documenting code

## Error Handling

The command will throw an error in the following cases:

- **Missing required parameters**: `Required parameters missing: input tensor required`
- **Invalid tensor name**: `Invalid input tensor`
- **Unknown parameter**: `Unknown parameter: -invalid. Valid parameters are: -input`
- **Missing parameter value**: `Missing value for parameter`

### Error Examples

```tcl
# Missing arguments
torch::trace
# Error: Required parameters missing: input tensor required

# Invalid tensor
torch::trace invalid_tensor
# Error: Invalid input tensor

# Unknown parameter
set matrix [torch::tensor_create -data {{1.0 2.0} {3.0 4.0}}]
torch::trace -input $matrix -invalid param
# Error: Unknown parameter: -invalid. Valid parameters are: -input

# Missing parameter value
torch::trace -input
# Error: Missing value for parameter
```

## Notes

- **Backward Compatibility**: The positional syntax is fully supported for backward compatibility
- **Matrix Requirements**: Works with any 2D tensor (square or rectangular)
- **Non-Square Matrices**: For rectangular matrices, computes trace of the square submatrix
- **Data Types**: Supports all numeric tensor data types
- **Performance**: Efficient computation for large matrices
- **Memory**: Creates a new scalar tensor, minimal memory overhead
- **Mathematical Accuracy**: Preserves numerical precision of the input tensor

## Mathematical Background

The trace is a linear operator that satisfies:

1. **tr(A + B) = tr(A) + tr(B)** - Additivity
2. **tr(cA) = c·tr(A)** - Homogeneity
3. **tr(AB) = tr(BA)** - Cyclic property
4. **tr(A^T) = tr(A)** - Invariance under transpose

For eigenvalues λᵢ of matrix A:
- **tr(A) = Σλᵢ** - Trace equals sum of eigenvalues
- **tr(A^n) = Σλᵢ^n** - Trace of powers

## Related Commands

- `torch::diag` - Extract diagonal elements or create diagonal matrix
- `torch::matrix_power` - Compute matrix powers
- `torch::eig` - Compute eigenvalues and eigenvectors
- `torch::det` - Compute matrix determinant
- `torch::inv` - Compute matrix inverse 