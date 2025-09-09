# torch::einsum

## Overview

Einstein summation (`torch::einsum`) is a powerful and compact notation for expressing multi-dimensional array operations. It provides a flexible way to perform tensor contractions, matrix multiplications, transpositions, reductions, and many other linear algebra operations using a simple equation string.

## Mathematical Formulation

Einstein summation convention allows expressing complex tensor operations using subscript notation:
- **Repeated indices** are automatically summed over
- **Free indices** appear in the output
- **Input tensors** are separated by commas
- **Output specification** follows the arrow (`->`)

### General Form
```
einsum("subscripts->output", tensor1, tensor2, ...)
```

### Examples of Operations
- Matrix multiplication: `"ij,jk->ik"`
- Trace: `"ii->"`
- Transpose: `"ij->ji"`
- Element-wise multiply: `"ij,ij->ij"`
- Sum over axis: `"ij->i"`

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::einsum equation tensor1 [tensor2 ...]
```

### Named Parameter Syntax
```tcl
torch::einsum -equation equation -tensors {tensor1 tensor2 ...}
torch::einsum -equation equation -tensors tensor1  # Single tensor
```

### CamelCase Alias
```tcl
torch::Einsum equation tensor1 [tensor2 ...]
torch::Einsum -equation equation -tensors {tensor1 tensor2 ...}
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `equation` | string | Yes | Einstein summation equation string |
| `tensors` | list/tensor | Yes | One or more input tensors |

### Equation Format
- **Input subscripts**: Letters representing tensor dimensions
- **Comma separation**: Between different input tensors
- **Arrow notation**: `->` separates inputs from output
- **Output subscripts**: Desired output dimension order
- **Implicit sum**: Repeated indices are summed over

## Common Operations

### 1. Matrix Multiplication
```tcl
# Create matrices
set A [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]
set B [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2}]

# Matrix multiplication A @ B
set result [torch::einsum "ij,jk->ik" $A $B]
# Equivalent to: torch::tensor_matmul $A $B
```

### 2. Batch Matrix Multiplication
```tcl
# Batch of matrices
set batch_A [torch::tensor_create -data {...} -shape {8 3 4}]
set batch_B [torch::tensor_create -data {...} -shape {8 4 5}]

# Batch matrix multiplication
set result [torch::einsum "bij,bjk->bik" $batch_A $batch_B]
# Output shape: {8 3 5}
```

### 3. Trace Computation
```tcl
# Square matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]

# Compute trace (sum of diagonal elements)
set trace [torch::einsum "ii->" $matrix]
# Result: scalar tensor with value 5.0 (1+4)
```

### 4. Transpose Operations
```tcl
# 2D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3}]

# Simple transpose
set transposed [torch::einsum "ij->ji" $tensor]
# Output shape: {3 2}

# Multi-dimensional transpose
set tensor_3d [torch::tensor_create -data {...} -shape {2 3 4}]
set reordered [torch::einsum "ijk->kji" $tensor_3d]
# Output shape: {4 3 2}
```

### 5. Element-wise Operations
```tcl
# Two matrices of same shape
set A [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]
set B [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2}]

# Element-wise multiplication (Hadamard product)
set result [torch::einsum "ij,ij->ij" $A $B]
# Equivalent to: torch::tensor_mul $A $B
```

### 6. Reduction Operations
```tcl
# 2D tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3}]

# Sum over columns (axis 1)
set col_sum [torch::einsum "ij->i" $tensor]
# Result shape: {2} - sums each row

# Sum over rows (axis 0)
set row_sum [torch::einsum "ij->j" $tensor]
# Result shape: {3} - sums each column

# Sum all elements
set total_sum [torch::einsum "ij->" $tensor]
# Result: scalar tensor
```

### 7. Outer Product
```tcl
# Two vectors
set vec1 [torch::tensor_create -data {1.0 2.0} -shape {2}]
set vec2 [torch::tensor_create -data {3.0 4.0} -shape {2}]

# Outer product
set outer [torch::einsum "i,j->ij" $vec1 $vec2]
# Result: 2x2 matrix
```

### 8. Inner Product (Dot Product)
```tcl
# Two vectors
set vec1 [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
set vec2 [torch::tensor_create -data {4.0 5.0 6.0} -shape {3}]

# Inner product
set dot [torch::einsum "i,i->" $vec1 $vec2]
# Result: scalar tensor with value 32.0 (1*4 + 2*5 + 3*6)
```

### 9. Diagonal Extraction
```tcl
# Square matrix
set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 3}]

# Extract diagonal elements
set diagonal [torch::einsum "ii->i" $matrix]
# Result: vector with diagonal elements [1.0, 5.0, 9.0]
```

### 10. Bilinear Operations
```tcl
# Three tensors for bilinear form
set x [torch::tensor_create -data {...} -shape {3}]
set A [torch::tensor_create -data {...} -shape {3 4}]
set y [torch::tensor_create -data {...} -shape {4}]

# Bilinear form: x^T A y
set result [torch::einsum "i,ij,j->" $x $A $y]
# Result: scalar tensor
```

## Advanced Examples

### Attention Mechanism (Simplified)
```tcl
# Query, Key, Value tensors
set Q [torch::tensor_create -data {...} -shape {8 64}]  # sequence_length x d_model
set K [torch::tensor_create -data {...} -shape {8 64}]
set V [torch::tensor_create -data {...} -shape {8 64}]

# Attention scores: Q @ K^T
set scores [torch::einsum "iq,kq->ik" $Q $K]

# After softmax (assume scores_normalized)
# Attention output: scores @ V
set output [torch::einsum "ik,kv->iv" $scores_normalized $V]
```

### Tensor Contraction with Multiple Indices
```tcl
# 4D tensors
set A [torch::tensor_create -data {...} -shape {2 3 4 5}]
set B [torch::tensor_create -data {...} -shape {4 5 6 7}]

# Contract over dimensions 2 and 3 of A with dimensions 0 and 1 of B
set result [torch::einsum "ijkl,klmn->ijmn" $A $B]
# Output shape: {2 3 6 7}
```

## Performance Considerations

### Optimization Tips
1. **Order matters**: Place smaller tensors first when possible
2. **Memory efficiency**: Consider intermediate tensor sizes
3. **Broadcasting**: Einsum handles broadcasting automatically
4. **Alternative operations**: For simple operations, use specialized functions

### Memory Usage
```tcl
# Memory-efficient: direct contraction
set result [torch::einsum "ijk,jkl->il" $A $B]

# Less efficient: intermediate tensor creation
set temp [torch::einsum "ijk->ikj" $A]
set result [torch::tensor_matmul $temp $B]
```

### When to Use Alternatives
- **Matrix multiplication**: Use `torch::tensor_matmul` for simple 2D cases
- **Element-wise operations**: Use `torch::tensor_mul`, `torch::tensor_add`, etc.
- **Transpose**: Use `torch::tensor_transpose` for simple transposes
- **Reductions**: Use `torch::tensor_sum`, `torch::tensor_mean` for axis-specific reductions

## Error Handling

Common error scenarios and their solutions:

### Invalid Equation Format
```tcl
# Error: malformed equation
catch {torch::einsum "invalid_equation" $tensor} error
# Solution: Use proper Einstein notation
```

### Dimension Mismatch
```tcl
# Error: incompatible tensor shapes
set A [torch::tensor_create -data {...} -shape {2 3}]
set B [torch::tensor_create -data {...} -shape {4 5}]
catch {torch::einsum "ij,jk->ik" $A $B} error
# Solution: Ensure shared dimensions match
```

### Invalid Tensor Names
```tcl
# Error: tensor doesn't exist
catch {torch::einsum "ij->" "nonexistent_tensor"} error
# Solution: Use valid tensor handles
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set result [torch::einsum "ij,jk->ik" $A $B]
```

**After:**
```tcl
set result [torch::einsum -equation "ij,jk->ik" -tensors [list $A $B]]
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Extensibility**: Easy to add new options in the future
- **Error prevention**: Reduces positional argument mistakes

## Comparison with Other Operations

| Operation | Einsum Notation | Equivalent Function |
|-----------|----------------|-------------------|
| Matrix multiply | `"ij,jk->ik"` | `torch::tensor_matmul` |
| Element-wise multiply | `"ij,ij->ij"` | `torch::tensor_mul` |
| Sum all | `"ij->"` | `torch::tensor_sum` |
| Sum axis 0 | `"ij->j"` | `torch::tensor_sum $tensor 0` |
| Sum axis 1 | `"ij->i"` | `torch::tensor_sum $tensor 1` |
| Transpose | `"ij->ji"` | `torch::tensor_transpose` |
| Trace | `"ii->"` | `torch::tensor_trace` |
| Outer product | `"i,j->ij"` | `torch::tensor_outer` |
| Inner product | `"i,i->"` | `torch::tensor_dot` |

## Implementation Details

### Dual Syntax Support
The implementation automatically detects syntax type:
- **Positional**: When first argument doesn't start with `-`
- **Named**: When first argument starts with `-`

### Parameter Validation
- Equation string cannot be empty
- At least one tensor must be provided
- All tensor names must be valid handles
- Tensor dimensions must be compatible with equation

### Return Value
Returns a new tensor handle containing the result of the Einstein summation operation.

## Examples in Context

### Neural Network Operations
```tcl
# Batch matrix multiplication in transformer
set batch_size 32
set seq_len 128
set d_model 512

# Multi-head attention computation
set Q [torch::tensor_create -data {...} -shape [list $batch_size $seq_len $d_model]]
set K [torch::tensor_create -data {...} -shape [list $batch_size $seq_len $d_model]]

# Scaled dot-product attention scores
set scores [torch::einsum "biq,bjq->bij" $Q $K]
```

### Scientific Computing
```tcl
# Tensor contraction in physics simulation
set force_field [torch::tensor_create -data {...} -shape {100 100 3}]
set displacement [torch::tensor_create -data {...} -shape {100 100 3}]

# Work calculation: dot product at each grid point
set work [torch::einsum "ijk,ijk->ij" $force_field $displacement]
```

### Data Analysis
```tcl
# Correlation matrix computation
set data [torch::tensor_create -data {...} -shape {1000 50}]  # samples x features

# Center the data (assume centered)
# Compute correlation: X^T @ X / (n-1)
set correlation [torch::einsum "ij,ik->jk" $centered_data $centered_data]
```

## Best Practices

1. **Use descriptive variable names** for clarity
2. **Comment complex equations** to explain the operation
3. **Consider memory usage** for large tensor operations
4. **Test with small examples** before scaling up
5. **Use named parameters** for complex operations
6. **Validate input dimensions** before calling einsum
7. **Choose appropriate alternatives** for simple operations

## See Also

- `torch::tensor_matmul` - Matrix multiplication
- `torch::tensor_bmm` - Batch matrix multiplication  
- `torch::tensor_transpose` - Tensor transposition
- `torch::tensor_sum` - Tensor reduction operations
- `torch::tensor_outer` - Outer product
- `torch::tensor_dot` - Dot product
- `torch::tensor_trace` - Matrix trace 