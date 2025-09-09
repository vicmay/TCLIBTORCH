# torch::tensor_matmul / torch::tensorMatmul

**Matrix Multiplication Operation**

Performs matrix multiplication between two tensors. Supports both batch and non-batch operations.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::tensor_matmul -input tensor1 -other tensor2
torch::tensorMatmul -input tensor1 -other tensor2  # camelCase alias
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::tensor_matmul tensor1 tensor2
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` | tensor | Yes | First input tensor |
| `-other` | tensor | Yes | Second input tensor |

### Positional Parameter Order (Legacy)
1. `tensor1` - First input tensor
2. `tensor2` - Second input tensor

## Description

Performs matrix multiplication between two tensors. The behavior depends on the dimensions:

- **1D x 1D**: Dot product (scalar result)
- **1D x 2D**: Vector-matrix multiplication 
- **2D x 1D**: Matrix-vector multiplication
- **2D x 2D**: Matrix-matrix multiplication
- **Batch operations**: Supports batched matrix multiplication for higher dimensional tensors

The operation follows standard matrix multiplication rules where the last dimension of the first tensor must match the second-to-last dimension of the second tensor.

## Returns

Returns a new tensor containing the result of the matrix multiplication.

## Examples

### Basic Matrix Multiplication
```tcl
# Create two 2x2 matrices
set a [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu]
set a_2d [torch::tensor_reshape $a {2 2}]
set b [torch::tensor_create -data {5.0 6.0 7.0 8.0} -dtype float32 -device cpu] 
set b_2d [torch::tensor_reshape $b {2 2}]

# New syntax
set result [torch::tensor_matmul -input $a_2d -other $b_2d]

# Legacy syntax
set result [torch::tensor_matmul $a_2d $b_2d]

# camelCase alias
set result [torch::tensorMatmul -input $a_2d -other $b_2d]
```

### Vector-Matrix Multiplication
```tcl
# Create 1D vector and 2D matrix
set vector [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu]
set matrix [torch::tensor_create -data {3.0 4.0 5.0 6.0} -dtype float32 -device cpu]
set matrix_2d [torch::tensor_reshape $matrix {2 2}]

# New syntax
set result [torch::tensor_matmul -input $vector -other $matrix_2d]
```

### Batch Matrix Multiplication
```tcl
# Create batch of matrices (batch_size=2, 2x2 each)
set batch_a [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32 -device cpu]
set batch_a_3d [torch::tensor_reshape $batch_a {2 2 2}]
set batch_b [torch::tensor_create -data {9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0} -dtype float32 -device cpu]
set batch_b_3d [torch::tensor_reshape $batch_b {2 2 2}]

# Batch matrix multiplication
set result [torch::tensor_matmul -input $batch_a_3d -other $batch_b_3d]
```

## Mathematical Notes

For matrices A (m×k) and B (k×n), the result C will be (m×n) where:
```
C[i,j] = Σ(A[i,k] * B[k,j]) for k=0 to k-1
```

## Error Handling

The function validates:
- Both inputs are valid tensors
- Tensor dimensions are compatible for matrix multiplication
- Data types are compatible

```tcl
# Error: Incompatible dimensions
catch {torch::tensor_matmul -input $tensor_2x3 -other $tensor_4x2} error
puts $error  # Will show dimension mismatch error

# Error: Invalid tensor
catch {torch::tensor_matmul -input "invalid" -other $tensor} error  
puts $error  # Will show invalid tensor error
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set result [torch::tensor_matmul $a $b]
```

**After:**
```tcl
set result [torch::tensor_matmul -input $a -other $b]
# or using camelCase
set result [torch::tensorMatmul -input $a -other $b]
```

### Benefits of New Syntax
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Safety**: Reduced risk of parameter mix-ups
- **Consistency**: Follows TCL best practices

## Performance Notes

- Matrix multiplication is optimized for the underlying device (CPU/CUDA)
- Batch operations are more efficient than individual matrix multiplications
- Consider using appropriate data types (float32 vs float64) based on precision needs

## See Also

- [torch::tensor_bmm](tensor_bmm.md) - Batch matrix multiplication (explicit batch dimension)
- [torch::tensor_add](tensor_add.md) - Element-wise tensor addition
- [torch::tensor_mul](tensor_mul.md) - Element-wise tensor multiplication

## Implementation Status

- ✅ **Dual Syntax**: Supports both positional and named parameters
- ✅ **camelCase Alias**: `torch::tensorMatmul` available  
- ✅ **Tests**: Comprehensive test suite in `tests/refactored/tensor_matmul_test.tcl`
- ✅ **Documentation**: Complete API documentation 