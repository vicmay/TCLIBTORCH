# torch::kron

Computes the Kronecker product of two tensors.

## Syntax

### Positional Syntax (Original)
```tcl
torch::kron input other
```

### Named Parameter Syntax (New)
```tcl
torch::kron -input tensor -other tensor
```

### camelCase Alias
```tcl
torch::kron input other
```
*Note: `kron` is already camelCase, so no separate alias is needed*

## Description

The `torch::kron` command computes the Kronecker product of two tensors. The Kronecker product is a generalization of the outer product from vectors to matrices, and from matrices to higher-dimensional tensors.

For two matrices A and B, the Kronecker product A ⊗ B is a block matrix:
```
A ⊗ B = [ a₁₁B  a₁₂B  ...  a₁ₙB ]
        [ a₂₁B  a₂₂B  ...  a₂ₙB ]
        [ ...   ...   ...  ...  ]
        [ aₘ₁B  aₘ₂B  ...  aₘₙB ]
```

## Parameters

### Positional Parameters
1. **input** - First input tensor
2. **other** - Second input tensor

### Named Parameters
- **-input** - First input tensor (required)
- **-other** - Second input tensor (required)

## Return Value

Returns a tensor handle containing the Kronecker product of the two input tensors.

The output tensor dimensions are the product of the input tensor dimensions:
- If input is size (a, b) and other is size (c, d), output is size (a×c, b×d)
- For 1D tensors: if input is size (m) and other is size (n), output is size (m×n)

## Examples

### Basic Usage - Positional Syntax

```tcl
# Create two 1D tensors
set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
set t2 [torch::tensor_create -data {3.0 4.0} -dtype float32]

# Compute Kronecker product
set result [torch::kron $t1 $t2]
# Result: [3.0 4.0 6.0 8.0] (size 4)

# Check the shape
set shape [torch::tensor_shape $result]
puts "Shape: $shape"  ;# Shape: 4
```

### Basic Usage - Named Parameters

```tcl
# Create two 2D tensors
set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set t2 [torch::tensor_create -data {5.0 6.0 7.0 8.0} -shape {2 2} -dtype float32]

# Compute Kronecker product using named parameters
set result [torch::kron -input $t1 -other $t2]

# Check the resulting shape
set shape [torch::tensor_shape $result]
puts "Shape: $shape"  ;# Shape: 4 4
```

### Advanced Examples

#### Identity Matrix Kronecker Product
```tcl
# Create a vector and identity matrix
set vec [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set identity [torch::tensor_create -data {1.0} -dtype float32]

# Kronecker product with identity preserves original structure
set result [torch::kron $vec $identity]
set shape [torch::tensor_shape $result]
puts "Shape: $shape"  ;# Shape: 3 (same as original)
```

#### Mixed Dimension Kronecker Product
```tcl
# Mix 1D and 2D tensors
set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32]
set t2 [torch::tensor_create -data {3.0 4.0 5.0 6.0} -shape {2 2} -dtype float32]

set result [torch::kron -input $t1 -other $t2]
set shape [torch::tensor_shape $result]
puts "Shape: $shape"  ;# Shape: 2 4
```

## Mathematical Properties

### Dimension Rules
- **1D ⊗ 1D**: If A is (m) and B is (n), result is (m×n)
- **2D ⊗ 2D**: If A is (m,n) and B is (p,q), result is (m×p, n×q)
- **General**: For tensors of any dimension, output dimensions are products of input dimensions

### Important Properties
1. **Non-commutative**: `kron(A, B) ≠ kron(B, A)` in general
2. **Associative**: `kron(kron(A, B), C) = kron(A, kron(B, C))`
3. **Distributive**: `kron(A + B, C) = kron(A, C) + kron(B, C)`
4. **Scalar multiplication**: `kron(cA, B) = c × kron(A, B)`

## Error Handling

The command will return an error in the following cases:

1. **Missing parameters**: Not enough arguments provided
2. **Invalid tensor handles**: Non-existent tensor names
3. **Unknown parameters**: Invalid parameter names in named syntax

### Common Error Messages

```tcl
# Missing arguments
torch::kron
# Error: Usage: kron input other

# Invalid tensor handle
torch::kron invalid_tensor $t2
# Error: Invalid input tensor

# Unknown parameter
torch::kron -input $t1 -other $t2 -invalid param
# Error: Unknown parameter: -invalid
```

## Performance Considerations

1. **Memory usage**: Kronecker products can create very large tensors
   - For (m×n) ⊗ (p×q) → (m×p, n×q), memory scales as m×n×p×q
   
2. **Computational complexity**: O(mnpq) for the tensor sizes above

3. **Data type**: Results preserve the precision of input tensors

## Compatibility

- **LibTorch version**: Compatible with LibTorch 1.9+
- **Data types**: Supports all numeric tensor types (float32, float64, int32, int64, etc.)
- **Devices**: Works on CPU and GPU tensors
- **Backward compatibility**: Positional syntax fully supported

## See Also

- [`torch::outer`](outer.md) - Outer product of vectors
- [`torch::bmm`](bmm.md) - Batch matrix multiplication
- [`torch::matmul`](matmul.md) - Matrix multiplication
- [`torch::einsum`](einsum.md) - Einstein summation notation

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::kron $tensor1 $tensor2]

# New named parameter syntax
set result [torch::kron -input $tensor1 -other $tensor2]
```

### Parameter Order Independence

With named parameters, you can specify arguments in any order:

```tcl
# Both are equivalent
set result1 [torch::kron -input $t1 -other $t2]
set result2 [torch::kron -other $t2 -input $t1]
```

## Notes

- The Kronecker product is fundamental in quantum computing, signal processing, and tensor analysis
- Consider memory requirements before computing Kronecker products of large tensors
- The operation is not commutative: order matters for the tensors being multiplied
- Results are always computed in the same dtype as the input tensors 