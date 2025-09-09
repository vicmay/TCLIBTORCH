# torch::outer / torch::Outer

Computes the outer product of two tensors.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::outer input other
torch::Outer input other  ;# camelCase alias
```

### Modern Syntax (Named Parameters)
```tcl
torch::outer -input tensor -other tensor
torch::Outer -input tensor -other tensor  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| input/tensor | tensor | The first input tensor |
| other | tensor | The second input tensor |

## Description

The `outer` command computes the outer product of two tensors. For vectors, if `input` is a tensor of size n and `other` is a tensor of size m, the result is a tensor of size (n × m) containing the outer product.

The outer product is computed as:
```
result[i][j] = input[i] * other[j]
```

This operation is useful in various linear algebra applications, including:
- Creating projection matrices
- Computing dyadic products
- Generating transformation matrices
- Neural network weight initialization

## Return Value

Returns a tensor handle for the resulting outer product tensor. The shape of the output tensor is `[*input.shape, *other.shape]`.

## Examples

### Using Legacy Syntax
```tcl
# Create input vectors
set v1 [torch::tensor_create {1.0 2.0 3.0} float32]
set v2 [torch::tensor_create {4.0 5.0} float32]

# Compute outer product
set result [torch::outer $v1 $v2]
# Result is a 3x2 matrix:
# [[ 4.0  5.0]
#  [ 8.0 10.0]
#  [12.0 15.0]]
```

### Using Modern Syntax
```tcl
# Using named parameters
set result [torch::outer -input $v1 -other $v2]

# Using camelCase alias
set result [torch::Outer -input $v1 -other $v2]
```

## Error Conditions

The command will return an error in the following cases:
- If required parameters are missing
- If any of the tensor handles are invalid
- If the input tensors have incompatible types

## See Also

- [torch::dot](dot.md) - Compute the dot product of two tensors
- [torch::cross](cross.md) - Compute the cross product of two tensors
- [torch::matmul](matmul.md) - Matrix multiplication 