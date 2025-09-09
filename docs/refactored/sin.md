# torch::sin

Computes the sine of each element in the input tensor.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::sin tensor

# Named parameter syntax
torch::sin -input tensor
torch::sin -tensor tensor
```

## Description

The `torch::sin` command computes the element-wise sine of a tensor. The input values are interpreted as angles in radians. The output has the same shape and dtype as the input tensor.

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor | Tensor | The input tensor containing angles in radians |
| -input, -tensor | Tensor | Alternative named parameter syntax for input tensor |

## Return Value

Returns a new tensor containing the sine of each element in the input tensor.

## Examples

```tcl
# Create a tensor with angles in radians
set angles [torch::tensor_create {0.0 1.5707963267948966 3.141592653589793} float32]

# Compute sine using positional syntax
set result1 [torch::sin $angles]
;# Result: tensor([0.0000, 1.0000, 0.0000])

# Compute sine using named parameter syntax
set result2 [torch::sin -input $angles]
;# Result: tensor([0.0000, 1.0000, 0.0000])

# Using -tensor parameter
set result3 [torch::sin -tensor $angles]
;# Result: tensor([0.0000, 1.0000, 0.0000])
```

## Common Values

| Input (radians) | Output |
|----------------|--------|
| 0 | 0 |
| π/6 (0.5236) | 0.5 |
| π/4 (0.7854) | 0.7071 |
| π/3 (1.0472) | 0.8660 |
| π/2 (1.5708) | 1.0 |
| π (3.1416) | 0 |
| 3π/2 (4.7124) | -1.0 |
| 2π (6.2832) | 0 |

## Error Handling

The command will return an error in the following cases:

- Missing input tensor
- Invalid tensor name
- Invalid parameter name
- Missing parameter value

## See Also

- [torch::cos](cos.md) - Computes the cosine of a tensor
- [torch::tan](tan.md) - Computes the tangent of a tensor
- [torch::asin](asin.md) - Computes the inverse sine (arcsine) of a tensor 