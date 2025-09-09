# torch::rsqrt

## Description
Computes the reciprocal of the square root of each element in the input tensor element-wise.

The function computes: `rsqrt(x) = 1 / sqrt(x)`

## Syntax

### Original (Positional Parameters)
```tcl
torch::rsqrt tensor
```

### New (Named Parameters)
```tcl
torch::rsqrt -input tensor
torch::rsqrt -tensor tensor
```

### CamelCase Alias
```tcl
torch::rSqrt tensor
torch::rSqrt -input tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor`/`input` | tensor | Input tensor (must contain positive values) |

## Return Value
Returns a new tensor with the reciprocal square root applied element-wise.

## Examples

### Basic Usage
```tcl
set input [torch::ones [list 2 3] float32]
set result [torch::rsqrt $input]
```

### Named Parameters
```tcl
set input [torch::ones [list 2 3] float32]
set result [torch::rsqrt -input $input]
```

### Alternative Parameter Name
```tcl
set input [torch::ones [list 2 3] float32]
set result [torch::rsqrt -tensor $input]
```

### CamelCase Alias
```tcl
set input [torch::ones [list 2 3] float32]
set result [torch::rSqrt -input $input]
```

### Mathematical Example
```tcl
# Create tensor with value 4.0
set input [torch::full [list 1] 4.0 float32]
set result [torch::rsqrt $input]
# Result will be 0.5 (since 1/sqrt(4) = 1/2 = 0.5)
set value [torch::tensor_item $result]
puts "rsqrt(4.0) = $value"
```

## Error Handling
- Throws error if required tensor parameter is missing
- Throws error if tensor handle is invalid
- Throws error for unknown parameters in named syntax
- Throws error if parameter values are missing
- Runtime error for negative or zero values (mathematically undefined)

## Notes
- Input tensor must contain positive values
- The function is mathematically equivalent to `1 / sqrt(x)`
- Commonly used in normalization operations
- More numerically efficient than computing `1 / sqrt(x)` separately
- Both `input` and `tensor` parameter names are supported for flexibility

## Mathematical Properties
- `rsqrt(x) = 1 / sqrt(x)` for `x > 0`
- `rsqrt(1) = 1`
- `rsqrt(4) = 0.5`
- `rsqrt(0.25) = 2`
- Undefined for `x <= 0`

## Common Use Cases
1. **Normalization**: Layer normalization, batch normalization
2. **Distance calculations**: Euclidean distance normalization
3. **Optimization**: Fast inverse square root computations
4. **Graphics**: Vector normalization in 3D graphics

## Migration Guide
Existing code continues to work unchanged:

```tcl
# Old code (continues to work)
set result [torch::rsqrt $input]

# New named parameter syntax
set result [torch::rsqrt -input $input]
set result [torch::rsqrt -tensor $input]

# CamelCase alias
set result [torch::rSqrt $input]
```

## See Also
- [torch::sqrt](sqrt.md) - Square root function
- [torch::pow](pow.md) - Power function
- [torch::reciprocal](reciprocal.md) - Reciprocal function
- [torch::norm](norm.md) - Tensor normalization 