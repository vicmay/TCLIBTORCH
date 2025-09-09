# torch::rrelu

## Description
Applies the Randomized Leaky Rectified Linear Unit (RReLU) function element-wise to the input tensor.

RReLU is defined as:
- `f(x) = x` if `x >= 0`
- `f(x) = a * x` if `x < 0`, where `a` is randomly sampled from uniform distribution `U(lower, upper)`

## Syntax

### Original (Positional Parameters)
```tcl
torch::rrelu tensor ?lower? ?upper?
```

### New (Named Parameters)
```tcl
torch::rrelu -input tensor ?-lower lower? ?-upper upper?
torch::rrelu -tensor tensor ?-lower lower? ?-upper upper?
```

### CamelCase Alias
```tcl
torch::rRelu tensor ?lower? ?upper?
torch::rRelu -input tensor ?-lower lower? ?-upper upper?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor`/`input` | tensor | required | Input tensor |
| `lower` | float | 0.125 | Lower bound of the uniform distribution |
| `upper` | float | 0.333333 | Upper bound of the uniform distribution |

## Return Value
Returns a new tensor with the RReLU function applied element-wise.

## Examples

### Basic Usage
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set result [torch::rrelu $input]
```

### With Custom Bounds
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set result [torch::rrelu $input 0.1 0.3]
```

### Named Parameters
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set result [torch::rrelu -input $input -lower 0.1 -upper 0.3]
```

### Different Parameter Order
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set result [torch::rrelu -upper 0.3 -input $input -lower 0.1]
```

### CamelCase Alias
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set result [torch::rRelu -input $input -lower 0.1 -upper 0.3]
```

## Error Handling
- Throws error if required tensor parameter is missing
- Throws error if tensor handle is invalid
- Throws error for unknown parameters in named syntax
- Throws error if parameter values are missing
- Validates that `lower <= upper`

## Notes
- RReLU introduces randomness during training, which can help with regularization
- The randomness is applied only to negative values
- Default bounds (1/8, 1/3) are commonly used in practice
- Both `input` and `tensor` parameter names are supported for flexibility

## Mathematical Properties
- For positive inputs: `rrelu(x) = x`
- For negative inputs: `rrelu(x) = a * x` where `a ~ U(lower, upper)`
- The function is piecewise linear
- Provides gradient flow for negative inputs unlike standard ReLU

## Migration Guide
Existing code continues to work unchanged:

```tcl
# Old code (continues to work)
set result [torch::rrelu $input]
set result [torch::rrelu $input 0.1]
set result [torch::rrelu $input 0.1 0.3]

# New named parameter syntax
set result [torch::rrelu -input $input]
set result [torch::rrelu -input $input -lower 0.1]
set result [torch::rrelu -input $input -lower 0.1 -upper 0.3]

# CamelCase alias
set result [torch::rRelu $input 0.1 0.3]
```

## See Also
- [torch::relu](relu.md) - Standard ReLU activation
- [torch::leaky_relu](leaky_relu.md) - Leaky ReLU activation
- [torch::elu](elu.md) - Exponential Linear Unit
- [torch::prelu](prelu.md) - Parametric ReLU 