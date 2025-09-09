# torch::jvp

Computes the Jacobian-vector product (JVP) of a function with respect to inputs.

## Syntax

### Positional Syntax (Original)
```tcl
torch::jvp func inputs v
```

### Named Parameter Syntax (New)
```tcl
torch::jvp -func function -inputs inputs -v vector
```

### camelCase Alias
```tcl
torch::jvp func inputs v
```
*Note: `jvp` is already camelCase, so no separate alias is needed*

## Description

The `torch::jvp` command computes the Jacobian-vector product (JVP), which is a fundamental operation in automatic differentiation. The JVP computes the directional derivative of a function in the direction of a given vector.

Given a function `f`, inputs `x`, and a vector `v`, the JVP computes `J(f)(x) * v`, where `J(f)(x)` is the Jacobian matrix of `f` at `x`.

## Parameters

### Positional Parameters
- `func` - Function name or handle (currently used for identification)
- `inputs` - Input tensor handle for the function evaluation point
- `v` - Vector tensor handle for the direction of differentiation

### Named Parameters
- `-func` or `-function` - Function name or handle
- `-inputs` or `-input` - Input tensor handle
- `-v` or `-vector` - Vector tensor handle

## Return Value

Returns a tensor handle containing the result of the Jacobian-vector product.

## Examples

### Basic Usage
```tcl
# Create input tensors
set inputs [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set v [torch::tensor_create -data {1.0 1.0 1.0} -dtype float32]

# Positional syntax
set result1 [torch::jvp "my_function" $inputs $v]

# Named parameter syntax
set result2 [torch::jvp -func "my_function" -inputs $inputs -v $v]
```

### Parameter Order Independence
```tcl
# Named parameters can be specified in any order
set result1 [torch::jvp -func "f" -inputs $inputs -v $v]
set result2 [torch::jvp -v $v -func "f" -inputs $inputs]
set result3 [torch::jvp -inputs $inputs -v $v -func "f"]
```

### Alternative Parameter Names
```tcl
# Alternative parameter names for flexibility
set result [torch::jvp -function "test" -input $inputs -vector $v]
```

### Different Data Types
```tcl
# Float64 tensors
set inputs [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64]
set v [torch::tensor_create -data {1.0 1.0 1.0} -dtype float64]
set result [torch::jvp -func "test" -inputs $inputs -v $v]
```

### Mathematical Operations
```tcl
# JVP with zero vector
set inputs [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set zero_v [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32]
set result [torch::jvp "func" $inputs $zero_v]

# JVP with negative values
set inputs [torch::tensor_create -data {-1.0 2.0 -3.0} -dtype float32]
set v [torch::tensor_create -data {1.0 -1.0 1.0} -dtype float32]
set result [torch::jvp "func" $inputs $v]
```

## Error Handling

The command provides comprehensive error handling:

### Missing Arguments
```tcl
# Error: Missing arguments
catch {torch::jvp} msg
# Returns: Usage: jvp func inputs v

# Error: Insufficient arguments
catch {torch::jvp "func"} msg
# Returns: Usage: jvp func inputs v
```

### Invalid Parameters
```tcl
# Error: Unknown parameter
catch {torch::jvp -func "f" -inputs $inputs -v $v -unknown "value"} msg
# Returns: Unknown parameter: -unknown

# Error: Missing parameter value
catch {torch::jvp -func "test" -inputs} msg
# Returns: Usage: jvp -func function -inputs inputs -v vector
```

### Invalid Tensor Handles
```tcl
# Error: Invalid tensor handle
catch {torch::jvp "func" invalid_tensor $v} msg
# Returns: Error in jvp: [tensor error details]
```

## Mathematical Background

The Jacobian-vector product is a key operation in automatic differentiation:

- **Forward-mode AD**: JVP computes derivatives in the direction of the input vector
- **Efficiency**: More efficient than computing the full Jacobian when only directional derivatives are needed
- **Chain Rule**: Essential for computing gradients through composite functions

## Implementation Notes

- The current implementation uses matrix multiplication between inputs and vector
- The function parameter is used for identification but doesn't affect the computation
- Both input tensors must have compatible shapes for matrix multiplication
- The result tensor type matches the input tensor types

## Performance Considerations

- **Memory**: JVP requires less memory than computing the full Jacobian
- **Computation**: More efficient for computing directional derivatives
- **Scalability**: Performance scales with vector size rather than full Jacobian size

## Related Commands

- [`torch::vjp`](vjp.md) - Vector-Jacobian product (reverse-mode AD)
- [`torch::jacobian`](jacobian.md) - Full Jacobian matrix computation
- [`torch::grad`](grad.md) - Gradient computation
- [`torch::hessian`](hessian.md) - Hessian matrix computation

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Named Parameters**: New syntax provides better readability and flexibility
- **Type Safety**: Comprehensive parameter validation and error handling
- **Cross-Platform**: Works on all supported platforms (CPU/CUDA)

## See Also

- [Automatic Differentiation Guide](../guides/autodiff.md)
- [Tensor Operations](../guides/tensor_ops.md)
- [PyTorch JVP Documentation](https://pytorch.org/docs/stable/generated/torch.autograd.functional.jvp.html) 