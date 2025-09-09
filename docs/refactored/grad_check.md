# torch::grad_check

Numerical gradient checking for validating automatic differentiation.

## Syntax

```tcl
# Positional syntax (backward compatibility)
torch::grad_check func inputs

# Named parameter syntax
torch::grad_check -func func -inputs inputs
torch::grad_check -function func -input inputs

# CamelCase alias
torch::gradCheck func inputs
torch::gradCheck -func func -inputs inputs
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `func` | string | Function handle or name to check | Required |
| `inputs` | tensor | Input tensor handle for gradient checking | Required |

### Parameter Aliases

- `-func` and `-function` are equivalent
- `-inputs` and `-input` are equivalent

## Description

The `torch::grad_check` command performs numerical gradient checking to validate automatic differentiation. It compares analytical gradients (computed using autograd) with numerical gradients (computed using finite differences) to ensure the correctness of gradient computations.

This is particularly useful for:
- Debugging custom functions and operations
- Validating gradient implementations
- Ensuring numerical stability of gradient computations

## Returns

Returns `1` (true) if the gradient check passes, `0` (false) otherwise.

## Examples

### Basic Usage

```tcl
# Create a tensor that requires gradients
set x [torch::randn -shape {2 3} -requiresGrad true]

# Positional syntax (backward compatibility)
set result [torch::grad_check "my_function" $x]
puts "Gradient check passed: $result"

# Named parameter syntax
set result [torch::grad_check -func "my_function" -inputs $x]
puts "Gradient check passed: $result"

# CamelCase alias
set result [torch::gradCheck -function "my_function" -input $x]
puts "Gradient check passed: $result"
```

### Different Tensor Types

```tcl
# Float64 tensor
set x [torch::randn -shape {3 2} -dtype float64 -requiresGrad true]
set result [torch::grad_check -func "my_function" -inputs $x]

# 1D tensor
set x [torch::randn -shape {10} -requiresGrad true]
set result [torch::grad_check -func "my_function" -inputs $x]

# Scalar tensor
set x [torch::randn -shape {} -requiresGrad true]
set result [torch::grad_check -func "my_function" -inputs $x]
```

### Error Handling

```tcl
# Handle potential errors
if {[catch {torch::grad_check -func "my_function" -inputs $x} result]} {
    puts "Error in gradient check: $result"
} else {
    puts "Gradient check result: $result"
}
```

## Implementation Notes

The current implementation returns `1` (true) by default. A full implementation would:

1. **Evaluate the function** at the input point
2. **Compute analytical gradients** using PyTorch's autograd system
3. **Compute numerical gradients** using finite differences
4. **Compare the gradients** for numerical accuracy within a specified tolerance
5. **Return the comparison result**

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (still supported)
torch::grad_check "my_function" $tensor

# NEW (recommended)
torch::grad_check -func "my_function" -inputs $tensor
```

### From snake_case to camelCase

```tcl
# OLD (still supported)
torch::grad_check "my_function" $tensor

# NEW (recommended)
torch::gradCheck -func "my_function" -inputs $tensor
```

## Backward Compatibility

- ✅ All existing positional syntax continues to work
- ✅ Original command name `torch::grad_check` remains available
- ✅ Same return values and behavior
- ✅ No breaking changes

## See Also

- [`torch::grad_check_finite_diff`](grad_check_finite_diff.md) - Gradient checking with finite differences
- [`torch::grad`](grad.md) - Compute gradients using autograd
- [`torch::jacobian`](jacobian.md) - Compute Jacobian matrix
- [`torch::hessian`](hessian.md) - Compute Hessian matrix

## Error Handling

The command will return an error in the following cases:

- Invalid or missing function parameter
- Invalid or missing inputs tensor handle
- Unknown parameter names
- Malformed parameter syntax

All errors include descriptive messages to help identify the issue. 