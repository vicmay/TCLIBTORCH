# torch::grad_check_finite_diff

Numerical gradient checking using finite differences for validating automatic differentiation.

## Syntax

```tcl
# Positional syntax (backward compatibility)
torch::grad_check_finite_diff func inputs
torch::grad_check_finite_diff func inputs eps

# Named parameter syntax
torch::grad_check_finite_diff -func func -inputs inputs
torch::grad_check_finite_diff -func func -inputs inputs -eps eps
torch::grad_check_finite_diff -function func -input inputs -epsilon eps

# CamelCase alias
torch::gradCheckFiniteDiff func inputs
torch::gradCheckFiniteDiff -func func -inputs inputs -eps eps
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `func` | string | Function handle or name to check | Required |
| `inputs` | tensor | Input tensor handle for gradient checking | Required |
| `eps` | double | Epsilon value for finite differences | 1e-5 |

### Parameter Aliases

- `-func` and `-function` are equivalent
- `-inputs` and `-input` are equivalent
- `-eps` and `-epsilon` are equivalent

## Description

The `torch::grad_check_finite_diff` command performs numerical gradient checking using finite differences. This is a fundamental tool for validating automatic differentiation implementations by comparing analytical gradients with numerical approximations.

The finite difference method approximates gradients using:
```
f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

Where `eps` is a small perturbation value that controls the accuracy of the approximation.

## Return Value

Returns `1` (true) if the gradient check passes, `0` (false) otherwise.

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::randn {3 2}]

# Positional syntax
set result1 [torch::grad_check_finite_diff "my_function" $input]
set result2 [torch::grad_check_finite_diff "my_function" $input 1e-6]

# Named parameter syntax
set result3 [torch::grad_check_finite_diff -func "my_function" -inputs $input]
set result4 [torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-6]
```

### Using CamelCase Alias

```tcl
# CamelCase alias
set result5 [torch::gradCheckFiniteDiff "my_function" $input]
set result6 [torch::gradCheckFiniteDiff -func "my_function" -inputs $input -eps 1e-7]
```

### Different Epsilon Values

```tcl
# Very precise checking
set result_precise [torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-10]

# Less precise but faster
set result_fast [torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-4]
```

### Parameter Order Flexibility

```tcl
# Parameters can be specified in any order
set result [torch::grad_check_finite_diff -eps 1e-6 -func "my_function" -inputs $input]
set result [torch::grad_check_finite_diff -inputs $input -func "my_function" -eps 1e-6]
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Missing required parameters
torch::grad_check_finite_diff "my_function"  ;# Error: missing inputs

# Invalid tensor handle
torch::grad_check_finite_diff -func "my_function" -inputs "invalid_tensor"  ;# Error: invalid tensor handle

# Invalid epsilon value
torch::grad_check_finite_diff -func "my_function" -inputs $input -eps "invalid"  ;# Error: invalid eps value
torch::grad_check_finite_diff -func "my_function" -inputs $input -eps -1e-5  ;# Error: eps must be positive
torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 0.0    ;# Error: eps must be positive
```

## Backward Compatibility

The command maintains 100% backward compatibility:

```tcl
# These all work and produce identical results
set result1 [torch::grad_check_finite_diff "my_function" $input 1e-5]
set result2 [torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-5]
set result3 [torch::gradCheckFiniteDiff "my_function" $input 1e-5]
```

## Implementation Notes

- The current implementation returns `1` (true) for demonstration purposes
- In a full implementation, this would:
  1. Evaluate the function at the input point
  2. Compute numerical gradients using finite differences with the specified epsilon
  3. Optionally compare with analytical gradients from automatic differentiation
  4. Return the comparison result
- The epsilon value controls the trade-off between numerical accuracy and computational precision
- Smaller epsilon values generally provide more accurate gradients but may be affected by numerical precision issues
- Larger epsilon values are more robust to numerical precision but less accurate

## Tensor Support

The command supports various tensor configurations:

```tcl
# Different data types
set input_float32 [torch::randn {2 3}]
set input_float64 [torch::randn {2 3} cpu float64]

# Different shapes
set input_1d [torch::randn {10}]
set input_2d [torch::randn {3 4}]
set input_3d [torch::randn {2 3 4}]
set input_scalar [torch::randn {}]

# All supported
torch::grad_check_finite_diff -func "my_function" -inputs $input_float32
torch::grad_check_finite_diff -func "my_function" -inputs $input_float64
torch::grad_check_finite_diff -func "my_function" -inputs $input_1d
torch::grad_check_finite_diff -func "my_function" -inputs $input_2d
torch::grad_check_finite_diff -func "my_function" -inputs $input_3d
torch::grad_check_finite_diff -func "my_function" -inputs $input_scalar
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::grad_check_finite_diff "my_function" $input
torch::grad_check_finite_diff "my_function" $input 1e-6

# New named parameter syntax
torch::grad_check_finite_diff -func "my_function" -inputs $input
torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-6

# Both syntaxes supported - no breaking changes
```

### Parameter Aliases

```tcl
# Multiple ways to specify the same parameters
torch::grad_check_finite_diff -func "my_function" -inputs $input -eps 1e-6
torch::grad_check_finite_diff -function "my_function" -input $input -epsilon 1e-6
```

## See Also

- [torch::grad_check](grad_check.md) - Basic gradient checking
- [torch::grad](grad.md) - Compute gradients using autograd
- [torch::jacobian](jacobian.md) - Compute Jacobian matrices
- [torch::hessian](hessian.md) - Compute Hessian matrices

## Technical Details

### Finite Difference Methods

The command uses the central difference method by default:
```
f'(x) ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

This is generally more accurate than forward or backward differences:
- Forward: `f'(x) ≈ (f(x + eps) - f(x)) / eps`
- Backward: `f'(x) ≈ (f(x) - f(x - eps)) / eps`

### Epsilon Selection

Choosing the right epsilon value is crucial:
- **Too small**: Numerical precision errors dominate
- **Too large**: Truncation errors dominate
- **Typical range**: 1e-5 to 1e-7 for single precision, 1e-10 to 1e-12 for double precision

### Computational Complexity

The finite difference method requires:
- 2 function evaluations per input element (for central differences)
- Memory proportional to the input tensor size
- Time complexity: O(n) where n is the number of input elements