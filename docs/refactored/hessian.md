# torch::hessian

Compute the Hessian matrix of a function with respect to its inputs.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::hessian -func function_name -inputs tensor
torch::hessian -function function_name -input tensor
```

### Positional Parameters (Legacy)
```tcl
torch::hessian function_name tensor
```

## Parameters

### Named Parameters
- **`-func`** or **`-function`** (string): Function handle or name for which to compute the Hessian
- **`-inputs`** or **`-input`** (tensor): Input tensor with respect to which the Hessian is computed

### Positional Parameters
1. **`function_name`** (string): Function handle or name for which to compute the Hessian
2. **`tensor`** (tensor): Input tensor with respect to which the Hessian is computed

## Returns

A tensor containing the Hessian matrix. The returned tensor has shape `[n, n]` where `n` is the number of elements in the input tensor.

## Description

The `torch::hessian` command computes the Hessian matrix, which is the square matrix of second-order partial derivatives of a scalar-valued function. The Hessian matrix provides information about the local curvature of the function.

**Note**: The current implementation returns an identity matrix of appropriate size as a placeholder. In a complete implementation, this would compute the actual Hessian matrix using automatic differentiation.

## Examples

### Basic Usage with Named Parameters

```tcl
# Create input tensor
set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# Compute Hessian using named parameters
set hessian_result [torch::hessian -func "quadratic_func" -inputs $x]

# Check result shape
set shape [torch::tensor_shape $hessian_result]
puts "Hessian shape: $shape"  ;# Output: Hessian shape: 3 3
```

### Using Alternative Parameter Names

```tcl
# Same functionality with alternative parameter names
set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
set hessian_result [torch::hessian -function "quadratic_func" -input $x]
```

### Legacy Positional Syntax

```tcl
# Backward compatibility with positional parameters
set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
set hessian_result [torch::hessian "quadratic_func" $x]
```

### Different Input Sizes

```tcl
# 1D tensor (single element)
set x [torch::tensor_create {1.0} float32 cpu true]
set hessian_result [torch::hessian -func "func" -inputs $x]
# Result shape: [1, 1]

# 2D tensor (4 elements)
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set hessian_result [torch::hessian -func "func" -inputs $x]
# Result shape: [4, 4]
```

### Error Handling

```tcl
# Missing parameters
if {[catch {torch::hessian -func "func"} error]} {
    puts "Error: $error"
}

# Invalid parameter name
if {[catch {torch::hessian -func "func" -inputs $x -unknown "value"} error]} {
    puts "Error: $error"
}
```

## Parameter Validation

The command validates that:
- Both function name and input tensor are provided
- Named parameters come in pairs
- Parameter names are recognized
- Input tensor handle is valid

## Mathematical Background

The Hessian matrix H of a function f(x) with respect to inputs x is defined as:

```
H[i,j] = ∂²f/∂x[i]∂x[j]
```

Where:
- `f` is a scalar-valued function
- `x` is the input vector
- `H[i,j]` is the second partial derivative with respect to variables `x[i]` and `x[j]`

The Hessian provides information about:
- Local curvature of the function
- Convexity/concavity properties
- Critical point classification (minimum, maximum, saddle point)

## Implementation Notes

- The current implementation returns an identity matrix as a placeholder
- In a complete implementation, this would use automatic differentiation to compute actual second derivatives
- The function requires gradients to be enabled for the input tensor
- The input tensor should have `requires_grad=true` for proper gradient computation

## Performance Considerations

- Computational complexity: O(n²) where n is the number of input elements
- Memory usage: O(n²) for storing the Hessian matrix
- For large input tensors, consider using approximate methods or computing only diagonal elements

## See Also

- [`torch::grad`](grad.md) - Compute gradients (first derivatives)
- [`torch::jacobian`](jacobian.md) - Compute Jacobian matrix
- [`torch::vjp`](vjp.md) - Vector-Jacobian product
- [`torch::jvp`](jvp.md) - Jacobian-vector product
- [`torch::grad_check`](grad_check.md) - Gradient checking utilities

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::hessian "function_name" $tensor

# New named parameter syntax
torch::hessian -func "function_name" -inputs $tensor
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make the code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to modify and extend
4. **Error Prevention**: Reduces parameter order mistakes

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters
- **v2.0**: Added camelCase command alias support
- **v2.0**: Enhanced error handling and validation 