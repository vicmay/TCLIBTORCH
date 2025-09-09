# torch::vjp

Computes the Vector-Jacobian Product (VJP) of a function.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::vjp func inputs v
```

### Named Parameter Syntax  
```tcl
torch::vjp -func function -inputs tensor -v vector
```

### CamelCase Alias
```tcl
torch::vectorJacobianProduct func inputs v
torch::vectorJacobianProduct -func function -inputs tensor -v vector
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | string | **Required.** Function handle or name for the computation |
| `inputs` | string | **Required.** Name of the input tensor |
| `v` | string | **Required.** Name of the vector tensor for VJP computation |

## Returns

Returns a new tensor handle containing the computed Vector-Jacobian Product.

## Description

The `torch::vjp` command computes the Vector-Jacobian Product, which is a fundamental operation in reverse-mode automatic differentiation. VJP is used to efficiently compute gradients in backpropagation algorithms and is essential for:

- **Automatic differentiation**: Computing gradients of scalar functions
- **Neural network training**: Backpropagation through computational graphs
- **Optimization algorithms**: Gradient-based optimization methods
- **Scientific computing**: Sensitivity analysis and parameter estimation

The VJP operation computes `v^T * J` where:
- `v` is a vector (typically representing upstream gradients)
- `J` is the Jacobian matrix of the function with respect to its inputs

## Examples

### Basic Usage
```tcl
# Create input tensor and vector
set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
set v [torch::tensor_create {1.0 1.0}]

# Compute VJP
set result [torch::vjp "my_function" $inputs $v]
puts "VJP result: [torch::tensor_to_list $result]"
```

### Named Parameter Syntax
```tcl
# Create tensors for VJP computation
set inputs [torch::tensor_create {{2.0 3.0} {4.0 5.0}}]
set vector [torch::tensor_create {0.5 1.5}]

# Use named parameters
set vjp_result [torch::vjp -func "gradient_func" -inputs $inputs -v $vector]
puts "VJP shape: [torch::tensor_shape $vjp_result]"

# Alternative parameter names
set vjp_result2 [torch::vjp -function "test_func" -input $inputs -vector $vector]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
set v [torch::tensor_create {1.0 1.0}]

# Positional syntax with camelCase
set result [torch::vectorJacobianProduct "my_func" $inputs $v]

# Named parameter syntax with camelCase
set result2 [torch::vectorJacobianProduct -func "my_func" -inputs $inputs -v $v]
```

### Different Tensor Dimensions
```tcl
# 1D tensors
set inputs_1d [torch::tensor_create {1.0 2.0 3.0}]
set v_1d [torch::tensor_create {1.0 1.0 1.0}]
set vjp_1d [torch::vjp "func_1d" $inputs_1d $v_1d]

# 3D tensors
set inputs_3d [torch::tensor_create {{{1.0 2.0} {3.0 4.0}} {{5.0 6.0} {7.0 8.0}}}]
set v_3d [torch::tensor_create {{1.0 1.0} {1.0 1.0}}]
set vjp_3d [torch::vjp "func_3d" $inputs_3d $v_3d]
```

### Practical Example: Gradient Computation
```tcl
# Simulate a simple neural network layer
set weights [torch::tensor_create {{0.5 -0.3} {0.2 0.8}}]
set upstream_grad [torch::tensor_create {1.0 -1.0}]

# Compute VJP for weight gradients
set weight_grad [torch::vjp "linear_layer" $weights $upstream_grad]
puts "Weight gradients: [torch::tensor_to_list $weight_grad]"
```

## Error Handling

The command will raise an error in the following cases:

- **Invalid tensor**: If any input tensor name doesn't exist
- **Dimension mismatch**: If vector and input tensor dimensions are incompatible for matrix multiplication
- **Missing required parameter**: If any required parameter is not provided
- **Unknown parameter**: If an unrecognized parameter name is used

### Error Examples
```tcl
# Error: Invalid tensor
catch {torch::vjp "func" "nonexistent" "invalid"} error
puts $error  ;# "Invalid tensor"

# Error: Missing required parameter  
catch {torch::vjp -func "test" -inputs "tensor1"} error
puts $error  ;# "Required parameters missing: -func, -inputs, and -v"

# Error: Dimension mismatch
set inputs [torch::tensor_create {1.0 2.0 3.0}]
set v [torch::tensor_create {1.0}]  ;# Wrong size
catch {torch::vjp "func" $inputs $v} error
puts $error  ;# "inconsistent tensor size"
```

## Mathematical Details

The Vector-Jacobian Product is computed as:

**VJP(f, x, v) = v^T * J_f(x)**

Where:
- `f` is the function
- `x` is the input tensor (primal)
- `v` is the vector (cotangent)
- `J_f(x)` is the Jacobian matrix of `f` at `x`

The operation is implemented using `torch::matmul(v, inputs)` which efficiently computes the matrix multiplication without explicitly forming the full Jacobian matrix.

### Computational Complexity
- **Time**: O(n*m) where n is the dimension of v and m is the dimension of inputs
- **Space**: O(m) for the output tensor
- **Memory efficient**: No need to store the full Jacobian matrix

## Relationship to Other AD Operations

VJP is closely related to other automatic differentiation operations:

- **Forward vs Reverse**: VJP is used in reverse-mode AD (backpropagation)
- **JVP**: Jacobian-Vector Product (`torch::jvp`) is the forward-mode counterpart
- **Gradients**: VJP with v=1 gives the gradient of a scalar function
- **Hessian**: Second-order derivatives can be computed using nested VJP calls

## Performance Notes

- **Efficient implementation**: Uses optimized BLAS operations for matrix multiplication
- **Memory efficient**: Avoids materializing the full Jacobian matrix
- **Parallelizable**: Can take advantage of multi-threaded BLAS libraries
- **GPU support**: Works efficiently on CUDA tensors when available

## See Also

- [`torch::jvp`](jvp.md) - Jacobian-Vector Product (forward-mode AD)
- [`torch::grad`](grad.md) - Gradient computation
- [`torch::jacobian`](jacobian.md) - Full Jacobian matrix computation
- [`torch::hessian`](hessian.md) - Hessian matrix computation

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::vjp "my_func" $inputs $vector]
```

**New named parameter syntax:**
```tcl
set result [torch::vjp -func "my_func" -inputs $inputs -v $vector]
```

**CamelCase alias:**
```tcl
set result [torch::vectorJacobianProduct -func "my_func" -inputs $inputs -v $vector]
```

Both syntaxes are fully supported and can be used interchangeably. The named parameter syntax is recommended for new code as it's more readable and self-documenting.

### Alternative Parameter Names

The named parameter syntax supports alternative parameter names for flexibility:

- `-func` or `-function`
- `-inputs` or `-input`  
- `-v` or `-vector`

This allows for more readable code based on context and personal preference. 