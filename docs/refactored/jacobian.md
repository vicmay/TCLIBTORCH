# torch::jacobian

Computes the Jacobian matrix of a function with respect to its inputs.

## Syntax

### Modern Syntax (Recommended)
```tcl
torch::jacobian -func FUNCTION -inputs TENSOR
torch::Jacobian -func FUNCTION -inputs TENSOR  ;# camelCase alias
```

### Legacy Syntax (Backward Compatible)
```tcl
torch::jacobian FUNCTION TENSOR
torch::Jacobian FUNCTION TENSOR  ;# camelCase alias
```

## Parameters

### Named Parameters (Modern Syntax)
- **`-func FUNCTION`** *(required)*: Function handle or name for which to compute the Jacobian
- **`-function FUNCTION`** *(alias)*: Alternative name for `-func` parameter
- **`-inputs TENSOR`** *(required)*: Input tensor containing the variables with respect to which the Jacobian is computed
- **`-input TENSOR`** *(alias)*: Alternative name for `-inputs` parameter

### Positional Parameters (Legacy Syntax)
1. **`FUNCTION`** *(required)*: Function handle or name for which to compute the Jacobian
2. **`TENSOR`** *(required)*: Input tensor containing the variables with respect to which the Jacobian is computed

## Return Value

Returns a new tensor representing the Jacobian matrix:
- **Shape**: `N × N` where `N` is the total number of elements in the input tensor
- **Type**: Same data type as the input tensor
- **Content**: Currently returns an identity matrix as a placeholder implementation

## Description

The `torch::jacobian` function computes the Jacobian matrix of a vector-valued function with respect to its input variables. The Jacobian matrix contains all first-order partial derivatives of the function.

**Mathematical Definition:**
For a function `f: ℝⁿ → ℝᵐ`, the Jacobian matrix `J` is defined as:

```
J[i,j] = ∂f_i/∂x_j
```

Where:
- `f_i` is the i-th component of the output
- `x_j` is the j-th component of the input
- `J[i,j]` is the partial derivative of `f_i` with respect to `x_j`

**Current Implementation:**
The current implementation returns an identity matrix as a placeholder. This is suitable for:
- **Testing and validation** of the dual syntax system
- **API development** and interface design
- **Future extension** with actual automatic differentiation

## Examples

### Basic Usage
```tcl
# Create input tensor
set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]

# Compute Jacobian using modern syntax
set jacobian1 [torch::jacobian -func "my_function" -inputs $inputs]
# Result: 3×3 identity matrix

# Compute Jacobian using legacy syntax
set jacobian2 [torch::jacobian "my_function" $inputs]
# Result: identical to jacobian1

# Compute using camelCase alias
set jacobian3 [torch::Jacobian "my_function" $inputs]
# Result: identical to jacobian1 and jacobian2
```

### Different Input Sizes
```tcl
# Single variable function
set single_var [torch::tensorCreate -data {5.0} -dtype float32]
set jacobian_1x1 [torch::jacobian "func" $single_var]
# Result: 1×1 identity matrix

# Two variable function
set two_vars [torch::tensorCreate -data {1.0 2.0} -dtype float32]
set jacobian_2x2 [torch::jacobian "func" $two_vars]
# Result: 2×2 identity matrix

# Multi-variable function
set multi_vars [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
set jacobian_5x5 [torch::jacobian "func" $multi_vars]
# Result: 5×5 identity matrix
```

### Multi-dimensional Input Tensors
```tcl
# 2D input tensor (flattened for Jacobian computation)
set matrix_input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set jacobian_matrix [torch::jacobian "func" $matrix_input]
# Result: 4×4 identity matrix (flattened from 2×2 input)

# 3D input tensor
set tensor_3d [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3 1} -dtype float32]
set jacobian_3d [torch::jacobian "func" $tensor_3d]
# Result: 6×6 identity matrix (flattened from 2×3×1 input)
```

### Parameter Aliases
```tcl
set inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]

# All equivalent ways to call the function
set j1 [torch::jacobian -func "f" -inputs $inputs]
set j2 [torch::jacobian -function "f" -input $inputs]
set j3 [torch::jacobian -inputs $inputs -func "f"]  # Parameter order doesn't matter
```

### Error Handling Examples
```tcl
# Invalid usage examples (will raise errors)

# Missing arguments
catch {torch::jacobian} error
# Error: Usage information displayed

# Missing second argument
catch {torch::jacobian "func"} error
# Error: Usage information displayed

# Invalid tensor name
catch {torch::jacobian "func" invalid_tensor} error
# Error: Error in jacobian: ...

# Unknown parameter
set tensor [torch::tensorCreate -data {1.0 2.0} -dtype float32]
catch {torch::jacobian -unknown_param "func" -inputs $tensor} error
# Error: Unknown parameter: -unknown_param

# Empty function name
catch {torch::jacobian "" $tensor} error
# Error: Required parameters missing: func and inputs required
```

## Mathematical Properties

1. **Square Matrix**: The Jacobian is always an N×N square matrix where N is the number of input variables
2. **Identity Matrix**: Current implementation returns identity matrix (∂xᵢ/∂xⱼ = δᵢⱼ)
3. **Input Flattening**: Multi-dimensional inputs are flattened to 1D for Jacobian computation
4. **Type Preservation**: Output tensor maintains the same data type as input

## Comparison with Related Functions

- **`torch::jacobian`**: Computes full Jacobian matrix (N×N for N variables)
- **`torch::hessian`**: Computes second-order derivatives (Hessian matrix)
- **`torch::grad`**: Computes gradients (first-order derivatives)
- **`torch::vjp`**: Vector-Jacobian product (more memory efficient for certain computations)
- **`torch::jvp`**: Jacobian-vector product (forward-mode automatic differentiation)

## Performance Notes

- **Memory**: Creates N×N matrix where N is the number of input elements
- **Computation**: Current placeholder implementation is O(N²) space, O(N²) time
- **Future**: Real implementation would use automatic differentiation for efficiency
- **CUDA Support**: Inherits GPU acceleration from underlying tensor operations

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Old style (still supported)
set jacobian [torch::jacobian "my_function" $input_tensor]

# New style (recommended)
set jacobian [torch::jacobian -func "my_function" -inputs $input_tensor]
```

### Benefits of Modern Syntax
- **Explicit parameter names** improve code readability
- **Parameter validation** provides better error messages
- **Flexible parameter order** allows any arrangement of named parameters
- **Extensibility** enables future parameter additions
- **IDE support** enables better autocompletion and documentation

## Use Cases

### 1. Automatic Differentiation
```tcl
# Compute derivatives of neural network layers
set network_inputs [torch::tensorCreate -data {0.5 -0.3 1.2} -dtype float32]
set jacobian [torch::jacobian "forward_pass" $network_inputs]
# Analyze gradient flow and sensitivity
```

### 2. Optimization and Training
```tcl
# Compute parameter sensitivities for optimization
set parameters [torch::tensorCreate -data {0.1 0.2 0.3 0.4} -dtype float32]
set param_jacobian [torch::jacobian "loss_function" $parameters]
# Use for gradient-based optimization algorithms
```

### 3. Sensitivity Analysis
```tcl
# Analyze how outputs change with respect to inputs
set model_inputs [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32]
set sensitivity [torch::jacobian "model_prediction" $model_inputs]
# Understand model behavior and robustness
```

### 4. Mathematical Function Analysis
```tcl
# Study mathematical functions and their derivatives
set variables [torch::tensorCreate -data {-1.0 0.0 1.0} -dtype float32]
set derivatives [torch::jacobian "polynomial_func" $variables]
# Analyze critical points and function behavior
```

## Implementation Notes

### Current Placeholder Implementation
The current implementation returns an identity matrix, which represents:
- **Linear functions**: For f(x) = x, the Jacobian is indeed the identity matrix
- **Testing framework**: Validates the dual syntax parsing and error handling
- **API stability**: Provides consistent interface for future enhancements

### Future Enhancements
- **Real automatic differentiation**: Integration with PyTorch's autograd system
- **Function evaluation**: Support for actual function handles and callable objects
- **Batched computation**: Efficient computation for multiple inputs
- **Sparse Jacobians**: Memory-efficient representation for sparse derivatives

## See Also

- [`torch::hessian`](hessian.md) - Second-order derivatives (Hessian matrix)
- [`torch::grad`](grad.md) - Gradient computation
- [`torch::vjp`](vjp.md) - Vector-Jacobian product
- [`torch::jvp`](jvp.md) - Jacobian-vector product
- [Automatic Differentiation Guide](../automatic_differentiation.md)
- [Mathematical Operations Guide](../mathematical_operations.md) 