# torch::gradient

## Overview
Computes the gradient of a tensor using finite differences. This function estimates the gradient by computing differences between adjacent elements along the specified dimension.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::gradient -input TENSOR ?-dim DIMENSION? ?-spacing SPACING?
torch::gradient -tensor TENSOR ?-dimension DIMENSION? ?-spacing SPACING?
```

### Positional Syntax (Legacy)
```tcl
torch::gradient TENSOR ?SPACING? ?DIMENSION?
```

### camelCase Alias
```tcl
torch::gradientCmd -input TENSOR ?-dim DIMENSION? ?-spacing SPACING?
```

## Parameters

### Required Parameters
- **`-input`** or **`-tensor`**: Input tensor for gradient computation
  - Type: Tensor handle
  - The input tensor for which to compute the gradient

### Optional Parameters
- **`-dim`** or **`-dimension`**: Dimension along which to compute gradient
  - Type: Integer
  - Default: -1 (automatically inferred dimension)
  - Specifies the dimension along which to compute the gradient

- **`-spacing`**: Spacing between points (currently accepted but not used)
  - Type: Float or list of floats
  - Default: 1.0
  - Note: This parameter is accepted for compatibility but not currently implemented

## Return Value
Returns a tensor handle containing the computed gradient. The output tensor has one less element than the input tensor along the specified dimension.

## Examples

### Basic Usage
```tcl
# Create a simple 1D tensor
set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]

# Compute gradient using named syntax
set result [torch::gradient -input $input]
puts "Gradient shape: [torch::tensor_shape $result]"  ;# Output: 3

# Same computation using positional syntax
set result2 [torch::gradient $input]
puts "Gradient shape: [torch::tensor_shape $result2]"  ;# Output: 3
```

### 2D Tensor Gradient
```tcl
# Create a 2D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]

# Compute gradient along dimension 0
set grad_dim0 [torch::gradient -input $input -dim 0]
puts "Gradient shape (dim 0): [torch::tensor_shape $grad_dim0]"  ;# Output: 1 3

# Compute gradient along dimension 1
set grad_dim1 [torch::gradient -input $input -dim 1]
puts "Gradient shape (dim 1): [torch::tensor_shape $grad_dim1]"  ;# Output: 2 2
```

### Using camelCase Alias
```tcl
set input [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float32]

# Using camelCase alias
set result [torch::gradientCmd -input $input]
puts "Gradient computed using camelCase alias: [torch::tensor_shape $result]"
```

### Parameter Variations
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]

# Using alternative parameter names
set result1 [torch::gradient -tensor $input -dimension 0]
set result2 [torch::gradient -input $input -dim 0]

# Both produce the same result
puts "Shape 1: [torch::tensor_shape $result1]"  ;# Output: 1 3
puts "Shape 2: [torch::tensor_shape $result2]"  ;# Output: 1 3
```

## Mathematical Notes

The gradient is computed using finite differences:
- For 1D tensors: `gradient[i] = input[i+1] - input[i]`
- For multi-dimensional tensors: differences are computed along the specified dimension

The output tensor has one less element than the input tensor along the gradient dimension.

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameter
catch {torch::gradient} msg
puts $msg  ;# "Required parameter missing: input tensor"

# Invalid tensor handle
catch {torch::gradient -input "invalid_tensor"} msg
puts $msg  ;# "Invalid tensor name"

# Invalid dimension parameter
catch {torch::gradient -input $tensor -dim "invalid"} msg
puts $msg  ;# "Invalid dim parameter: must be integer"

# Unknown parameter
catch {torch::gradient -input $tensor -unknown_param value} msg
puts $msg  ;# "Unknown parameter: -unknown_param. Valid parameters are: -input/-tensor, -dim/-dimension, -spacing"
```

## Data Type Support

The gradient function supports all numeric tensor types:
- `float32`, `float64`
- `int32`, `int64`
- Other numeric types

```tcl
# Float64 tensor
set input_f64 [torch::tensor_create -data {1.0 2.0 4.0 7.0} -shape {4} -dtype float64]
set result_f64 [torch::gradient -input $input_f64]

# Integer tensor
set input_int [torch::tensor_create -data {1 2 4 7} -shape {4} -dtype int32]
set result_int [torch::gradient -input $input_int]
```

## Performance Notes

- The gradient computation is implemented using `torch::diff` internally
- Performance scales with tensor size and dimensionality
- For large tensors, consider computing gradients along specific dimensions rather than the full tensor

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set result [torch::gradient $input]
set result [torch::gradient $input {} 0]
set result [torch::gradient $input 1.0 1]

# New named syntax
set result [torch::gradient -input $input]
set result [torch::gradient -input $input -dim 0]
set result [torch::gradient -input $input -spacing 1.0 -dim 1]
```

### Parameter Mapping

| Positional | Named | Alternative |
|------------|-------|-------------|
| `tensor` | `-input` | `-tensor` |
| `spacing` | `-spacing` | N/A |
| `dim` | `-dim` | `-dimension` |

## See Also

- [torch::diff](diff.md) - Computes differences between adjacent elements
- [torch::tensor_create](tensor_create.md) - Create tensors
- [torch::tensor_shape](tensor_shape.md) - Get tensor dimensions

## Implementation Notes

- The current implementation uses `torch::diff` for gradient approximation
- Future versions may include more sophisticated gradient computation methods
- The spacing parameter is accepted but not currently used in the computation
- For exact gradients in automatic differentiation, use the autograd functionality 