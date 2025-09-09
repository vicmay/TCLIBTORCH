# torch::weight_norm

Applies weight normalization to a tensor. Weight normalization is a technique that reparameterizes the weight vectors in a neural network by decoupling the length of those weight vectors from their direction.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::weight_norm tensor ?dim?
```

### Named Parameter Syntax
```tcl
torch::weight_norm -input tensor ?-dim dim?
torch::weight_norm -tensor tensor ?-dim dim?
```

### CamelCase Alias
```tcl
torch::weightNorm tensor ?dim?
torch::weightNorm -input tensor ?-dim dim?
torch::weightNorm -tensor tensor ?-dim dim?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` / `-input` / `-tensor` | tensor handle | required | Input tensor to normalize |
| `dim` / `-dim` | integer | 0 | Dimension along which to compute the norm |

## Description

Weight normalization reparameterizes the weight vectors by decoupling their magnitude from their direction. The formula used is:

```
w = g * v / ||v||
```

Where:
- `w` is the normalized weight
- `g` is a learnable scalar (magnitude)
- `v` is the weight vector (direction)
- `||v||` is the L2 norm of the weight vector

This technique helps with training stability and can improve convergence in deep neural networks.

## Examples

### Basic Usage

```tcl
;# Create a weight tensor
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

;# Apply weight normalization with default dimension (0)
set normalized [torch::weight_norm $weights]

;# Check the shape is preserved
puts [torch::tensor_shape $normalized]
;# Output: 2 2
```

### Specifying Dimension

```tcl
;# Create a 3D weight tensor
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -shape {2 2 2} -dtype float32]

;# Apply weight normalization along dimension 1
set normalized [torch::weight_norm $weights 1]

;# Check the shape is preserved
puts [torch::tensor_shape $normalized]
;# Output: 2 2 2
```

### Using Named Parameters

```tcl
;# Create a weight tensor
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

;# Apply weight normalization using named parameters
set normalized [torch::weight_norm -input $weights -dim 1]

;# Alternative syntax
set normalized2 [torch::weight_norm -tensor $weights -dim 1]
```

### Using CamelCase Alias

```tcl
;# Create a weight tensor
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

;# Apply weight normalization using camelCase alias
set normalized [torch::weightNorm $weights]

;# With named parameters
set normalized2 [torch::weightNorm -input $weights -dim 0]
```

### Neural Network Layer Example

```tcl
;# Simulate a linear layer weight matrix
set weight_matrix [torch::tensor_create -data {0.5 1.0 1.5 2.0 2.5 3.0} -shape {2 3} -dtype float32]

;# Apply weight normalization to the weight matrix
set normalized_weights [torch::weight_norm $weight_matrix 1]

;# The normalized weights can now be used in forward pass
;# Each column (dimension 1) will have unit norm
```

## Mathematical Properties

### Norm Preservation
After weight normalization, the L2 norm along the specified dimension will be 1:

```tcl
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set normalized [torch::weight_norm $weights 0]

;# Check that the norm along dimension 0 is 1
set norms [torch::tensor_norm $normalized 2 0]
puts [torch::tensor_data $norms]
;# Output: approximately [1.0 1.0]
```

### Shape Preservation
Weight normalization preserves the shape of the input tensor:

```tcl
set original_shape [torch::tensor_shape $weights]
set result_shape [torch::tensor_shape $normalized]
puts [expr {$original_shape == $result_shape}]
;# Output: 1 (true)
```

## Error Handling

### Missing Tensor
```tcl
catch {torch::weight_norm} result
puts $result
;# Output: Wrong number of arguments
```

### Invalid Dimension
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
catch {torch::weight_norm $tensor 10} result
puts $result
;# Output: Dimension out of range (expected to be in range of [-2, 1], but got 10)
```

### Unknown Parameter
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
catch {torch::weight_norm -invalid $tensor} result
puts $result
;# Output: Unknown parameter: -invalid. Valid parameters are: -input/-tensor, -dim
```

### Missing Parameter Value
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
catch {torch::weight_norm -input} result
puts $result
;# Output: Missing value for parameter
```

## Edge Cases

### Zero Tensor
```tcl
set zero_tensor [torch::tensor_create -data {0.0 0.0 0.0 0.0} -shape {2 2} -dtype float32]
set result [torch::weight_norm $zero_tensor]
;# This will return a tensor handle, but the normalization may not be meaningful
```

### Single Element Tensor
```tcl
set single [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
set result [torch::weight_norm $single]
puts [torch::tensor_shape $result]
;# Output: 1
```

### Large Values
```tcl
set large_tensor [torch::tensor_create -data {1000.0 2000.0 3000.0 4000.0} -shape {2 2} -dtype float32]
set result [torch::weight_norm $large_tensor]
;# Weight normalization will scale the values appropriately
```

## Data Type Support

The command supports various floating-point data types:

```tcl
;# Float32
set tensor_f32 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set result_f32 [torch::weight_norm $tensor_f32]
puts [torch::tensor_dtype $result_f32]
;# Output: Float32

;# Float64
set tensor_f64 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float64]
set result_f64 [torch::weight_norm $tensor_f64]
puts [torch::tensor_dtype $result_f64]
;# Output: Float64
```

## Consistency Between Syntaxes

Both positional and named parameter syntaxes produce identical results:

```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]

;# Positional syntax
set result1 [torch::weight_norm $tensor 1]

;# Named parameter syntax
set result2 [torch::weight_norm -input $tensor -dim 1]

;# Both produce the same shape
puts [expr {[torch::tensor_shape $result1] == [torch::tensor_shape $result2]}]
;# Output: 1 (true)
```

## Migration from Positional to Named Syntax

### Before (Positional Syntax)
```tcl
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set normalized [torch::weight_norm $weights 0]
```

### After (Named Parameter Syntax)
```tcl
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set normalized [torch::weight_norm -input $weights -dim 0]
```

### Using CamelCase Alias
```tcl
set weights [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2} -dtype float32]
set normalized [torch::weightNorm -input $weights -dim 0]
```

## Return Value

Returns a tensor handle to the normalized weight tensor. The shape and data type are preserved from the input tensor.

## Notes

- Weight normalization is commonly used in deep learning to improve training stability
- The default dimension (0) is typically used for weight matrices in linear layers
- The operation preserves the shape and data type of the input tensor
- Both snake_case (`torch::weight_norm`) and camelCase (`torch::weightNorm`) aliases are available
- The command maintains 100% backward compatibility with the original positional syntax 