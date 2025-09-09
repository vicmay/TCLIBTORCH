# torch::tensor_sigmoid / torch::tensorSigmoid

Applies the sigmoid function element-wise to the input tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_sigmoid tensor
```

### Named Parameters (New Syntax)
```tcl
torch::tensor_sigmoid -input tensor
torch::tensorSigmoid -input tensor
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |

## Returns

Returns a string handle to the tensor containing the result of applying the sigmoid function: `sigmoid(x) = 1 / (1 + exp(-x))`.

## Mathematical Description

The sigmoid function is defined as:
```
σ(x) = 1 / (1 + exp(-x))
```

Properties:
- Output range: (0, 1)
- Monotonically increasing
- S-shaped curve
- Center point at (0, 0.5)
- Commonly used as activation function in neural networks

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]

# Apply sigmoid (positional syntax)
set result1 [torch::tensor_sigmoid $tensor]

# Apply sigmoid (named syntax)
set result2 [torch::tensor_sigmoid -input $tensor]

# Apply sigmoid (camelCase alias)
set result3 [torch::tensorSigmoid -input $tensor]
```

### Mathematical Examples

```tcl
# Input: [-2, -1, 0, 1, 2]
# Expected output: [0.119, 0.269, 0.5, 0.731, 0.881] (approximately)
set tensor [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]

# Single value at origin
set tensor [torch::tensor_create -data {0.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]  # Should be 0.5

# Large positive value (approaches 1)
set tensor [torch::tensor_create -data {10.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]  # Should be ~0.99995

# Large negative value (approaches 0)
set tensor [torch::tensor_create -data {-10.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]  # Should be ~0.00005
```

### Different Data Types

```tcl
# Float32 tensor
set tensor [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]

# Float64 tensor
set tensor [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float64 -device cpu]
set result [torch::tensor_sigmoid $tensor]

# Note: Sigmoid is typically used with floating-point tensors
# Integer tensors will be promoted to float
set tensor [torch::tensor_create -data {-1 0 1} -dtype int32 -device cpu]
set result [torch::tensor_sigmoid $tensor]
```

### Multi-dimensional Tensors

```tcl
# 2D tensor
set tensor [torch::tensor_create -data {{-1.0 0.0} {1.0 2.0}} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]

# 3D tensor
set tensor [torch::zeros {2 3 4} float32 cpu]
set result [torch::tensor_sigmoid $tensor]  # All values will be 0.5

# Matrix of values
set tensor [torch::tensor_create -data {{-2.0 -1.0 0.0} {1.0 2.0 3.0}} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]
```

### Neural Network Usage

```tcl
# Common use case: sigmoid activation for binary classification
# Linear layer output
set logits [torch::tensor_create -data {-0.5 2.1 -1.8 0.3} -dtype float32 -device cpu]

# Apply sigmoid to get probabilities
set probabilities [torch::tensor_sigmoid $logits]
torch::tensor_print $probabilities  # Values between 0 and 1

# Threshold for binary classification
set predictions [torch::threshold $probabilities 0.5 0.0 1.0]
```

### Edge Cases

```tcl
# Very small values
set tensor [torch::tensor_create -data {-100.0 -50.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]  # Should be very close to 0

# Very large values
set tensor [torch::tensor_create -data {50.0 100.0} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]  # Should be very close to 1

# Zero tensor
set tensor [torch::zeros {3 3} float32 cpu]
set result [torch::tensor_sigmoid $tensor]  # All values will be 0.5

# Boundary cases
set tensor [torch::tensor_create -data {-0.0 0.0 1e-10 -1e-10} -dtype float32 -device cpu]
set result [torch::tensor_sigmoid $tensor]
```

### CUDA Usage

```tcl
# Create tensor on CUDA device (if available)
if {[torch::cuda_is_available]} {
    set tensor [torch::tensor_create -data {-1.0 0.0 1.0 2.0} -dtype float32 -device cuda]
    set result [torch::tensor_sigmoid $tensor]
    torch::tensor_print $result
}
```

## Notes

- The sigmoid function is numerically stable for most practical inputs
- For very large positive inputs, the result approaches 1.0
- For very large negative inputs, the result approaches 0.0
- The function is differentiable everywhere
- Commonly used in neural networks, especially for binary classification
- Both `torch::tensor_sigmoid` and `torch::tensorSigmoid` are equivalent
- The output tensor has the same shape and device as the input tensor

## Error Handling

The command will return an error if:
- The input tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old code (still works)
set result [torch::tensor_sigmoid $tensor]

# New code (recommended)
set result [torch::tensor_sigmoid -input $tensor]
```

## See Also

- `torch::tensor_tanh` - Hyperbolic tangent activation function
- `torch::tensor_relu` - ReLU activation function
- `torch::softmax` - Softmax activation function for multi-class classification
- `torch::tensor_exp` - Exponential function
- `torch::tensor_log` - Natural logarithm function 