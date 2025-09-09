# RMS Normalization (rms_norm)

## Description

The RMS (Root Mean Square) normalization layer normalizes the input tensor by dividing it by the RMS value computed over the specified dimensions. This is a simpler alternative to layer normalization that doesn't require learnable parameters.

The RMS value is calculated as: `RMS(x) = sqrt(mean(x^2))`, and the output is computed as: `y = x / (RMS(x) + eps)`.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::rms_norm tensor normalized_shape ?eps?
```

### Named Parameter Syntax (New)
```tcl
torch::rmsNorm -input tensor -normalizedShape shape_list ?-eps value?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor/input | tensor | required | The input tensor to normalize |
| normalized_shape/normalizedShape | list | required | The shape of the normalization window. Must match the last dimensions of the input tensor |
| eps | float | 1e-5 | A small value added to the denominator for numerical stability |

## Return Value

Returns a tensor of the same shape as the input tensor, normalized using RMS normalization.

## Examples

### Basic Usage - 1D Tensor
```tcl
# Create a 1D tensor
set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]

# Using positional syntax
set result1 [torch::rms_norm $x {4}]

# Using named parameter syntax
set result2 [torch::rmsNorm -input $x -normalizedShape {4}]
```

### 2D Tensor with Custom Epsilon
```tcl
# Create a 2D tensor
set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
set x [torch::tensor_reshape $x {2 3}]

# Using positional syntax
set result1 [torch::rms_norm $x {3} 1e-3]

# Using named parameter syntax
set result2 [torch::rmsNorm -input $x -normalizedShape {3} -eps 1e-3]
```

### Batch Normalization
```tcl
# Create a batch of 2D data (batch_size x channels x height x width)
set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} float32]
set x [torch::tensor_reshape $x {2 2 2}]

# Normalize each sample independently
set result [torch::rmsNorm -input $x -normalizedShape {2 2}]
```

## Error Handling

The command will throw an error in the following cases:
- Input tensor is empty
- Invalid normalized_shape (dimensions don't match input tensor)
- Negative eps value
- Missing required parameters

## Migration Guide

### From Positional to Named Parameter Syntax

Old code:
```tcl
torch::rms_norm $tensor {4} 1e-5
```

New code:
```tcl
torch::rmsNorm -input $tensor -normalizedShape {4} -eps 1e-5
```

### Key Differences
1. Command name changes from `rms_norm` to `rmsNorm`
2. Parameters are explicitly named with `-input`, `-normalizedShape`, and `-eps`
3. The order of parameters doesn't matter in the new syntax
4. Both syntaxes remain supported for backward compatibility

## See Also

- `torch::layer_norm` - Layer normalization with learnable parameters
- `torch::batch_norm` - Batch normalization
- `torch::instance_norm` - Instance normalization 