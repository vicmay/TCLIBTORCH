# torch::softmax2d

Applies the 2D Softmax activation function to an input tensor.

## Syntax

### Positional Syntax (Original)
```tcl
torch::softmax2d tensor ?dimension?
```

### Named Parameter Syntax (New)
```tcl
torch::softmax2d -input tensor ?-dim dimension?
torch::softmax2d -tensor tensor ?-dimension dimension?
```

### CamelCase Alias
```tcl
torch::softmax2D tensor ?dimension?
torch::softmax2D -input tensor ?-dim dimension?
torch::softmax2D -tensor tensor ?-dimension dimension?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` / `-input` / `-tensor` | tensor | required | Input tensor to apply softmax2d to |
| `dimension` / `-dim` / `-dimension` | integer | 1 | Dimension along which to apply softmax |

## Description

The `torch::softmax2d` function applies the 2D Softmax function to an input tensor. The 2D Softmax is commonly used in computer vision applications and neural networks for producing probability distributions across spatial dimensions.

### Mathematical Formula

For a tensor `x` along dimension `d`:

```
softmax2d(x_i) = exp(x_i) / sum(exp(x_j)) for j in dimension d
```

The function ensures that the output values sum to 1.0 along the specified dimension, creating a probability distribution.

## Examples

### Basic Usage - Positional Syntax
```tcl
# 3D tensor (batch, channel, spatial)
set input [torch::randn {2 3 4}]
set result [torch::softmax2d $input]
puts "Result: $result"

# With custom dimension
set result [torch::softmax2d $input 0]
```

### Basic Usage - Named Parameter Syntax
```tcl
# Using -input parameter
set input [torch::randn {2 3 4}]
set result [torch::softmax2d -input $input]

# With custom dimension
set result [torch::softmax2d -input $input -dim 2]

# Alternative parameter names
set result [torch::softmax2d -tensor $input -dimension 1]
```

### CamelCase Alias
```tcl
# CamelCase version
set input [torch::randn {3 4 5}]
set result [torch::softmax2D $input]
set result [torch::softmax2D -input $input -dim 1]
```

### Computer Vision Example
```tcl
# Typical image classification scenario
set features [torch::randn {1 3 224 224}]  ;# Batch, channels, height, width
set probs [torch::softmax2d $features 1]   ;# Along channel dimension
```

### Multi-dimensional Example
```tcl
# 4D tensor processing
set batch_data [torch::randn {8 3 32 32}]
set result [torch::softmax2d -input $batch_data -dim 1]

# 5D tensor
set volume [torch::randn {2 3 4 5 6}]
set result [torch::softmax2d $volume 2]
```

## Common Use Cases

1. **Image Classification**: Apply softmax across channel dimensions
2. **Attention Mechanisms**: Create probability distributions over spatial dimensions
3. **Neural Network Outputs**: Convert logits to probabilities
4. **Computer Vision**: Normalize feature maps

## Dimension Guidelines

- **Default (dim=1)**: Usually the channel dimension in image data
- **dim=0**: Along batch dimension (less common)
- **dim=-1**: Last dimension
- **dim=-2**: Second to last dimension

## Performance Notes

- The function preserves the input tensor's shape
- Memory efficient implementation
- Supports all PyTorch tensor types
- Works with CUDA tensors when available

## Error Handling

The function validates:
- Input tensor must be valid
- Dimension parameter must be within tensor bounds
- Proper parameter syntax

```tcl
# Error examples
catch {torch::softmax2d} error          ;# Missing tensor
catch {torch::softmax2d invalid} error  ;# Invalid tensor
catch {torch::softmax2d $tensor abc} error  ;# Invalid dimension
```

## Mathematical Properties

1. **Output Range**: All values in [0, 1]
2. **Sum Property**: Values sum to 1.0 along the specified dimension
3. **Monotonic**: Preserves relative ordering
4. **Differentiable**: Smooth gradient for backpropagation

## Migration Guide

### From Positional to Named Syntax

```tcl
# Old positional syntax
set result [torch::softmax2d $tensor]
set result [torch::softmax2d $tensor 2]

# New named parameter syntax
set result [torch::softmax2d -input $tensor]
set result [torch::softmax2d -input $tensor -dim 2]

# CamelCase alias
set result [torch::softmax2D -input $tensor -dim 2]
```

### Parameter Mapping

| Positional | Named Parameter | Alternative |
|------------|-----------------|-------------|
| `tensor` | `-input tensor` | `-tensor tensor` |
| `dimension` | `-dim dimension` | `-dimension dimension` |

## Return Value

Returns a new tensor with the same shape as the input, containing the softmax2d values.

## See Also

- `torch::softmax` - 1D softmax function
- `torch::log_softmax` - Log-softmax function
- `torch::sigmoid` - Sigmoid activation function
- `torch::relu` - ReLU activation function

## Version Information

- **Dual Syntax Support**: ✅ Available
- **CamelCase Alias**: ✅ `torch::softmax2D`
- **Named Parameters**: ✅ `-input`, `-tensor`, `-dim`, `-dimension`
- **Backward Compatibility**: ✅ Full support for positional syntax 