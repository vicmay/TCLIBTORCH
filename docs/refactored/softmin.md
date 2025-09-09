# torch::softmin

Applies the Softmin activation function to the input tensor. Softmin is the inverse of Softmax - it produces a probability distribution where smaller input values get higher probabilities.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::softmin -input tensor ?-dim dimension?
torch::softmin -tensor tensor ?-dimension dimension?
torch::softMin -input tensor ?-dim dimension?
```

### Positional Parameters (Legacy)
```tcl
torch::softmin tensor ?dim?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input`/`tensor` | tensor | required | Input tensor |
| `dim`/`dimension` | int | -1 | Dimension along which to apply Softmin |

## Description

The Softmin function is computed as:
```
softmin(x_i) = exp(-x_i) / sum(exp(-x_j) for all j)
```

This is equivalent to applying Softmax to the negated input. It produces a probability distribution where smaller values get higher probabilities.

## Examples

### Basic Usage
```tcl
# Create input tensor
set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -shape {4} -dtype float32]

# Apply softmin - smaller values get higher probabilities
set result [torch::softmin -input $input]
puts "Softmin result: [torch::tensorToList $result]"
# Output: Softmin result: {0.576 0.212 0.078 0.029} (approximately)
```

### Multi-dimensional Tensor
```tcl
# Create 2D tensor
set input [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]

# Apply softmin along dimension 1 (columns)
set result [torch::softmin -input $input -dim 1]
puts "Shape: [torch::tensorShape $result]"
# Output: Shape: 2 3
```

### Mathematical Properties
```tcl
# Demonstrate that softmin sums to 1
set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
set result [torch::softmin -input $input]
set sum [torch::tensorSum $result]
puts "Sum: [torch::tensorItem $sum]"
# Output: Sum: 1.0 (approximately)

# Show that smaller values get higher probabilities
set values [torch::tensorToList $result]
puts "Values: $values"
# Output: Values: {0.665 0.245 0.090} (approximately)
# Note: first value is highest because 1.0 is the smallest input
```

## Legacy Syntax

```tcl
# Positional syntax (backward compatible)
set input [torch::tensorCreate -data {1.0 2.0 3.0} -shape {3} -dtype float32]
set result [torch::softmin $input]

# With explicit dimension
set result [torch::softmin $input 0]
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::softmin $tensor]
set result [torch::softmin $tensor $dim]
```

**New (Named):**
```tcl
set result [torch::softmin -input $tensor]
set result [torch::softmin -input $tensor -dim $dim]
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is self-documenting
- **Consistency**: Follows modern API patterns

## Mathematical Background

Softmin is the inverse of Softmax:
- **Softmax**: Larger values → higher probabilities
- **Softmin**: Smaller values → higher probabilities

This makes Softmin useful when you want to emphasize smaller values in a probability distribution.

## Common Use Cases

1. **Attention Mechanisms**: When smaller distances/costs should get higher attention
2. **Nearest Neighbor**: When closer points should have higher weights
3. **Cost-based Selection**: When lower costs should have higher probabilities
4. **Inverse Temperature**: When cooler (smaller) values should dominate

## Dimensions

The `dim` parameter specifies which dimension to reduce:
- `-1` (default): Last dimension
- `0`: First dimension (rows in 2D)
- `1`: Second dimension (columns in 2D)
- etc.

## Return Value

Returns a new tensor with the same shape as the input, containing the Softmin probabilities. Each slice along the specified dimension will sum to 1.0.

## See Also

- `torch::softmax` - Standard softmax (emphasizes larger values)
- `torch::log_softmax` - Log of softmax for numerical stability
- `torch::gumbel_softmax` - Differentiable sampling from categorical distribution 