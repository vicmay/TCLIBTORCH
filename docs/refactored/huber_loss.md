# torch::huber_loss

Creates a criterion that uses a squared term if the absolute element-wise error falls below delta and a linear term otherwise. This loss combines the best properties of L1 and L2 losses.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::huber_loss -input tensor -target tensor ?-reduction string? ?-delta double?
torch::huberLoss -input tensor -target tensor ?-reduction string? ?-delta double?
```

### Positional Syntax (Legacy)
```tcl
torch::huber_loss input target ?reduction? ?delta?
torch::huberLoss input target ?reduction? ?delta?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-input` | tensor | - | Input tensor (required) |
| `-target` | tensor | - | Target tensor (required) |
| `-reduction` | string | "mean" | Reduction type: "none", "mean", or "sum" |
| `-delta` | double | 1.0 | Threshold for switching between squared and linear loss |

## Mathematical Definition

The Huber loss is defined as:

```
loss(x, y) = 1/n * Σ l(x_i, y_i)
```

where:

```
         ⎧ 0.5 * (x_i - y_i)²                    if |x_i - y_i| ≤ δ
l(x_i, y_i) = ⎨
         ⎩ δ * (|x_i - y_i| - 0.5 * δ)           otherwise
```

- When the error is small (≤ δ), it behaves like L2 loss (squared error)
- When the error is large (> δ), it behaves like L1 loss (absolute error)
- δ (delta) controls the threshold between the two behaviors

## Return Value

Returns a tensor containing the Huber loss. The shape depends on the reduction parameter:
- `"none"`: Returns a tensor of the same shape as input, containing element-wise losses
- `"mean"`: Returns a scalar tensor with the mean loss
- `"sum"`: Returns a scalar tensor with the sum of all losses

## Examples

### Basic Usage with Named Parameters (Recommended)
```tcl
# Create input and target tensors
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target [torch::tensor_create {1.5 1.5 3.5 3.5} float32]

# Compute Huber loss with default parameters
set loss [torch::huber_loss -input $input -target $target]
set loss_value [torch::tensor_item $loss]
puts "Huber loss: $loss_value"
```

### Custom Delta and Reduction
```tcl
# Use custom delta value
set loss_custom [torch::huber_loss -input $input -target $target -delta 0.5 -reduction "sum"]
set loss_sum [torch::tensor_item $loss_custom]
puts "Huber loss (delta=0.5, sum): $loss_sum"

# No reduction - get per-element losses
set loss_none [torch::huber_loss -input $input -target $target -reduction "none"]
set shape [torch::tensor_shape $loss_none]
puts "Per-element losses shape: $shape"
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::huberLoss -input $input -target $target -delta 2.0]
set loss_value [torch::tensor_item $loss]
puts "Huber loss (camelCase): $loss_value"
```

### Positional Syntax (Legacy)
```tcl
# Basic positional usage
set loss1 [torch::huber_loss $input $target]

# With custom parameters
set loss2 [torch::huber_loss $input $target "mean" 1.5]
```

## Mathematical Properties

### Small Error Behavior (|error| ≤ δ)
```tcl
# When error is small, behaves like L2 loss
set input [torch::tensor_create {2.0} float32]
set target [torch::tensor_create {1.5} float32]  ; # error = 0.5 < delta = 1.0

set loss [torch::huber_loss -input $input -target $target -delta 1.0 -reduction "none"]
set loss_value [torch::tensor_item $loss]
# loss_value = 0.5 * (2.0 - 1.5)² = 0.5 * 0.25 = 0.125
```

### Large Error Behavior (|error| > δ)
```tcl
# When error is large, behaves like L1 loss
set input [torch::tensor_create {3.0} float32]
set target [torch::tensor_create {1.0} float32]  ; # error = 2.0 > delta = 1.0

set loss [torch::huber_loss -input $input -target $target -delta 1.0 -reduction "none"]
set loss_value [torch::tensor_item $loss]
# loss_value = 1.0 * (2.0 - 0.5 * 1.0) = 1.0 * 1.5 = 1.5
```

## Use Cases

### Robust Regression
```tcl
# Huber loss is less sensitive to outliers than L2 loss
set predictions [torch::tensor_create {1.0 2.0 10.0 4.0} float32]  ; # 10.0 is outlier
set targets [torch::tensor_create {1.1 2.1 3.0 4.1} float32]

# Small delta makes it more like L1 (robust to outliers)
set robust_loss [torch::huber_loss -input $predictions -target $targets -delta 0.5]

# Large delta makes it more like L2 (smooth gradients)
set smooth_loss [torch::huber_loss -input $predictions -target $targets -delta 2.0]
```

### Multi-dimensional Data
```tcl
# Works with multi-dimensional tensors
set input_2d [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target_2d [torch::tensor_create {1.5 1.5 3.5 3.5} float32]

set input_reshaped [torch::tensor_reshape $input_2d {2 2}]
set target_reshaped [torch::tensor_reshape $target_2d {2 2}]

set loss_2d [torch::huber_loss -input $input_reshaped -target $target_reshaped]
```

## Delta Parameter Effects

The delta parameter controls the transition point between quadratic and linear behavior:

- **Small delta (< 1.0)**: More like L1 loss, robust to outliers, less smooth gradients
- **Large delta (> 1.0)**: More like L2 loss, smooth gradients, less robust to outliers
- **Default delta (1.0)**: Balanced between robustness and smoothness

## Reduction Options

| Reduction | Description | Output Shape |
|-----------|-------------|--------------|
| `"none"` | No reduction, return per-element losses | Same as input |
| `"mean"` | Average of all losses | Scalar |
| `"sum"` | Sum of all losses | Scalar |

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::huber_loss -target $target} error
# Error: Required parameters -input and -target must be provided

# Invalid tensor names
catch {torch::huber_loss "invalid" $target} error
# Error: Invalid input tensor name

# Unknown parameters
catch {torch::huber_loss -input $input -target $target -unknown "value"} error
# Error: Unknown parameter: -unknown
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set loss [torch::huber_loss $input $target "mean" 1.5]

# New named parameter syntax
set loss [torch::huber_loss -input $input -target $target -reduction "mean" -delta 1.5]
```

### Advantages of Named Parameters

1. **Clarity**: Parameter names make code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Defaults**: Optional parameters can be omitted
4. **Validation**: Better error messages for incorrect usage

## Comparison with Other Losses

| Loss Function | Behavior | Sensitivity to Outliers | Gradient Smoothness |
|---------------|----------|------------------------|-------------------|
| L1 (MAE) | Linear everywhere | Low | Poor (non-differentiable at 0) |
| L2 (MSE) | Quadratic everywhere | High | Excellent |
| Huber | Quadratic + Linear | Medium | Good |
| Smooth L1 | Quadratic + Linear (different formulation) | Medium | Good |

## Performance Notes

- Huber loss computation is efficient and scales well with tensor size
- Both positional and named parameter syntaxes have identical performance
- The `"none"` reduction is fastest for element-wise analysis
- Memory usage scales linearly with input tensor size

## See Also

- `torch::l1_loss` - L1 loss function
- `torch::mse_loss` - Mean squared error loss
- `torch::smooth_l1_loss` - Smooth L1 loss (related but different formulation)
- `torch::tensor_create` - Creating tensors
- `torch::tensor_item` - Extracting scalar values 