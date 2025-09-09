# torch::smooth_l1_loss

**Smooth L1 Loss (Huber Loss with β=1)**

Computes the smooth L1 loss between input and target tensors. This loss function combines the advantages of L1 and L2 losses by being less sensitive to outliers than L2 loss while providing better gradients near zero than L1 loss.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::smooth_l1_loss -input tensor -target tensor ?-reduction string? ?-beta double?
torch::smoothL1Loss -input tensor -target tensor ?-reduction string? ?-beta double?
```

### Positional Syntax (Legacy)
```tcl
torch::smooth_l1_loss input target ?reduction? ?beta?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **input** | tensor | - | Input tensor containing predictions |
| **target** | tensor | - | Target tensor containing ground truth values |
| **reduction** | string | "mean" | Reduction mode: "none", "mean", or "sum" |
| **beta** | double | 1.0 | Threshold value that determines the transition point between L1 and L2 loss |

## Returns

Returns a tensor handle containing the computed smooth L1 loss.

- If `reduction="none"`: Returns tensor with same shape as input
- If `reduction="mean"`: Returns scalar tensor with mean loss
- If `reduction="sum"`: Returns scalar tensor with sum of losses

## Mathematical Details

The smooth L1 loss is computed as:

```
smooth_l1_loss(x, y) = {
    0.5 * (x - y)² / beta,  if |x - y| < beta
    |x - y| - 0.5 * beta,   otherwise
}
```

Where:
- `x` is the input (prediction)
- `y` is the target (ground truth)
- `beta` is the transition threshold

### Key Properties:
- **Smooth transition**: Unlike L1 loss, it's differentiable everywhere
- **Robust to outliers**: Less sensitive to large errors than L2 loss
- **Good gradients**: Provides stable gradients near zero
- **Huber loss variant**: When β=1, it's equivalent to Huber loss

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create input tensor (predictions)
set input [torch::tensorCreate -data {1.0 2.0 -0.5 -1.5} -shape {2 2}]

# Create target tensor (ground truth)
set target [torch::tensorCreate -data {1.2 1.8 -0.3 -1.2} -shape {2 2}]

# Compute smooth L1 loss
set loss [torch::smooth_l1_loss -input $input -target $target]
```

#### With Custom Parameters
```tcl
# Using different beta threshold
set loss [torch::smooth_l1_loss -input $input -target $target -beta 0.5 -reduction "sum"]

# No reduction - preserve input shape
set loss [torch::smooth_l1_loss -input $input -target $target -reduction "none"]
```

#### Using camelCase Alias
```tcl
# Equivalent using camelCase alias
set loss [torch::smoothL1Loss -input $input -target $target -beta 2.0 -reduction "mean"]
```

### Positional Syntax (Legacy)

#### Basic Usage
```tcl
# Basic usage with defaults
set loss [torch::smooth_l1_loss $input $target]

# With reduction parameter (0=none, 1=mean, 2=sum)
set loss [torch::smooth_l1_loss $input $target 0]

# With all parameters
set loss [torch::smooth_l1_loss $input $target 1 0.5]
```

### Different Reduction Modes

```tcl
# No reduction - preserve input shape
set loss_none [torch::smooth_l1_loss -input $input -target $target -reduction "none"]

# Mean reduction (default)
set loss_mean [torch::smooth_l1_loss -input $input -target $target -reduction "mean"]

# Sum reduction
set loss_sum [torch::smooth_l1_loss -input $input -target $target -reduction "sum"]
```

### Beta Parameter Effects

```tcl
# Small beta (more like L1 loss)
set loss_small_beta [torch::smooth_l1_loss -input $input -target $target -beta 0.1]

# Standard beta (balanced)
set loss_standard [torch::smooth_l1_loss -input $input -target $target -beta 1.0]

# Large beta (more like L2 loss)
set loss_large_beta [torch::smooth_l1_loss -input $input -target $target -beta 5.0]
```

## Use Cases

### Object Detection
```tcl
# Bounding box regression in object detection models
set predicted_boxes [torch::tensorCreate -data {10.2 15.8 50.3 40.1}]
set ground_truth_boxes [torch::tensorCreate -data {10.0 16.0 50.0 40.0}]
set loss [torch::smooth_l1_loss -input $predicted_boxes -target $ground_truth_boxes -beta 1.0]
```

### Robust Regression
```tcl
# Regression with outliers - smooth L1 is more robust than L2
set predictions [torch::linear $features $weights]
set true_values [torch::tensorCreate -data {2.1 3.8 1.2 4.5 2.9}]
set loss [torch::smooth_l1_loss -input $predictions -target $true_values -reduction "mean"]
```

### Image Super-Resolution
```tcl
# Pixel-wise loss for image reconstruction
set reconstructed_image [torch::conv2d $low_res_input $upscale_kernel]
set high_res_target [torch::tensorCreate -data $hr_image_data]
set loss [torch::smooth_l1_loss -input $reconstructed_image -target $high_res_target -beta 0.5]
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::smooth_l1_loss -input $input} error
# Error: Required parameters -input and -target must be provided

# Invalid tensor names
catch {torch::smooth_l1_loss -input "invalid" -target $target} error
# Error: Invalid input tensor name

# Invalid beta value
catch {torch::smooth_l1_loss -input $input -target $target -beta "invalid"} error
# Error: Invalid beta parameter value

# Unknown parameters
catch {torch::smooth_l1_loss -input $input -target $target -unknown value} error
# Error: Unknown parameter: -unknown
```

## Performance Notes

- **Computational Efficiency**: Similar computational cost to L1 or L2 loss
- **Memory Usage**: Scales linearly with input tensor size
- **GPU Acceleration**: Fully supports CUDA tensors and GPU computation
- **Gradient Stability**: Provides stable gradients for optimization

## Comparison with Other Loss Functions

| Loss Function | Sensitivity to Outliers | Differentiability | Gradient Behavior |
|---------------|------------------------|-------------------|-------------------|
| **L1 Loss** | Low | Not at zero | Constant gradients |
| **L2 Loss** | High | Everywhere | Linear gradients |
| **Smooth L1** | Medium | Everywhere | Smooth transition |
| **Huber Loss** | Medium | Everywhere | Configurable threshold |

## Compatibility

- **Backward Compatible**: Original positional syntax remains fully supported
- **Thread Safe**: Can be used safely in multi-threaded environments
- **Device Agnostic**: Works with CPU and CUDA tensors
- **Data Types**: Supports float32, float64, and other floating-point types

## Advanced Examples

### Custom Beta for Different Problem Types

```tcl
# For fine-grained predictions (small errors matter)
set loss_fine [torch::smooth_l1_loss -input $predictions -target $targets -beta 0.1]

# For coarse predictions (larger tolerance)
set loss_coarse [torch::smooth_l1_loss -input $predictions -target $targets -beta 2.0]
```

### Batch Processing

```tcl
# Process multiple samples with batch dimension
set batch_predictions [torch::tensorCreate -data $batch_data -shape {32 10}]  # 32 samples, 10 features
set batch_targets [torch::tensorCreate -data $batch_labels -shape {32 10}]
set batch_loss [torch::smooth_l1_loss -input $batch_predictions -target $batch_targets -reduction "mean"]
```

### Multi-dimensional Data

```tcl
# For image data or multi-dimensional outputs
set image_pred [torch::tensorCreate -shape {1 3 224 224}]  # Batch, Channels, Height, Width
set image_true [torch::tensorCreate -shape {1 3 224 224}]
set pixel_loss [torch::smooth_l1_loss -input $image_pred -target $image_true -reduction "mean" -beta 1.0]
```

## See Also

- [`torch::l1_loss`](l1_loss.md) - L1/Mean Absolute Error loss
- [`torch::mse_loss`](mse_loss.md) - Mean Squared Error loss
- [`torch::huber_loss`](huber_loss.md) - Huber loss (configurable delta)
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Cross entropy loss

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set loss [torch::smooth_l1_loss $input $target 1 0.5]

# NEW (Named Parameters)
set loss [torch::smooth_l1_loss -input $input -target $target -reduction "mean" -beta 0.5]

# NEW (camelCase)
set loss [torch::smoothL1Loss -input $input -target $target -reduction "mean" -beta 0.5]
```

### Benefits of Named Parameters

1. **Self-Documenting**: Parameter names make code more readable
2. **Flexible Order**: Parameters can be specified in any order
3. **Optional Parameters**: Only specify the parameters you need
4. **Less Error-Prone**: No need to remember parameter positions
5. **Better Maintainability**: Code is easier to understand and modify

### Beta Parameter Guidelines

- **β = 0.1**: Very sensitive to small errors, almost like L1 loss
- **β = 1.0**: Standard setting, good balance (equivalent to Huber loss)
- **β = 2.0-5.0**: More tolerant of large errors, closer to L2 loss behavior

Choose β based on your problem characteristics:
- **Small β**: When small prediction errors are critical
- **Large β**: When you want to penalize large errors more heavily 