# torch::tversky_loss

## Overview

The `torch::tversky_loss` command implements the Tversky Loss function, which is a generalization of the Dice Loss commonly used in image segmentation tasks. It's particularly useful for handling class imbalanced datasets in medical imaging and computer vision applications.

## Mathematical Definition

The Tversky Loss is defined as:

```
TverskyLoss = 1 - TverskyIndex
```

Where the Tversky Index is:

```
TverskyIndex = (TP + smooth) / (TP + α×FP + β×FN + smooth)
```

Where:
- **TP** = True Positives (sum of predicted probabilities × target values)
- **FP** = False Positives (sum of predicted probabilities × (1 - target values))
- **FN** = False Negatives (sum of (1 - predicted probabilities) × target values)
- **α** = Weight for False Positives (controls precision)
- **β** = Weight for False Negatives (controls recall/sensitivity)
- **smooth** = Smoothing factor to avoid division by zero

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tversky_loss input target ?alpha? ?beta? ?smooth? ?reduction?
```

### Named Parameter Syntax  
```tcl
torch::tversky_loss -input tensor -target tensor ?-alpha double? ?-beta double? ?-smooth double? ?-reduction string?
```

### CamelCase Alias
```tcl
torch::tverskyLoss -input tensor -target tensor ?-alpha double? ?-beta double? ?-smooth double? ?-reduction string?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | tensor | *required* | Input predictions (logits, will be passed through sigmoid) |
| `target` | tensor | *required* | Ground truth binary targets (0 or 1) |
| `alpha` | double | 0.7 | Weight for false positives (precision control) |
| `beta` | double | 0.3 | Weight for false negatives (recall/sensitivity control) |
| `smooth` | double | 1.0 | Smoothing factor to prevent division by zero |
| `reduction` | string | "mean" | Reduction method: "mean", "sum", or "none" |

## Parameter Selection Guidelines

### Alpha and Beta Values

- **α = β = 0.5**: Equivalent to Dice Loss (balanced precision and recall)
- **α < β**: Emphasizes recall/sensitivity (reduces false negatives)
- **α > β**: Emphasizes precision (reduces false positives)

#### Common Use Cases:
- **Medical Segmentation**: α = 0.3, β = 0.7 (emphasize sensitivity)
- **Object Detection**: α = 0.7, β = 0.3 (emphasize precision)
- **Balanced Classification**: α = 0.5, β = 0.5 (balanced)

### Smooth Parameter
- **Small values (0.001-1.0)**: Better gradient flow
- **Large values (10-100)**: More stable training but less sensitive

## Examples

### Basic Usage
```tcl
# Create input logits and target
set predictions [torch::tensor_create -data {0.8 0.2 0.1 0.9} -dtype float32 -device cpu]
set targets [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]

# Positional syntax
set loss [torch::tversky_loss $predictions $targets]

# Named parameter syntax
set loss [torch::tversky_loss -input $predictions -target $targets]

# CamelCase alias
set loss [torch::tverskyLoss -input $predictions -target $targets]
```

### Medical Segmentation (Sensitivity-Focused)
```tcl
set predictions [torch::tensor_create -data {0.9 0.1 0.2 0.8} -dtype float32 -device cpu]
set ground_truth [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]

# Emphasize sensitivity for medical diagnosis
set loss [torch::tversky_loss -input $predictions -target $ground_truth -alpha 0.3 -beta 0.7]
```

### Object Detection (Precision-Focused)
```tcl
set predictions [torch::tensor_create -data {0.7 0.3 0.2 0.8} -dtype float32 -device cpu]
set targets [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]

# Emphasize precision for object detection
set loss [torch::tversky_loss -input $predictions -target $targets -alpha 0.7 -beta 0.3]
```

### All Parameters
```tcl
set predictions [torch::tensor_create -data {0.6 0.4 0.2 0.8} -dtype float32 -device cpu]
set targets [torch::tensor_create -data {1.0 0.0 0.0 1.0} -dtype float32 -device cpu]

# Full parameter specification
set loss [torch::tversky_loss -input $predictions -target $targets \
    -alpha 0.4 -beta 0.6 -smooth 2.0 -reduction mean]
```

## Use Cases

### 1. Medical Image Segmentation
Tversky Loss is particularly effective for medical imaging where:
- **Missing lesions** (false negatives) are more critical than false alarms
- **Class imbalance** is severe (small lesions vs. large background)
- **Sensitivity** is more important than specificity

```tcl
# Tumor segmentation - emphasize sensitivity
set loss [torch::tversky_loss -input $predictions -target $masks -alpha 0.2 -beta 0.8]
```

### 2. Object Detection
When precision is crucial and false positives are costly:

```tcl
# Quality control - emphasize precision
set loss [torch::tversky_loss -input $detections -target $annotations -alpha 0.8 -beta 0.2]
```

### 3. Imbalanced Classification
For datasets with severe class imbalance:

```tcl
# Rare event detection
set loss [torch::tversky_loss -input $predictions -target $labels -alpha 0.3 -beta 0.7]
```

## Reduction Modes

### Mean Reduction (Default)
```tcl
set loss [torch::tversky_loss -input $predictions -target $targets -reduction mean]
# Returns scalar loss averaged over all samples
```

### Sum Reduction
```tcl
set loss [torch::tversky_loss -input $predictions -target $targets -reduction sum]
# Returns scalar loss summed over all samples
```

### No Reduction
```tcl
set loss [torch::tversky_loss -input $predictions -target $targets -reduction none]
# Returns per-sample losses
```

## Error Handling

The command provides comprehensive error handling:

```tcl
# Missing required parameter
catch {torch::tversky_loss $predictions} error
puts $error  # Will indicate missing target parameter

# Invalid tensor name
catch {torch::tversky_loss "invalid" $targets} error
puts $error  # Will indicate invalid tensor

# Invalid parameter name
catch {torch::tversky_loss -input $pred -target $targ -invalid_param 0.5} error
puts $error  # Will indicate unknown parameter

# Missing parameter value
catch {torch::tversky_loss -input $pred -target $targ -alpha} error
puts $error  # Will indicate missing parameter value
```

## Migration Guide

### From Positional to Named Parameters

**Old Syntax:**
```tcl
set loss [torch::tversky_loss $input $target 0.3 0.7 2.0 mean]
```

**New Syntax:**
```tcl
set loss [torch::tversky_loss -input $input -target $target \
    -alpha 0.3 -beta 0.7 -smooth 2.0 -reduction mean]
```

### Migration Benefits
1. **Parameter clarity**: Named parameters make code self-documenting
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Error prevention**: Reduces parameter ordering mistakes

## Performance Notes

### Training Tips
1. **Start with balanced parameters** (α = β = 0.5) then adjust based on validation metrics
2. **Monitor both precision and recall** during training
3. **Use appropriate smooth values** based on batch size and data characteristics
4. **Consider learning rate scheduling** as Tversky Loss can have different gradient characteristics

### Computational Efficiency
- The loss computation is efficient for typical tensor sizes
- Sigmoid is applied automatically to input predictions
- All computations are performed in PyTorch's optimized backend

## Comparison with Other Losses

| Loss Function | Use Case | Alpha/Beta |
|---------------|----------|------------|
| **Dice Loss** | Balanced segmentation | α = β = 0.5 |
| **Sensitivity-focused** | Medical diagnosis | α < β (e.g., 0.3, 0.7) |
| **Precision-focused** | Quality control | α > β (e.g., 0.7, 0.3) |
| **Custom balance** | Application-specific | Custom α, β values |

## Related Commands

- `torch::dice_loss` - Equivalent to Tversky Loss with α = β = 0.5
- `torch::focal_loss` - Alternative for class imbalance with different weighting strategy
- `torch::bce_loss` - Basic binary cross-entropy loss
- `torch::cross_entropy_loss` - Multi-class classification loss

## References

- Sadegh, S. S. M., et al. "Tversky loss function for image segmentation using 3D fully convolutional deep networks." *International workshop on machine learning in medical imaging*. Springer, 2017.
- The Tversky index is based on Tversky's similarity measure from cognitive psychology.

---

*This command supports both backward-compatible positional syntax and modern named parameter syntax with camelCase aliases.* 