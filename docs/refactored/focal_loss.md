# torch::focal_loss

## Overview

Computes Focal Loss, a modification of Cross-Entropy Loss designed to address class imbalance problems by focusing learning on hard examples. The Focal Loss applies a modulating factor to the standard cross-entropy loss, reducing the relative loss for well-classified examples and focusing on hard-to-classify examples.

Focal Loss is defined as: **FL(p_t) = -α(1-p_t)^γ log(p_t)**

Where:
- **p_t** is the probability of the correct class
- **α** (alpha) is a weighting factor for class imbalance
- **γ** (gamma) is the focusing parameter (modulating factor)

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::focal_loss -input tensor -target tensor ?-alpha double? ?-gamma double? ?-reduction string?
torch::focalLoss -input tensor -target tensor ?-alpha double? ?-gamma double? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::focal_loss input target ?alpha? ?gamma? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **input** | tensor | required | Input logits tensor of shape (N, C) where N is batch size, C is number of classes |
| **target** | tensor | required | Target class indices of shape (N,) with values in [0, C-1], **must be int64 dtype** |
| **alpha** | double | 1.0 | Weighting factor for rare class (balancing factor) |
| **gamma** | double | 2.0 | Focusing parameter (higher values focus more on hard examples) |
| **reduction** | string | "mean" | Specifies the reduction to apply: "none", "mean", or "sum" |

## Returns

Returns a tensor handle containing the computed focal loss:
- If reduction="none": tensor of shape (N,) with loss for each sample
- If reduction="mean": scalar tensor with mean loss
- If reduction="sum": scalar tensor with sum of losses

## Mathematical Formula

The Focal Loss is computed as:

1. **Softmax**: p_i = softmax(logits)_i
2. **Probability of correct class**: p_t = p_target_class
3. **Modulating factor**: (1-p_t)^γ
4. **Focal Loss**: FL(p_t) = -α(1-p_t)^γ log(p_t)

The key insight is that when an example is misclassified and p_t is small, the modulating factor is near 1 and the loss is unaffected. As p_t approaches 1, the factor approaches 0 and well-classified examples are down-weighted.

## Examples

### Basic Usage
```tcl
# Create input logits (2 samples, 3 classes)
set logits [torch::tensor_create {2.0 1.0 0.1 0.5 3.0 1.5} -dtype float32]
set logits [torch::tensor_reshape $logits {2 3}]

# Create target classes (must be int64)
set targets [torch::tensor_create {0 2} -dtype int64]

# Compute focal loss with default parameters
set loss [torch::focal_loss -input $logits -target $targets]
puts "Focal loss: [torch::tensor_item $loss]"
```

### Custom Alpha and Gamma Parameters
```tcl
# Higher alpha gives more weight to minority class
# Higher gamma focuses more on hard examples
set loss [torch::focal_loss -input $logits -target $targets -alpha 0.25 -gamma 2.0]
```

### Different Reduction Options
```tcl
# No reduction - returns per-sample losses
set losses [torch::focal_loss -input $logits -target $targets -reduction none]
set shape [torch::tensor_shape $losses]  ;# Should be {2}

# Sum reduction
set total_loss [torch::focal_loss -input $logits -target $targets -reduction sum]
```

### Object Detection Use Case
```tcl
# Typical parameters for object detection (from RetinaNet paper)
set alpha 0.25    ;# Balance between foreground/background
set gamma 2.0     ;# Standard focusing parameter

set loss [torch::focalLoss -input $predictions -target $ground_truth \
                           -alpha $alpha -gamma $gamma -reduction mean]
```

### Comparison with Cross-Entropy
```tcl
# Standard cross-entropy loss
set ce_loss [torch::cross_entropy_loss -input $logits -target $targets]

# Focal loss (reduces loss for easy examples)
set focal_loss [torch::focal_loss -input $logits -target $targets -alpha 1.0 -gamma 2.0]

puts "Cross-entropy: [torch::tensor_item $ce_loss]"
puts "Focal loss: [torch::tensor_item $focal_loss]"
```

## Use Cases

### 1. Object Detection
Focal Loss was originally designed for object detection tasks where there's extreme class imbalance between foreground and background classes.

```tcl
# RetinaNet-style focal loss
set loss [torch::focal_loss -input $class_predictions -target $class_targets \
                           -alpha 0.25 -gamma 2.0 -reduction mean]
```

### 2. Medical Image Classification
When certain medical conditions are rare but critical to detect correctly.

```tcl
# Higher alpha for rare disease detection
set loss [torch::focal_loss -input $medical_predictions -target $diagnoses \
                           -alpha 0.75 -gamma 1.5 -reduction mean]
```

### 3. Text Classification with Imbalanced Classes
For sentiment analysis or spam detection with unbalanced datasets.

```tcl
# Focus on misclassified examples
set loss [torch::focal_loss -input $text_features -target $labels \
                           -alpha 0.5 -gamma 3.0 -reduction mean]
```

## Parameter Guidelines

### Alpha (α) - Class Balancing
- **0.25**: Good for severe imbalance (e.g., object detection)
- **0.5**: Moderate imbalance
- **1.0**: No class weighting (balanced classes)

### Gamma (γ) - Focusing Parameter
- **0.0**: Equivalent to cross-entropy loss
- **1.0**: Mild focusing on hard examples
- **2.0**: Standard focusing (recommended starting point)
- **5.0**: Strong focusing on very hard examples

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
torch::focal_loss $input $target 0.25 2.0 1

# New named syntax
torch::focal_loss -input $input -target $target -alpha 0.25 -gamma 2.0 -reduction mean
```

### Reduction Parameter Changes
```tcl
# Old integer reduction values
# 0 = none, 1 = mean, 2 = sum

# New string reduction values
-reduction none   # Per-sample losses
-reduction mean   # Average loss
-reduction sum    # Total loss
```

## Performance Notes

- **Computational Cost**: Slightly higher than cross-entropy due to the modulating factor computation
- **Memory Usage**: Similar to cross-entropy loss
- **Convergence**: May require tuning of α and γ parameters for optimal results
- **Gradient Flow**: The modulating factor can reduce gradients for easy examples, potentially requiring learning rate adjustments

## Error Handling

The command validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::focal_loss -input $input} error
# Returns: "Required parameters -input and -target must be provided"

# Invalid reduction type
catch {torch::focal_loss -input $input -target $target -reduction invalid} error
# Returns: "Invalid reduction type: invalid"

# Invalid tensor references
catch {torch::focal_loss invalid_input $target} error
# Returns: "Invalid input tensor name"
```

## Advantages and Limitations

### Advantages
- **Class Imbalance**: Effectively handles severe class imbalance
- **Hard Example Mining**: Automatically focuses on difficult examples
- **Proven Effectiveness**: Widely used in computer vision tasks
- **Parameter Control**: Flexible tuning with α and γ parameters

### Limitations
- **Parameter Sensitivity**: Requires tuning of α and γ for optimal performance
- **Computational Overhead**: Slightly more expensive than standard cross-entropy
- **Limited to Classification**: Designed specifically for classification tasks

## See Also

- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Standard cross-entropy loss
- [`torch::bce_loss`](bce_loss.md) - Binary cross-entropy loss  
- [`torch::dice_loss`](dice_loss.md) - Alternative for segmentation tasks
- [`torch::tensor_create`](../tensor_create.md) - Creating input tensors
- [`torch::tensor_reshape`](../tensor_reshape.md) - Reshaping tensors

## References

1. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). "Focal Loss for Dense Object Detection" - Original paper introducing Focal Loss
2. RetinaNet: Single-stage object detection architecture using Focal Loss
3. PyTorch Focal Loss implementation for reference 