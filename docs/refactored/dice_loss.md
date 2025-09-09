# torch::dice_loss / torch::diceLoss

## Overview
Computes the Dice loss for binary and multi-class segmentation tasks. The Dice loss is particularly useful for medical image segmentation where class imbalance is common. It measures the overlap between predicted and ground truth regions.

## Syntax

### Positional Syntax (Original)
```tcl
torch::dice_loss input_tensor target_tensor ?smooth? ?reduction?
```

### Named Parameter Syntax (New)
```tcl
torch::dice_loss -input tensor -target tensor ?-smooth double? ?-reduction string?
torch::diceLoss -input tensor -target tensor ?-smooth double? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | Tensor | Yes | - | Input tensor containing logits (before sigmoid) |
| `-target` | Tensor | Yes | - | Target tensor containing binary masks (0 or 1) |
| `-smooth` | Double | No | 1.0 | Smoothing factor to avoid division by zero |
| `-reduction` | String | No | "mean" | Reduction method: "none", "mean", or "sum" |

## Mathematical Formula

The Dice coefficient is computed as:

```
Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
```

Where:
- `X` is the predicted segmentation (after sigmoid)
- `Y` is the ground truth segmentation
- `|X ∩ Y|` is the intersection (element-wise product)
- `|X|` and `|Y|` are the sums of predicted and ground truth respectively

The Dice loss is:
```
Dice Loss = 1 - Dice Coefficient
```

## Tensor Requirements

### Input Tensor
- **Shape**: Any shape (typically `(N, C, H, W)` for 2D segmentation)
- **Type**: Float (float32 or float64)
- **Content**: Raw logits (will be passed through sigmoid internally)

### Target Tensor
- **Shape**: Same as input tensor
- **Type**: Float (float32 or float64)
- **Content**: Binary masks with values 0.0 or 1.0

## Examples

### Basic Binary Segmentation
```tcl
# Create predicted segmentation logits (2x2 image)
set predictions [torch::tensor_create {2.0 -1.0 1.5 -0.5} float32]
set predictions [torch::tensor_reshape $predictions {2 2}]

# Create ground truth binary mask
set ground_truth [torch::tensor_create {1.0 0.0 1.0 0.0} float32]
set ground_truth [torch::tensor_reshape $ground_truth {2 2}]

# Compute Dice loss - both syntaxes equivalent
set loss1 [torch::dice_loss $predictions $ground_truth]
set loss2 [torch::dice_loss -input $predictions -target $ground_truth]
set loss3 [torch::diceLoss -input $predictions -target $ground_truth]

puts "Dice loss: [torch::tensor_item $loss1]"
```

### Medical Image Segmentation
```tcl
# Brain tumor segmentation example
set brain_scan [torch::randn {1 1 256 256}]  ; # Single channel brain MRI
set tumor_mask [torch::zeros {1 1 256 256}]  ; # Ground truth tumor mask

# Simulate some tumor regions
set tumor_region [torch::ones {1 1 50 50}]
# ... (in practice, load actual medical imaging data)

# Compute segmentation loss with custom smoothing
set loss [torch::diceLoss -input $brain_scan -target $tumor_mask -smooth 2.0]
puts "Segmentation loss: [torch::tensor_item $loss]"
```

### Multi-class Segmentation (per-class)
```tcl
# For multi-class segmentation, compute Dice loss for each class separately
set num_classes 3
set predictions [torch::randn {1 $num_classes 64 64}]  ; # 3-class predictions

for {set class 0} {$class < $num_classes} {incr class} {
    # Extract predictions and ground truth for current class
    set class_pred [torch::tensor_select $predictions 1 $class]
    set class_target [torch::tensor_create_class_mask $class]  ; # Hypothetical function
    
    set class_loss [torch::diceLoss -input $class_pred -target $class_target]
    puts "Class $class Dice loss: [torch::tensor_item $class_loss]"
}
```

### With Different Smoothing Values
```tcl
set pred [torch::tensor_create {1.0 -0.5 2.0 0.0} float32]
set pred [torch::tensor_reshape $pred {2 2}]
set target [torch::tensor_create {1.0 0.0 1.0 0.0} float32]
set target [torch::tensor_reshape $target {2 2}]

# Compare different smoothing values
set loss_smooth1 [torch::dice_loss -input $pred -target $target -smooth 1.0]
set loss_smooth5 [torch::dice_loss -input $pred -target $target -smooth 5.0]

puts "Smooth 1.0: [torch::tensor_item $loss_smooth1]"
puts "Smooth 5.0: [torch::tensor_item $loss_smooth5]"
```

### Batch Processing
```tcl
# Process multiple segmentation images in a batch
set batch_predictions [torch::randn {8 1 128 128}]  ; # 8 images
set batch_targets [torch::zeros {8 1 128 128}]      ; # 8 ground truth masks

# Compute loss for entire batch
set batch_loss [torch::diceLoss -input $batch_predictions -target $batch_targets -reduction mean]
puts "Average batch loss: [torch::tensor_item $batch_loss]"

# Compute per-image losses
set per_image_losses [torch::diceLoss -input $batch_predictions -target $batch_targets -reduction none]
puts "Per-image losses shape: [torch::tensor_shape $per_image_losses]"
```

## Return Value
Returns a tensor handle containing the computed Dice loss:
- **Scalar tensor** when reduction is "mean" or "sum"
- **Vector tensor** when reduction is "none"

## Error Handling

### Common Errors
```tcl
# Error: Missing required parameters
catch {torch::dice_loss -input $predictions} error
# Error: Required parameters -input and -target must be provided

# Error: Invalid parameter
catch {torch::dice_loss -input $pred -target $target -invalid value} error
# Error: Unknown parameter: -invalid

# Error: Invalid smooth parameter
catch {torch::dice_loss -input $pred -target $target -smooth "invalid"} error
# Error: Invalid smooth parameter value
```

## Use Cases

### 1. Medical Image Segmentation
```tcl
# Organ segmentation in CT scans
set ct_scan [torch::load_medical_image "patient_ct.nii"]
set organ_mask [torch::load_ground_truth "organ_mask.nii"]
set loss [torch::diceLoss -input $ct_scan -target $organ_mask -smooth 1.0]
```

### 2. Semantic Segmentation
```tcl
# Road segmentation in autonomous driving
set camera_image [torch::load_image "road_image.jpg"]
set road_mask [torch::load_mask "road_mask.png"]
set loss [torch::diceLoss -input $camera_image -target $road_mask]
```

### 3. Training Loop Integration
```tcl
# Training a segmentation model
for {set epoch 0} {$epoch < 100} {incr epoch} {
    set predictions [torch::forward_pass $model $input_batch]
    set dice_loss [torch::diceLoss -input $predictions -target $target_batch]
    
    # Backward pass
    torch::tensor_backward $dice_loss
    torch::optimizer_step $optimizer
}
```

### 4. Class Imbalance Handling
```tcl
# For highly imbalanced segmentation (e.g., small tumors)
set loss [torch::diceLoss -input $predictions -target $targets -smooth 0.1]
# Smaller smooth value makes the loss more sensitive to small objects
```

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old positional syntax
set loss [torch::dice_loss $input $target 2.0 1]

# New named syntax (equivalent)
set loss [torch::dice_loss -input $input -target $target -smooth 2.0 -reduction mean]

# CamelCase alias
set loss [torch::diceLoss -input $input -target $target -smooth 2.0 -reduction mean]
```

### Reduction Parameter Changes
```tcl
# Old integer-based reduction
set loss [torch::dice_loss $input $target 1.0 0]  ; # 0 = none
set loss [torch::dice_loss $input $target 1.0 1]  ; # 1 = mean
set loss [torch::dice_loss $input $target 1.0 2]  ; # 2 = sum

# New string-based reduction
set loss [torch::dice_loss -input $input -target $target -reduction none]
set loss [torch::dice_loss -input $input -target $target -reduction mean]
set loss [torch::dice_loss -input $input -target $target -reduction sum]
```

## Performance Notes

1. **Input preprocessing**: Logits are automatically passed through sigmoid
2. **Smoothing**: Higher smoothing values provide more stability but less sensitivity
3. **Memory efficiency**: Use "mean" reduction for batch training
4. **Class imbalance**: Dice loss naturally handles class imbalance better than cross-entropy

## Advantages of Dice Loss

1. **Robust to class imbalance**: Works well when background >> foreground
2. **Direct optimization**: Optimizes the metric you care about (overlap)
3. **Smooth gradients**: Provides stable training signals
4. **Scale invariant**: Works regardless of object size

## Limitations

1. **Requires binary targets**: Not directly applicable to multi-class (use per-class)
2. **Gradient issues**: Can have vanishing gradients for perfect predictions
3. **Smoothing dependency**: Results depend on smoothing parameter choice

## Cross-References
- [`torch::focal_loss`](focal_loss.md) - Alternative for class imbalance
- [`torch::tversky_loss`](tversky_loss.md) - Generalization of Dice loss
- [`torch::bce_loss`](bce_loss.md) - Binary cross-entropy alternative
- [`torch::sigmoid`](sigmoid.md) - Applied internally to logits

## See Also
- [Dice Coefficient on Wikipedia](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [Medical Image Segmentation with Dice Loss](https://arxiv.org/abs/1606.04797)
- [PyTorch Segmentation Losses](https://pytorch.org/docs/stable/nn.html#loss-functions) 