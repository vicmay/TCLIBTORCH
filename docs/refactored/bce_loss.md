# torch::bce_loss

## Overview
Computes the binary cross entropy loss between predicted probabilities and binary targets.

## Syntax

### Current Syntax (Positional)
```tcl
torch::bce_loss input target ?weight? ?reduction?
```

### New Syntax (Named Parameters)
```tcl
torch::bce_loss -input tensor -target tensor ?-weight tensor? ?-reduction string?
```

### CamelCase Alias
```tcl
torch::bceLoss -input tensor -target tensor ?-weight tensor? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| input | tensor | Yes | - | Input tensor containing predicted probabilities (values should be in [0,1]) |
| target | tensor | Yes | - | Target tensor containing binary labels (0 or 1) |
| weight | tensor | No | none | Manual rescaling weight given to the loss of each batch element |
| reduction | string | No | "mean" | Specifies reduction: "none", "mean", or "sum" |

## Returns
Returns a tensor handle containing the computed binary cross entropy loss.

## Examples

### Basic Usage
```tcl
# Create input probabilities (values between 0 and 1)
set input [torch::tensor_create {0.8 0.2 0.3 0.9} -dtype float32]

# Create binary targets (0 or 1)
set target [torch::tensor_create {1.0 0.0 0.0 1.0} -dtype float32]

# Positional syntax
set loss1 [torch::bce_loss $input $target]

# Named parameter syntax
set loss2 [torch::bce_loss -input $input -target $target]

# CamelCase alias
set loss3 [torch::bceLoss -input $input -target $target]

puts "Loss: [torch::tensor_item $loss1]"  ; # ~0.227
```

### With Reduction Options
```tcl
set input [torch::tensor_create {0.7 0.3 0.6 0.4} -dtype float32]
set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]

# No reduction (returns per-element loss)
set loss_none [torch::bce_loss -input $input -target $target -reduction none]

# Sum reduction
set loss_sum [torch::bce_loss -input $input -target $target -reduction sum]

# Mean reduction (default)
set loss_mean [torch::bce_loss -input $input -target $target -reduction mean]

puts "Shape with none reduction: [torch::tensor_shape $loss_none]"
puts "Sum loss: [torch::tensor_item $loss_sum]"
puts "Mean loss: [torch::tensor_item $loss_mean]"
```

### With Custom Weights
```tcl
set input [torch::tensor_create {0.8 0.2 0.6 0.4} -dtype float32]
set target [torch::tensor_create {1.0 0.0 1.0 0.0} -dtype float32]

# Create weight tensor (higher weight for positive examples)
set weights [torch::tensor_create {2.0 1.0 2.0 1.0} -dtype float32]

# Apply weighted BCE loss
set weighted_loss [torch::bce_loss -input $input -target $target -weight $weights]

puts "Weighted loss: [torch::tensor_item $weighted_loss]"
```

## Mathematical Formula

The binary cross entropy loss is computed as:

```
loss(x, y) = -[y * log(x) + (1 - y) * log(1 - x)]
```

Where:
- `x` is the input (predicted probabilities)
- `y` is the target (true binary labels)

With reduction:
- `none`: No reduction, returns per-element loss
- `mean`: Returns mean of all losses
- `sum`: Returns sum of all losses

## Error Handling

The function validates:
- Both input and target tensors exist
- Required parameters are provided
- Parameter names are valid (for named syntax)

```tcl
# Missing required parameter
catch {torch::bce_loss -input $input} result
puts $result  ; # "Required parameters -input and -target must be provided"

# Invalid parameter name
catch {torch::bce_loss -input $input -target $target -invalid value} result
puts $result  ; # "Unknown parameter: -invalid"
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set loss [torch::bce_loss $input $target $weight "sum"]

# New named parameter syntax
set loss [torch::bce_loss -input $input -target $target -weight $weight -reduction "sum"]
```

### Parameter Mapping

| Position | Named Parameter | Description |
|----------|----------------|-------------|
| 1 | -input | Input tensor |
| 2 | -target | Target tensor |
| 3 | -weight | Weight tensor (optional) |
| 4 | -reduction | Reduction method (optional) |

## Notes

- Input values should be probabilities in the range [0, 1]
- For raw logits, use `torch::bce_with_logits_loss` instead
- The input and target tensors must have the same shape
- Binary cross entropy is suitable for binary classification tasks
- Ensure proper gradient flow by setting `requires_grad` on input tensors

## See Also
- [torch::bce_with_logits_loss](bce_with_logits_loss.md) - BCE loss with logits
- [torch::cross_entropy_loss](cross_entropy_loss.md) - Multi-class cross entropy
- [torch::mse_loss](mse_loss.md) - Mean squared error loss
- [torch::nll_loss](nll_loss.md) - Negative log likelihood loss 