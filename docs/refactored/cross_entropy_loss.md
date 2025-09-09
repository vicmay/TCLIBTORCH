# torch::cross_entropy_loss / torch::crossEntropyLoss

## Overview
Computes the cross-entropy loss between input logits and target class indices. This is widely used for multi-class classification tasks.

## Syntax

### Positional Syntax (Original)
```tcl
torch::cross_entropy_loss input_tensor target_tensor ?weight_tensor? ?reduction?
```

### Named Parameter Syntax (New)
```tcl
torch::cross_entropy_loss -input tensor -target tensor ?-weight tensor? ?-reduction string?
torch::crossEntropyLoss -input tensor -target tensor ?-weight tensor? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` | Tensor | Yes | - | Input tensor containing logits (raw scores) |
| `-target` | Tensor | Yes | - | Target tensor containing class indices (int64/Long) |
| `-weight` | Tensor | No | none | Class weights for unbalanced datasets |
| `-reduction` | String | No | "mean" | Reduction method: "none", "mean", or "sum" |

## Mathematical Formula

The cross-entropy loss for a single sample is computed as:

```
loss(x, class) = -log(exp(x[class]) / Σⱼ exp(x[j]))
              = -x[class] + log(Σⱼ exp(x[j]))
```

Where:
- `x` is the input logits tensor
- `class` is the target class index
- The sum is over all classes

For weighted loss:
```
loss(x, class) = weight[class] * (-x[class] + log(Σⱼ exp(x[j])))
```

## Tensor Requirements

### Input Tensor
- **Shape**: `(N, C)` where N is batch size, C is number of classes
- **Type**: Float (float32 or float64)
- **Content**: Raw logits (unnormalized scores)

### Target Tensor
- **Shape**: `(N,)` for single label per sample
- **Type**: Long (int64) - **Critical requirement**
- **Content**: Class indices in range [0, C-1]

### Weight Tensor (Optional)
- **Shape**: `(C,)` where C is number of classes
- **Type**: Float (same as input)
- **Content**: Per-class weights

## Examples

### Basic Usage
```tcl
# Create logits for 3 classes, 2 samples
set logits [torch::tensor_create {2.0 1.0 0.5 1.5 3.0 0.1} float32]
set logits [torch::tensor_reshape $logits {2 3}]

# Create target class indices (MUST be int64)
set targets [torch::tensor_create {0 2} int64]

# Compute loss - both syntaxes equivalent
set loss1 [torch::cross_entropy_loss $logits $targets]
set loss2 [torch::cross_entropy_loss -input $logits -target $targets]
set loss3 [torch::crossEntropyLoss -input $logits -target $targets]

puts "Loss: [torch::tensor_item $loss1]"
```

### With Class Weights
```tcl
# Create unbalanced dataset logits
set logits [torch::tensor_create {1.0 2.0 0.5 0.8 1.5 2.1} float32]
set logits [torch::tensor_reshape $logits {2 3}]
set targets [torch::tensor_create {0 1} int64]

# Create class weights (higher weight for rare classes)
set weights [torch::tensor_create {2.0 1.0 3.0} float32]

# Compute weighted loss
set loss [torch::cross_entropy_loss -input $logits -target $targets -weight $weights]
puts "Weighted loss: [torch::tensor_item $loss]"
```

### Different Reductions
```tcl
set logits [torch::tensor_create {1.0 2.0 0.5 0.8 1.5 2.1} float32]
set logits [torch::tensor_reshape $logits {2 3}]
set targets [torch::tensor_create {0 1} int64]

# Mean reduction (default)
set loss_mean [torch::cross_entropy_loss -input $logits -target $targets -reduction mean]

# Sum reduction
set loss_sum [torch::cross_entropy_loss -input $logits -target $targets -reduction sum]

# No reduction (per-sample losses)
set loss_none [torch::cross_entropy_loss -input $logits -target $targets -reduction none]

puts "Mean loss: [torch::tensor_item $loss_mean]"
puts "Sum loss: [torch::tensor_item $loss_sum]"
puts "Per-sample losses: [torch::tensor_data $loss_none]"
```

## Return Value
Returns a tensor handle containing the computed loss:
- **Scalar tensor** when reduction is "mean" or "sum"
- **Vector tensor** when reduction is "none" (shape: `(N,)`)

## Error Handling

### Common Errors
```tcl
# Error: Target tensor must be int64 (Long)
set targets [torch::tensor_create {0 1} float32]  ; # Wrong dtype
catch {torch::cross_entropy_loss -input $logits -target $targets} error
# Error: expected scalar type Long but found Float

# Error: Missing required parameters
catch {torch::cross_entropy_loss -input $logits} error
# Error: Required parameters -input and -target must be provided

# Error: Invalid parameter
catch {torch::cross_entropy_loss -input $logits -target $targets -invalid value} error
# Error: Unknown parameter: -invalid
```

## Use Cases

### 1. Multi-class Image Classification
```tcl
# ImageNet-style classification (1000 classes)
set logits [torch::zeros {32 1000}]  ; # Batch of 32 images
set targets [torch::randint 0 1000 {32} int64]  ; # Random class labels
set loss [torch::crossEntropyLoss -input $logits -target $targets]
```

### 2. Text Classification
```tcl
# Sentiment analysis (3 classes: negative, neutral, positive)
set logits [torch::randn {64 3}]  ; # Batch of 64 texts
set targets [torch::randint 0 3 {64} int64]  ; # Sentiment labels
set loss [torch::crossEntropyLoss -input $logits -target $targets]
```

### 3. Training Loop with Backpropagation
```tcl
# Forward pass
set predictions [torch::linear $input $weight $bias]
set loss [torch::crossEntropyLoss -input $predictions -target $labels]

# Backward pass
torch::tensor_backward $loss
```

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old positional syntax
set loss [torch::cross_entropy_loss $input $target $weight "sum"]

# New named syntax (equivalent)
set loss [torch::cross_entropy_loss -input $input -target $target -weight $weight -reduction sum]

# CamelCase alias
set loss [torch::crossEntropyLoss -input $input -target $target -weight $weight -reduction sum]
```

### Target Tensor Creation
```tcl
# Ensure targets are int64 (critical!)
set targets [torch::tensor_create {0 1 2} int64]  ; # Correct
# NOT: set targets [torch::tensor_create {0 1 2} float32]  ; # Wrong!
```

## Performance Notes

1. **Input preprocessing**: Logits should be raw scores (no softmax needed)
2. **Target format**: Always use int64 for target tensors
3. **Memory efficiency**: Use "mean" reduction for batch training
4. **Numerical stability**: PyTorch handles log-sum-exp trick internally

## Cross-References
- [`torch::nll_loss`](nll_loss.md) - Negative log-likelihood loss (used internally)
- [`torch::softmax`](softmax.md) - Softmax activation function  
- [`torch::log_softmax`](log_softmax.md) - Log-softmax activation
- [`torch::bce_loss`](bce_loss.md) - Binary cross-entropy for binary classification

## See Also
- [PyTorch CrossEntropyLoss Documentation](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [Understanding Cross-Entropy Loss](https://pytorch.org/docs/stable/nn.html#crossentropyloss) 