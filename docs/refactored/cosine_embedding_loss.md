# torch::cosine_embedding_loss

## Overview
Computes the cosine embedding loss between two input tensors using cosine similarity and target labels.

## Syntax

### Current Syntax (Positional)
```tcl
torch::cosine_embedding_loss input1 input2 target ?margin? ?reduction?
```

### New Syntax (Named Parameters)
```tcl
torch::cosine_embedding_loss -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?
```

### CamelCase Alias
```tcl
torch::cosineEmbeddingLoss -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| input1 | tensor | Yes | - | First input tensor (2D when target is 1D) |
| input2 | tensor | Yes | - | Second input tensor (same shape as input1) |
| target | tensor | Yes | - | Target labels (1 for similar, -1 for dissimilar) |
| margin | double | No | 0.0 | Margin for the loss computation |
| reduction | string | No | "mean" | Specifies reduction: "none", "mean", or "sum" |

## Returns
Returns a tensor handle containing the computed cosine embedding loss.

## Examples

### Basic Usage
```tcl
# Create 2D input tensors (batch_size=1, features=3)
set input1 [torch::tensor_create {1.0 2.0 3.0} -dtype float32]
set input1 [torch::tensor_reshape $input1 {1 3}]
set input2 [torch::tensor_create {2.0 3.0 4.0} -dtype float32]
set input2 [torch::tensor_reshape $input2 {1 3}]

# Create target label (1 for similar pairs, -1 for dissimilar)
set target [torch::tensor_create {1.0} -dtype float32]

# Positional syntax
set loss1 [torch::cosine_embedding_loss $input1 $input2 $target]

# Named parameter syntax
set loss2 [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target]

# CamelCase alias
set loss3 [torch::cosineEmbeddingLoss -input1 $input1 -input2 $input2 -target $target]

puts "Loss: [torch::tensor_item $loss1]"
```

### Batch Processing
```tcl
# Process multiple pairs at once
set input1 [torch::tensor_create {1.0 0.0 0.0 0.0 1.0 0.0} -dtype float32]
set input1 [torch::tensor_reshape $input1 {2 3}]  ; # 2 samples, 3 features
set input2 [torch::tensor_create {0.0 1.0 0.0 1.0 0.0 0.0} -dtype float32]
set input2 [torch::tensor_reshape $input2 {2 3}]

# Mixed targets: first pair similar, second dissimilar
set target [torch::tensor_create {1.0 -1.0} -dtype float32]

set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target]
puts "Batch loss: [torch::tensor_item $loss]"
```

### With Custom Margin
```tcl
# Using a margin to control the loss behavior
set input1 [torch::tensor_create {1.0 0.0 0.0} -dtype float32]
set input1 [torch::tensor_reshape $input1 {1 3}]
set input2 [torch::tensor_create {0.0 1.0 0.0} -dtype float32]
set input2 [torch::tensor_reshape $input2 {1 3}]
set target [torch::tensor_create {-1.0} -dtype float32]  ; # Dissimilar

# Different margins
set loss_small [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.1]
set loss_large [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.5]

puts "Loss with margin 0.1: [torch::tensor_item $loss_small]"
puts "Loss with margin 0.5: [torch::tensor_item $loss_large]"
```

### Different Reduction Options
```tcl
set input1 [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
set input1 [torch::tensor_reshape $input1 {2 3}]
set input2 [torch::tensor_create {2.0 3.0 4.0 5.0 6.0 7.0} -dtype float32]
set input2 [torch::tensor_reshape $input2 {2 3}]
set target [torch::tensor_create {1.0 -1.0} -dtype float32]

# No reduction (returns per-sample loss)
set loss_none [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -reduction none]

# Sum reduction
set loss_sum [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -reduction sum]

# Mean reduction (default)
set loss_mean [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -reduction mean]

puts "Per-sample shape: [torch::tensor_shape $loss_none]"
puts "Sum loss: [torch::tensor_item $loss_sum]"
puts "Mean loss: [torch::tensor_item $loss_mean]"
```

## Mathematical Formula

The cosine embedding loss is computed as:

For target = 1 (similar pairs):
```
loss = 1 - cos(x1, x2)
```

For target = -1 (dissimilar pairs):
```
loss = max(0, cos(x1, x2) - margin)
```

Where:
- `cos(x1, x2)` is the cosine similarity between x1 and x2
- `margin` is the margin parameter (default 0.0)

The cosine similarity is defined as:
```
cos(x1, x2) = (x1 · x2) / (||x1|| * ||x2||)
```

## Input Requirements

- **input1**: 2D tensor of shape (N, D) where N is batch size and D is feature dimension
- **input2**: 2D tensor of same shape as input1
- **target**: 1D tensor of shape (N,) with values 1 (similar) or -1 (dissimilar)

## Error Handling

The function validates:
- All required tensors exist
- Required parameters are provided
- Parameter names are valid (for named syntax)
- Tensor shapes are compatible

```tcl
# Missing required parameter
catch {torch::cosine_embedding_loss -input1 $input1 -input2 $input2} result
puts $result  ; # "Required parameters -input1, -input2, and -target must be provided"

# Invalid parameter name
catch {torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -invalid value} result
puts $result  ; # "Unknown parameter: -invalid"
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set loss [torch::cosine_embedding_loss $input1 $input2 $target 0.3 2]

# New named parameter syntax
set loss [torch::cosine_embedding_loss -input1 $input1 -input2 $input2 -target $target -margin 0.3 -reduction "sum"]
```

### Parameter Mapping

| Position | Named Parameter | Description |
|----------|----------------|-------------|
| 1 | -input1 | First input tensor |
| 2 | -input2 | Second input tensor |
| 3 | -target | Target labels tensor |
| 4 | -margin | Margin value (optional) |
| 5 | -reduction | Reduction method (optional, use string instead of integer) |

### Reduction Parameter Changes

The reduction parameter now uses string values instead of integers:

| Old (Integer) | New (String) | Description |
|---------------|--------------|-------------|
| 0 | "none" | No reduction |
| 1 | "mean" | Mean reduction |
| 2 | "sum" | Sum reduction |

## Use Cases

1. **Similarity Learning**: Train models to distinguish similar vs dissimilar pairs
2. **Face Verification**: Compare face embeddings for verification
3. **Metric Learning**: Learn embeddings where similar items are close
4. **Contrastive Learning**: Train representations using positive/negative pairs

## Notes

- Input tensors must be 2D when target is 1D
- Target values should be exactly 1.0 or -1.0
- The margin parameter only affects dissimilar pairs (target = -1)
- Cosine similarity is computed along the feature dimension
- For numerical stability, ensure input vectors are not zero vectors

## See Also
- [torch::bce_loss](bce_loss.md) - Binary cross entropy loss
- [torch::margin_ranking_loss](margin_ranking_loss.md) - Margin ranking loss
- [torch::triplet_margin_loss](triplet_margin_loss.md) - Triplet margin loss
- [torch::mse_loss](mse_loss.md) - Mean squared error loss 