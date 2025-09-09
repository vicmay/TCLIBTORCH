# torch::hinge_embedding_loss

Computes the Hinge Embedding loss between input and target tensors, typically used for learning embeddings with binary similarity labels.

## Syntax

### Positional Parameters (Original)
```tcl
torch::hinge_embedding_loss input target ?margin? ?reduction?
```

### Named Parameters (New)
```tcl
torch::hinge_embedding_loss -input tensor -target tensor ?-margin double? ?-reduction string?
```

### CamelCase Alias
```tcl
torch::hingeEmbeddingLoss -input tensor -target tensor ?-margin double? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | tensor | Yes | Input tensor containing similarity scores |
| `target` | tensor | Yes | Target tensor with values +1 (similar) or -1 (dissimilar) |
| `margin` | double | No | Margin value for dissimilar pairs (default: 1.0) |
| `reduction` | string | No | Reduction mode: "mean" (default), "sum", or "none" |

## Mathematical Definition

The Hinge Embedding loss is defined as:

For input tensor `x`, target tensor `y`, and margin `m`:

**Individual Loss:**
```
loss(x, y) = x                      if y = 1
loss(x, y) = max(0, m - x)          if y = -1
```

**Reduction Modes:**
- `none`: No reduction, return raw losses for each sample
- `mean`: Average loss across all samples  
- `sum`: Sum of all losses

## Key Properties

1. **Binary Similarity**: Target must contain only +1 (similar) or -1 (dissimilar) values
2. **Asymmetric Behavior**: Different loss computation for similar vs dissimilar pairs
3. **Margin-based**: Uses margin to define when dissimilar pairs are "far enough apart"
4. **Non-negativity**: For dissimilar pairs (target=-1), loss is always ≥ 0
5. **Identity for Similar**: For similar pairs (target=1), loss equals input

## Examples

### Basic Usage (Positional)
```tcl
# Create similarity scores for 4 pairs
set input [torch::tensor_create {1.0 -1.5 2.0 -0.5} float32]

# Create similarity targets: 1=similar, -1=dissimilar
set target [torch::tensor_create {1 -1 1 -1} float32]

# Compute hinge embedding loss (mean reduction by default)
set loss [torch::hinge_embedding_loss $input $target]
puts "Hinge Embedding Loss: [torch::tensor_item $loss]"
```

### Named Parameters with Custom Margin
```tcl
# Use larger margin for dissimilar pairs
set loss [torch::hinge_embedding_loss -input $input -target $target -margin 2.0]

# Different reduction modes
set loss_sum [torch::hinge_embedding_loss -input $input -target $target -reduction sum]
set loss_none [torch::hinge_embedding_loss -input $input -target $target -reduction none]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::hingeEmbeddingLoss -input $input -target $target -margin 1.5]
```

## Reduction Modes

### Mean (Default)
```tcl
set loss_mean [torch::hinge_embedding_loss -input $input -target $target -reduction mean]
# Returns scalar: average loss across all pairs
```

### Sum
```tcl
set loss_sum [torch::hinge_embedding_loss -input $input -target $target -reduction sum]
# Returns scalar: total loss across all pairs
```

### None
```tcl
set loss_none [torch::hinge_embedding_loss -input $input -target $target -reduction none]
# Returns tensor with individual losses for each pair
```

## Use Cases

### 1. Siamese Networks
```tcl
# Learning embeddings where similar pairs should have high similarity
set embedding1 [model_forward $image1]
set embedding2 [model_forward $image2]

# Compute cosine similarity
set similarity [torch::cosine_similarity $embedding1 $embedding2]
set target [torch::tensor_create {1}]  ; # Similar pair

set loss [torch::hingeEmbeddingLoss -input $similarity -target $target]
```

### 2. Contrastive Learning
```tcl
# For multiple positive and negative pairs
set similarities [torch::tensor_create {0.8 -0.2 0.9 -0.5} float32]
set labels [torch::tensor_create {1 -1 1 -1} float32]  ; # Similar, Dissimilar, Similar, Dissimilar

set loss [torch::hingeEmbeddingLoss -input $similarities -target $labels -margin 1.0]
```

### 3. Ranking Loss
```tcl
# Learning to rank where higher scores indicate more similarity
set scores [torch::tensor_create {2.1 0.3 1.8 0.1} float32]
set relevance [torch::tensor_create {1 -1 1 -1} float32]

set loss [torch::hingeEmbeddingLoss -input $scores -target $relevance -margin 1.5]
```

## Mathematical Behavior

### For Similar Pairs (target = 1)
```tcl
# Loss = input (regardless of sign)
set input_pos [torch::tensor_create {2.0} float32]
set input_neg [torch::tensor_create {-0.5} float32]
set target_sim [torch::tensor_create {1} float32]

set loss_pos [torch::hingeEmbeddingLoss -input $input_pos -target $target_sim]  ; # Result: 2.0
set loss_neg [torch::hingeEmbeddingLoss -input $input_neg -target $target_sim]  ; # Result: -0.5
```

### For Dissimilar Pairs (target = -1)
```tcl
# Loss = max(0, margin - input)
set input1 [torch::tensor_create {-2.0} float32]  ; # Very different
set input2 [torch::tensor_create {0.5} float32]   ; # Somewhat similar
set target_diff [torch::tensor_create {-1} float32]

set loss1 [torch::hingeEmbeddingLoss -input $input1 -target $target_diff -margin 1.0]  ; # max(0, 1-(-2)) = 3.0
set loss2 [torch::hingeEmbeddingLoss -input $input2 -target $target_diff -margin 1.0]  ; # max(0, 1-0.5) = 0.5
```

## Margin Effects

### Small Margin
```tcl
set input [torch::tensor_create {0.5} float32]
set target [torch::tensor_create {-1} float32]

set loss_small [torch::hingeEmbeddingLoss -input $input -target $target -margin 0.5]  ; # max(0, 0.5-0.5) = 0
```

### Large Margin
```tcl
set loss_large [torch::hingeEmbeddingLoss -input $input -target $target -margin 2.0]   ; # max(0, 2.0-0.5) = 1.5
```

## Comparison with Other Loss Functions

### Hinge Embedding vs Contrastive Loss
- **Hinge Embedding**: Asymmetric loss (different formulas for similar/dissimilar)
- **Contrastive Loss**: Symmetric loss with margin for dissimilar pairs

### Hinge Embedding vs Triplet Loss
- **Hinge Embedding**: Operates on pairs with binary labels
- **Triplet Loss**: Operates on triplets (anchor, positive, negative)

### Hinge Embedding vs MSE
```tcl
# Hinge Embedding (for similarity learning)
set hinge_loss [torch::hingeEmbeddingLoss -input $similarities -target $binary_labels]

# MSE (for regression tasks)
set mse_loss [torch::mse_loss -input $predictions -target $continuous_targets]
```

## Performance Notes

1. **Computational Efficiency**: Simple element-wise operations
2. **Memory Usage**: Minimal overhead beyond input tensors
3. **Gradient Flow**: Provides clear gradients for optimization
4. **Numerical Stability**: Stable for reasonable input ranges

## Common Patterns

### Training Loop
```tcl
# Forward pass through siamese network
set features1 [model_forward $batch1]
set features2 [model_forward $batch2]

# Compute similarity
set similarities [torch::cosine_similarity $features1 $features2]

# Compute loss
set loss [torch::hingeEmbeddingLoss -input $similarities -target $batch_labels]

# Backward pass
torch::backward $loss
```

### Evaluation
```tcl
# Test similarity predictions
foreach {test_pair label} $test_data {
    set sim [compute_similarity $test_pair]
    set prediction [expr {$sim > 0 ? 1 : -1}]
    
    # Accumulate loss for evaluation
    set eval_loss [torch::hingeEmbeddingLoss -input $sim -target $label]
}
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameter
catch {torch::hinge_embedding_loss -input $tensor} error
# Error: Required parameters -input and -target must be provided

# Invalid parameter name
catch {torch::hinge_embedding_loss -input $input -target $target -invalid param} error
# Error: Unknown parameter: -invalid

# Dimension mismatch
catch {torch::hinge_embedding_loss $wrong_shape_input $target} error
# Error: Tensor dimension mismatch
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set loss [torch::hinge_embedding_loss $input $target 2.0 "sum"]
```

**New (Named):**
```tcl
set loss [torch::hinge_embedding_loss -input $input -target $target -margin 2.0 -reduction sum]
```

### Advantages of Named Parameters

1. **Self-documenting**: Parameter purpose is explicit
2. **Flexible Ordering**: Parameters can be specified in any order
3. **Optional Parameters**: Easy to specify only needed parameters
4. **Error Prevention**: Less likely to mix up parameter order

### Backward Compatibility

All existing positional syntax continues to work:
```tcl
# These all continue to work
set loss1 [torch::hinge_embedding_loss $input $target]
set loss2 [torch::hinge_embedding_loss $input $target 1.5]
set loss3 [torch::hinge_embedding_loss $input $target 1.0 "none"]
```

### Integer Reduction Compatibility

Legacy integer reduction values are still supported:
```tcl
# Old integer format (0=none, 1=mean, 2=sum)
set loss_old [torch::hinge_embedding_loss $input $target 1.0 2]

# New string format
set loss_new [torch::hingeEmbeddingLoss -input $input -target $target -reduction sum]
```

## Mathematical Properties

1. **Loss Range**: (-∞, +∞) for similar pairs, [0, +∞) for dissimilar pairs
2. **Convexity**: Convex in input for dissimilar pairs
3. **Margin Sensitivity**: Larger margins require greater separation for dissimilar pairs
4. **Identity Property**: For similar pairs, loss directly reflects input similarity
5. **Zero Loss Condition**: Dissimilar pairs have zero loss when input ≤ -margin

## Best Practices

1. **Margin Selection**: Choose margin based on expected similarity range
2. **Data Normalization**: Normalize similarities to consistent range
3. **Balanced Data**: Ensure roughly equal similar/dissimilar pairs
4. **Learning Rate**: Use appropriate learning rates for embedding dimensions
5. **Regularization**: Consider L2 regularization for embedding weights

## See Also

- `torch::cosine_similarity` - Compute cosine similarity between tensors
- `torch::triplet_margin_loss` - Triplet-based ranking loss
- `torch::margin_ranking_loss` - Ranking loss with margin
- `torch::mse_loss` - Mean squared error loss for regression 