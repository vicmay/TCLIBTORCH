# torch::margin_ranking_loss

Creates a criterion that measures the loss given inputs and a label tensor containing either 1 or -1. Used for ranking tasks, where the goal is to learn relative ordering between pairs of inputs.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::margin_ranking_loss -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?
torch::marginRankingLoss -input1 tensor -input2 tensor -target tensor ?-margin double? ?-reduction string?
```

### Positional Syntax (Legacy)
```tcl
torch::margin_ranking_loss input1 input2 target ?margin? ?reduction?
torch::marginRankingLoss input1 input2 target ?margin? ?reduction?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-input1` | tensor | - | First input tensor (required) |
| `-input2` | tensor | - | Second input tensor (required) |
| `-target` | tensor | - | Target tensor with values 1 or -1 (required) |
| `-margin` | double | 0.0 | Margin value for the ranking constraint |
| `-reduction` | string | "mean" | Reduction type: "none", "mean", or "sum" |

## Mathematical Definition

The margin ranking loss is defined as:

```
loss(x1, x2, y) = max(0, -y * (x1 - x2) + margin)
```

where:
- `x1` and `x2` are the input tensors
- `y` is the target tensor (1 or -1)
- `margin` is the margin parameter

### Behavior by Target Value:

**When target = 1:**
- We want `x1 > x2` (first input should be ranked higher)
- Loss = `max(0, -(x1 - x2) + margin) = max(0, -diff + margin)`
- If `x1 > x2` by at least `margin`, loss = 0
- Otherwise, loss increases as the difference decreases

**When target = -1:**
- We want `x2 > x1` (second input should be ranked higher)  
- Loss = `max(0, (x1 - x2) + margin) = max(0, diff + margin)`
- If `x2 > x1` by at least `margin`, loss = 0
- Otherwise, loss increases as the difference decreases

## Return Value

Returns a tensor containing the margin ranking loss. The shape depends on the reduction parameter:
- `"none"`: Returns a tensor of the same shape as input, containing element-wise losses
- `"mean"`: Returns a scalar tensor with the mean loss
- `"sum"`: Returns a scalar tensor with the sum of all losses

## Examples

### Basic Usage with Named Parameters (Recommended)
```tcl
# Create input tensors and targets
set input1 [torch::tensor_create {2.0 3.0 1.0} float32]  ; # First rankings
set input2 [torch::tensor_create {1.0 2.0 3.0} float32]  ; # Second rankings
set target [torch::tensor_create {1 1 -1} float32]       ; # 1=prefer input1, -1=prefer input2

# Compute margin ranking loss
set loss [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target]
set loss_value [torch::tensor_item $loss]
puts "Margin ranking loss: $loss_value"
```

### Custom Margin and Reduction
```tcl
# Use custom margin to enforce stronger ranking constraints
set loss_margin [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.5]
set loss_value [torch::tensor_item $loss_margin]
puts "Margin ranking loss (margin=0.5): $loss_value"

# No reduction - get per-element losses
set loss_none [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -reduction "none"]
set shape [torch::tensor_shape $loss_none]
puts "Per-element losses shape: $shape"
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::marginRankingLoss -input1 $input1 -input2 $input2 -target $target -margin 1.0]
set loss_value [torch::tensor_item $loss]
puts "Margin ranking loss (camelCase): $loss_value"
```

### Positional Syntax (Legacy)
```tcl
# Basic positional usage
set loss1 [torch::margin_ranking_loss $input1 $input2 $target]

# With custom parameters
set loss2 [torch::margin_ranking_loss $input1 $input2 $target 0.8 "sum"]
```

## Mathematical Examples

### Perfect Ranking (Target = 1)
```tcl
# When target=1, we want input1 > input2
set input1 [torch::tensor_create {3.0} float32]  ; # Higher value
set input2 [torch::tensor_create {1.0} float32]  ; # Lower value
set target [torch::tensor_create {1} float32]    ; # Prefer input1

set loss [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.0]
set loss_value [torch::tensor_item $loss]
# loss_value ≈ 0 (perfect ranking, no margin violation)
```

### Reverse Ranking (Target = -1)
```tcl
# When target=-1, we want input2 > input1
set input1 [torch::tensor_create {1.0} float32]  ; # Lower value
set input2 [torch::tensor_create {3.0} float32]  ; # Higher value
set target [torch::tensor_create {-1} float32]   ; # Prefer input2

set loss [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.0]
set loss_value [torch::tensor_item $loss]
# loss_value ≈ 0 (perfect reverse ranking)
```

### Margin Violation Example
```tcl
# Test margin effect - inputs are correctly ordered but within margin
set input1 [torch::tensor_create {2.0} float32]
set input2 [torch::tensor_create {1.5} float32]  ; # diff = 0.5
set target [torch::tensor_create {1} float32]    ; # Prefer input1

# With margin 0.0 - no violation
set loss1 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.0]
# loss1 ≈ 0

# With margin 1.0 - violation occurs
set loss2 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 1.0]
# loss2 ≈ 0.5 (margin violation: max(0, -0.5 + 1.0) = 0.5)
```

## Use Cases

### Learning to Rank
```tcl
# Train a model to rank search results or recommendations
set doc_scores1 [torch::tensor_create {0.8 0.6 0.9} float32]  ; # Document relevance scores
set doc_scores2 [torch::tensor_create {0.7 0.8 0.5} float32]  ; # Alternative scoring
set preferences [torch::tensor_create {1 -1 1} float32]       ; # Human preferences

set ranking_loss [torch::margin_ranking_loss -input1 $doc_scores1 -input2 $doc_scores2 -target $preferences]
```

### Contrastive Learning
```tcl
# Learn embeddings where similar items have higher scores than dissimilar ones
set anchor_scores [torch::tensor_create {0.9 0.7 0.8} float32]    ; # Similarity to anchor
set negative_scores [torch::tensor_create {0.3 0.4 0.2} float32]  ; # Negative examples
set targets [torch::tensor_create {1 1 1} float32]               ; # Prefer anchor matches

set contrastive_loss [torch::margin_ranking_loss -input1 $anchor_scores -input2 $negative_scores -target $targets -margin 0.2]
```

### Preference Learning
```tcl
# Learn user preferences between items
set item_scores_a [torch::tensor_create {4.2 3.1 5.0} float32]  ; # Predicted scores for item A
set item_scores_b [torch::tensor_create {3.8 4.2 3.5} float32]  ; # Predicted scores for item B
set user_prefs [torch::tensor_create {1 -1 1} float32]          ; # User preferences

set pref_loss [torch::margin_ranking_loss -input1 $item_scores_a -input2 $item_scores_b -target $user_prefs]
```

## Margin Parameter Effects

The margin parameter controls how strictly the ranking constraint is enforced:

- **Margin = 0.0**: Only penalizes incorrect ordering
- **Small margin (0.1-0.5)**: Requires small separation between correct and incorrect rankings
- **Large margin (1.0+)**: Requires strong separation, leading to more confident predictions

### Margin Comparison
```tcl
set input1 [torch::tensor_create {2.1} float32]
set input2 [torch::tensor_create {2.0} float32]  ; # Very close values
set target [torch::tensor_create {1} float32]

# Different margin values
set loss_0 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.0]   ; # ≈ 0
set loss_02 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.2]  ; # ≈ 0.1
set loss_05 [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.5]  ; # ≈ 0.4
```

## Reduction Options

| Reduction | Description | Output Shape | Use Case |
|-----------|-------------|--------------|----------|
| `"none"` | No reduction, return per-element losses | Same as input | Analysis of individual ranking errors |
| `"mean"` | Average of all losses | Scalar | Standard training loss |
| `"sum"` | Sum of all losses | Scalar | Batch-sensitive training |

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameters
catch {torch::margin_ranking_loss -input1 $input1 -input2 $input2} error
# Error: Required parameters -input1, -input2, and -target must be provided

# Invalid tensor names
catch {torch::margin_ranking_loss "invalid" $input2 $target} error
# Error: Invalid input1 tensor name

# Unknown parameters
catch {torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -unknown "value"} error
# Error: Unknown parameter: -unknown
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set loss [torch::margin_ranking_loss $input1 $input2 $target 0.5 "mean"]

# New named parameter syntax
set loss [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin 0.5 -reduction "mean"]
```

### Advantages of Named Parameters

1. **Clarity**: Parameter names make the ranking relationship explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Defaults**: Optional parameters can be omitted
4. **Validation**: Better error messages for incorrect usage

## Comparison with Other Ranking Losses

| Loss Function | Purpose | Inputs | Characteristics |
|---------------|---------|---------|-----------------|
| Margin Ranking | Pairwise ranking | 2 tensors + target | Simple, interpretable margin |
| Triplet Margin | Triplet ranking | 3 tensors (anchor/pos/neg) | Contrastive learning focused |
| Hinge Embedding | Binary similarity | 2 tensors + binary target | Binary classification oriented |

## Performance Notes

- Margin ranking loss is computationally efficient for pairwise comparisons
- Both positional and named parameter syntaxes have identical performance
- The `"none"` reduction is useful for analyzing individual ranking errors
- Memory usage scales linearly with tensor size

## Mathematical Properties

### Convexity
The margin ranking loss is convex, making it suitable for gradient-based optimization.

### Gradient Behavior
- When the constraint is satisfied (loss = 0), gradient = 0
- When violated, provides clear directional gradients

### Margin Sensitivity
```tcl
# Demonstrate how loss changes with margin
set input1 [torch::tensor_create {1.5} float32]
set input2 [torch::tensor_create {1.0} float32]
set target [torch::tensor_create {1} float32]

# Different margins show increasing constraint strength
foreach margin {0.0 0.5 1.0 1.5} {
    set loss [torch::margin_ranking_loss -input1 $input1 -input2 $input2 -target $target -margin $margin]
    set loss_value [torch::tensor_item $loss]
    puts "Margin $margin: Loss = $loss_value"
}
```

## See Also

- `torch::triplet_margin_loss` - Triplet-based ranking loss
- `torch::hinge_embedding_loss` - Binary similarity loss
- `torch::cosine_embedding_loss` - Cosine similarity-based loss
- `torch::tensor_create` - Creating tensors
- `torch::tensor_item` - Extracting scalar values 