# torch::multi_margin_loss

## Overview
Computes the multi-class margin loss for classification tasks. This loss function is particularly effective for multi-class classification problems and works by maximizing the margin between the correct class and the other classes.

## Syntax

### Current Syntax (snake_case + positional)
```tcl
torch::multi_margin_loss input target ?p? ?margin? ?reduction?
```

### New Syntax (camelCase + named parameters)  
```tcl
torch::multiMarginLoss -input input -target target -p p -margin margin -reduction reduction
```

### Alternative Alias
```tcl
torch::multiMarginLoss input target ?p? ?margin? ?reduction?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | - | Input tensor name (class scores) |
| `target` | string | Yes | - | Target tensor name (class indices) |
| `p` | int | No | 1 | Norm degree (1 or 2) |
| `margin` | double | No | 1.0 | Margin value |
| `reduction` | string/int | No | "mean" | Loss reduction method |

### Named Parameters
- `-input` - Input tensor containing class scores/logits
- `-target` - Target tensor containing class indices
- `-p` - Norm degree: 1 (L1 norm) or 2 (L2 norm)
- `-margin` - Margin value for the loss computation
- `-reduction` - Reduction method: "none", "mean", "sum" (or 0, 1, 2)

### Reduction Options
- `"none"` (0): No reduction, return per-sample loss
- `"mean"` (1): Return mean of all elements (default)
- `"sum"` (2): Return sum of all elements

## Return Value
Returns a tensor handle containing the computed multi-class margin loss.

## Mathematical Background

Multi-class margin loss is defined as:

**For p=1 (L1 norm):**
```
loss(x, y) = max(0, margin - x[y] + x[j])
```

**For p=2 (L2 norm):**
```
loss(x, y) = max(0, margin - x[y] + x[j])²
```

Where:
- `x` is the input tensor (class scores)
- `y` is the target class index
- `j` represents all classes except the target class
- The loss encourages the score of the correct class to be at least `margin` higher than other classes

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create input tensor (class scores for 2 samples, 3 classes each)
set input [torch::tensor_create -data {2.0 1.0 0.5 1.5 0.0 1.0} -dtype float32 -device cpu -requiresGrad false]

# Create target tensor (class indices)
set target [torch::tensor_create -data {0 1} -dtype int64 -device cpu -requiresGrad false]

# Compute multi-class margin loss
set loss [torch::multi_margin_loss $input $target]
puts "Multi-margin Loss: [torch::tensor_item $loss]"
```

### Named Parameter Syntax
```tcl
# Same computation using named parameters
set loss [torch::multi_margin_loss -input $input -target $target -p 1 -margin 1.0]

# With custom parameters
set loss_custom [torch::multi_margin_loss \
    -input $input \
    -target $target \
    -p 2 \
    -margin 0.5 \
    -reduction sum]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::multiMarginLoss $input $target]

# With named parameters
set loss [torch::multiMarginLoss -input $input -target $target -p 2 -reduction none]
```

### Different Norm Degrees
```tcl
# L1 norm (default)
set loss_l1 [torch::multi_margin_loss $input $target 1]

# L2 norm
set loss_l2 [torch::multi_margin_loss $input $target 2]

# Using string parameters
set loss_l1_named [torch::multi_margin_loss -input $input -target $target -p 1]
set loss_l2_named [torch::multi_margin_loss -input $input -target $target -p 2]
```

### Different Margin Values
```tcl
# Small margin
set loss_small [torch::multi_margin_loss $input $target 1 0.5]

# Large margin
set loss_large [torch::multi_margin_loss $input $target 1 2.0]

# Using named parameters
set loss_custom_margin [torch::multi_margin_loss -input $input -target $target -margin 1.5]
```

### Different Reductions
```tcl
# No reduction - return per-sample loss
set loss_none [torch::multi_margin_loss $input $target 1 1.0 0]

# Mean reduction (default)
set loss_mean [torch::multi_margin_loss $input $target 1 1.0 1]

# Sum reduction
set loss_sum [torch::multi_margin_loss $input $target 1 1.0 2]

# Using string reduction values
set loss_none_str [torch::multi_margin_loss -input $input -target $target -reduction none]
```

### Batch Classification Example
```tcl
# Example: 4 samples, 3 classes
set batch_input [torch::tensor_create -data {
    2.1 1.5 0.2
    0.8 2.3 1.1
    1.7 0.9 2.0
    1.2 1.8 0.5
} -dtype float32 -device cpu -requiresGrad false]

set batch_target [torch::tensor_create -data {0 1 2 1} -dtype int64 -device cpu -requiresGrad false]

# Compute batch loss
set batch_loss [torch::multiMarginLoss \
    -input $batch_input \
    -target $batch_target \
    -p 1 \
    -margin 1.0 \
    -reduction mean]

puts "Batch Loss: [torch::tensor_item $batch_loss]"
```

### Training Example
```tcl
# Example training scenario
proc compute_multiclass_loss {predictions targets} {
    return [torch::multiMarginLoss \
        -input $predictions \
        -target $targets \
        -p 1 \
        -margin 1.0 \
        -reduction mean]
}

# Use in training loop
set model_output [some_model $training_data]
set training_loss [compute_multiclass_loss $model_output $training_labels]
```

## Use Cases

### 1. Multi-class Classification
- **Purpose**: Standard multi-class classification with margin-based loss
- **Input**: Raw class scores or logits from model
- **Target**: Class indices (0, 1, 2, ...)

### 2. Large Margin Classification
- **Purpose**: Enforce larger separation between classes
- **Settings**: Increase margin parameter for better generalization
- **Benefits**: Often leads to more robust classifiers

### 3. Support Vector Machine (SVM) Style Classification
- **Purpose**: Neural network equivalent of SVM classification
- **Settings**: Use L2 norm (p=2) for quadratic penalty
- **Applications**: When you want SVM-like behavior in neural networks

### 4. Imbalanced Classification
- **Purpose**: Alternative to cross-entropy for imbalanced datasets
- **Benefits**: Can be more robust to class imbalance than cross-entropy
- **Settings**: Adjust margin based on class distribution

## Error Handling

```tcl
# Handle invalid tensor names
if {[catch {torch::multi_margin_loss invalid_tensor $target} error]} {
    puts "Error: $error"
}

# Handle invalid parameters
if {[catch {torch::multi_margin_loss -input $input -invalid param} error]} {
    puts "Error: $error"
}

# Handle invalid p values
if {[catch {torch::multi_margin_loss -input $input -target $target -p 3} error]} {
    puts "Error: $error"
}
```

## Performance Considerations

1. **Tensor Size**: Larger tensors require more computation time
2. **Norm Degree**: L2 norm (p=2) is slightly more expensive than L1 norm (p=1)
3. **Reduction**: "none" reduction returns larger tensors
4. **Memory**: Consider memory usage for large batch sizes

## Comparison with Other Loss Functions

### vs Cross-Entropy Loss
- **Multi-margin**: Focuses on margin between classes
- **Cross-entropy**: Focuses on probability distribution
- **When to use multi-margin**: When you want explicit margin control

### vs Hinge Loss
- **Multi-margin**: Multi-class extension of hinge loss
- **Hinge**: Binary classification margin loss
- **Relationship**: Multi-margin generalizes hinge to multiple classes

## Backward Compatibility

All existing code using the positional syntax will continue to work:

```tcl
# Old code still works
set loss [torch::multi_margin_loss $input $target]
set loss [torch::multi_margin_loss $input $target 1]
set loss [torch::multi_margin_loss $input $target 1 1.0]
set loss [torch::multi_margin_loss $input $target 1 1.0 1]
```

## Migration Guide

### From Positional to Named Parameters

**Old:**
```tcl
set loss [torch::multi_margin_loss $input $target 2 0.5 1]
```

**New:**
```tcl
set loss [torch::multi_margin_loss -input $input -target $target -p 2 -margin 0.5 -reduction mean]
# Or using camelCase
set loss [torch::multiMarginLoss -input $input -target $target -p 2 -margin 0.5 -reduction mean]
```

### Parameter Mapping
- Position 1: `input` → `-input`
- Position 2: `target` → `-target`  
- Position 3: `p` (1/2) → `-p` (1/2)
- Position 4: `margin` → `-margin`
- Position 5: `reduction` (0/1/2) → `-reduction` ("none"/"mean"/"sum")

## Common Patterns

### Classification Training
```tcl
proc train_classifier {model data labels} {
    set predictions [$model forward $data]
    set loss [torch::multiMarginLoss -input $predictions -target $labels -margin 1.0]
    return $loss
}
```

### Evaluation
```tcl
proc evaluate_model {model test_data test_labels} {
    set predictions [$model forward $test_data]
    set loss [torch::multiMarginLoss -input $predictions -target $test_labels -reduction mean]
    return [torch::tensor_item $loss]
}
```

## Related Commands

- `torch::cross_entropy_loss` - Cross-entropy loss for classification
- `torch::nll_loss` - Negative log-likelihood loss
- `torch::hinge_embedding_loss` - Binary hinge loss
- `torch::multilabel_margin_loss` - Multi-label variant
- `torch::triplet_margin_loss` - Triplet margin loss

## See Also

- [Loss Functions Documentation](../loss_functions.md)
- [Classification Tutorial](../tutorials/classification.md)
- [Tensor Operations](../tensor_operations.md) 