# torch::multilabel_soft_margin_loss

Computes multilabel soft margin loss (also known as multilabel logistic loss) for multilabel classification tasks.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::multilabel_soft_margin_loss -input TENSOR -target TENSOR [-reduction REDUCTION]
torch::multilabelSoftMarginLoss -input TENSOR -target TENSOR [-reduction REDUCTION]
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::multilabel_soft_margin_loss input target [reduction]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | Tensor | Required | Input tensor (predictions) of shape (N, C) where N is batch size and C is number of classes |
| target | Tensor | Required | Target tensor of shape (N, C) with binary labels (0 or 1) - can be float type |
| reduction | String/Integer | "mean" | Reduction method: "none", "mean", "sum" (or legacy integers: 0=none, 1=mean, 2=sum) |

## Description

The multilabel soft margin loss creates a criterion that optimizes a multi-class multi-classification soft margin (logistic) loss between input x and output y. This loss is particularly useful for multilabel classification tasks where each sample can belong to multiple classes simultaneously.

The loss function is defined as:
```
loss(x, y) = sum_i(-y[i] * log(sigmoid(x[i])) - (1 - y[i]) * log(1 - sigmoid(x[i])))
```

This is essentially a sigmoid-based binary cross-entropy loss applied to each class independently, making it suitable for multilabel scenarios where classes are not mutually exclusive.

## Examples

### Basic Usage
```tcl
# Create input tensor (predictions/logits)
set input [torch::tensor_randn -shape {4 5} -dtype float32]

# Create target tensor (binary labels) - can be float type
set target [torch::zeros -shape {4 5} -dtype float32]

# Named parameter syntax
set loss [torch::multilabel_soft_margin_loss -input $input -target $target]

# Legacy positional syntax  
set loss [torch::multilabel_soft_margin_loss $input $target]

# camelCase alias
set loss [torch::multilabelSoftMarginLoss -input $input -target $target]
```

### Different Reduction Types
```tcl
# No reduction - return loss for each sample
set loss [torch::multilabel_soft_margin_loss -input $input -target $target -reduction none]

# Mean reduction (default)
set loss [torch::multilabel_soft_margin_loss -input $input -target $target -reduction mean]

# Sum reduction
set loss [torch::multilabel_soft_margin_loss -input $input -target $target -reduction sum]

# Legacy integer reduction values
set loss [torch::multilabel_soft_margin_loss $input $target 0]  # none
set loss [torch::multilabel_soft_margin_loss $input $target 1]  # mean  
set loss [torch::multilabel_soft_margin_loss $input $target 2]  # sum
```

### Multilabel Classification Training
```tcl
# Training example with multilabel data
set batch_size 8
set num_classes 10

# Create predictions (logits)
set predictions [torch::tensor_randn -shape [list $batch_size $num_classes] -dtype float32]

# Create multilabel targets (binary matrix)
set targets [torch::tensor_rand -shape [list $batch_size $num_classes] -dtype float32]
# Convert to binary (0 or 1)
set targets [torch::tensor_round $targets]

# Compute multilabel soft margin loss
set loss [torch::multilabel_soft_margin_loss -input $predictions -target $targets -reduction mean]

puts "Multilabel soft margin loss: $loss"
```

### Comparison with Multilabel Margin Loss
```tcl
# Both losses can be used for multilabel classification
set input [torch::tensor_randn -shape {4 5} -dtype float32]

# Soft margin loss (sigmoid-based) - target can be float
set target_float [torch::zeros -shape {4 5} -dtype float32]
set soft_loss [torch::multilabel_soft_margin_loss -input $input -target $target_float]

# Hard margin loss (hinge-based) - target must be int
set target_int [torch::zeros -shape {4 5} -dtype int64]
set hard_loss [torch::multilabel_margin_loss -input $input -target $target_int]

puts "Soft margin loss: $soft_loss"
puts "Hard margin loss: $hard_loss"
```

## Return Value

Returns a tensor containing the multilabel soft margin loss:
- If reduction is "none": tensor of shape (N,) with loss for each sample
- If reduction is "mean": scalar tensor with mean loss across all samples
- If reduction is "sum": scalar tensor with sum of all losses

## Notes

- The target tensor should contain binary values (0 or 1) and can be float type (unlike hard margin loss)
- Input tensor typically contains raw logits (unbounded real values)
- This loss applies sigmoid internally, so don't apply sigmoid to inputs beforehand
- The loss is smooth and differentiable everywhere (unlike hard margin loss)
- Particularly effective for multilabel classification with probabilistic outputs

## Mathematical Details

The multilabel soft margin loss is essentially:
1. Apply sigmoid to each element of the input tensor
2. Compute binary cross-entropy loss for each class independently
3. Sum/average the losses according to the reduction parameter

This makes it equivalent to applying `BCEWithLogitsLoss` to each class channel independently.

## Error Handling

The function validates:
- Both input and target tensors must exist
- Target tensor can be float or integer type (more flexible than hard margin loss)
- Reduction parameter must be valid ("none", "mean", "sum", or legacy integers 0, 1, 2)
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::multilabelSoftMarginLoss` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::multilabel_soft_margin_loss $input $target → torch::multilabel_soft_margin_loss -input $input -target $target
torch::multilabel_soft_margin_loss $input $target 0 → torch::multilabel_soft_margin_loss -input $input -target $target -reduction none
torch::multilabel_soft_margin_loss $input $target 1 → torch::multilabel_soft_margin_loss -input $input -target $target -reduction mean

# Modern camelCase
torch::multilabel_soft_margin_loss $input $target → torch::multilabelSoftMarginLoss -input $input -target $target
```

## See Also

- `torch::multilabel_margin_loss` - Hard margin version (hinge loss)
- `torch::bce_with_logits_loss` - Binary cross-entropy with logits
- `torch::cross_entropy_loss` - Multiclass cross-entropy loss 