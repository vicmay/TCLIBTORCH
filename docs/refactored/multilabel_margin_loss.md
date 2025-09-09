# torch::multilabel_margin_loss

Computes multilabel margin loss (also known as multilabel hinge loss) for multilabel classification tasks.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::multilabel_margin_loss -input TENSOR -target TENSOR [-reduction REDUCTION]
torch::multilabelMarginLoss -input TENSOR -target TENSOR [-reduction REDUCTION]
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::multilabel_margin_loss input target [reduction]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | Tensor | Required | Input tensor (predictions) of shape (N, C) where N is batch size and C is number of classes |
| target | Tensor | Required | Target tensor of shape (N, C) with binary labels (0 or 1) - must be integer type |
| reduction | String/Integer | "mean" | Reduction method: "none", "mean", "sum" (or legacy integers: 0=none, 1=mean, 2=sum) |

## Description

The multilabel margin loss creates a criterion that optimizes a multi-class multi-classification hinge loss (margin-based loss) between input x and output y. This loss is particularly useful for multilabel classification tasks where each sample can belong to multiple classes simultaneously.

The loss function is defined as:
```
loss(x, y) = sum_ij(max(0, 1 - (x[y[j]] - x[i]))) / x.size(0)
```

where i == 0 to x.size(0), j == 0 to y.size(0), y[j] >= 0, and i != y[j] for all i and j.

## Examples

### Basic Usage
```tcl
# Create input tensor (predictions)
set input [torch::tensor_randn -shape {4 5} -dtype float32]

# Create target tensor (binary labels) - must be integer type
set target [torch::zeros -shape {4 5} -dtype int64]

# Named parameter syntax
set loss [torch::multilabel_margin_loss -input $input -target $target]

# Legacy positional syntax  
set loss [torch::multilabel_margin_loss $input $target]

# camelCase alias
set loss [torch::multilabelMarginLoss -input $input -target $target]
```

### Different Reduction Types
```tcl
# No reduction - return loss for each sample
set loss [torch::multilabel_margin_loss -input $input -target $target -reduction none]

# Mean reduction (default)
set loss [torch::multilabel_margin_loss -input $input -target $target -reduction mean]

# Sum reduction
set loss [torch::multilabel_margin_loss -input $input -target $target -reduction sum]

# Legacy integer reduction values
set loss [torch::multilabel_margin_loss $input $target 0]  # none
set loss [torch::multilabel_margin_loss $input $target 1]  # mean  
set loss [torch::multilabel_margin_loss $input $target 2]  # sum
```

### Multilabel Classification Training
```tcl
# Training example with multilabel data
set batch_size 8
set num_classes 10

# Create predictions (logits)
set predictions [torch::tensor_randn -shape [list $batch_size $num_classes] -dtype float32]

# Create multilabel targets (binary matrix)
set targets [torch::tensor_randint -low 0 -high 2 -shape [list $batch_size $num_classes] -dtype int64]

# Compute multilabel margin loss
set loss [torch::multilabel_margin_loss -input $predictions -target $targets -reduction mean]

puts "Multilabel margin loss: $loss"
```

## Return Value

Returns a tensor containing the multilabel margin loss:
- If reduction is "none": tensor of shape (N,) with loss for each sample
- If reduction is "mean": scalar tensor with mean loss across all samples
- If reduction is "sum": scalar tensor with sum of all losses

## Notes

- The target tensor must contain binary values (0 or 1) and be of integer type (int64)
- Input tensor typically contains raw logits (unbounded real values)
- This loss is commonly used in multilabel classification where samples can belong to multiple classes
- The loss encourages correct labels to have higher scores than incorrect labels by at least a margin of 1

## Error Handling

The function validates:
- Both input and target tensors must exist
- Target tensor must be integer type
- Reduction parameter must be valid ("none", "mean", "sum", or legacy integers 0, 1, 2)
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::multilabelMarginLoss` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::multilabel_margin_loss $input $target → torch::multilabel_margin_loss -input $input -target $target
torch::multilabel_margin_loss $input $target 0 → torch::multilabel_margin_loss -input $input -target $target -reduction none
torch::multilabel_margin_loss $input $target 1 → torch::multilabel_margin_loss -input $input -target $target -reduction mean

# Modern camelCase
torch::multilabel_margin_loss $input $target → torch::multilabelMarginLoss -input $input -target $target
``` 