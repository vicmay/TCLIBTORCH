# torch::kl_div_loss

## Overview
Computes the Kullback-Leibler (KL) divergence loss between input and target tensors. KL divergence is commonly used to measure how one probability distribution diverges from a second, reference probability distribution.

## Syntax

### Current Syntax (snake_case + positional)
```tcl
torch::kl_div_loss input target ?reduction? ?log_target?
```

### New Syntax (camelCase + named parameters)  
```tcl
torch::klDivLoss -input input -target target -reduction reduction -logTarget logTarget
```

### Alternative Alias
```tcl
torch::klDivLoss input target ?reduction? ?log_target?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `input` | string | Yes | - | Input tensor name (predicted log-probabilities) |
| `target` | string | Yes | - | Target tensor name (true probabilities) |
| `reduction` | string/int | No | "mean" | Loss reduction method |
| `logTarget` | bool | No | false | Whether target is in log-space |

### Named Parameters
- `-input` - Input tensor containing predicted log-probabilities
- `-target` - Target tensor containing true probabilities
- `-reduction` - Reduction method: "none", "mean", "sum" (or 0, 1, 2)
- `-logTarget` - If true, target is expected to be in log-space

### Reduction Options
- `"none"` (0): No reduction, return full tensor
- `"mean"` (1): Return mean of all elements (default)
- `"sum"` (2): Return sum of all elements

## Return Value
Returns a tensor handle containing the computed KL divergence loss.

## Mathematical Background

KL divergence measures the difference between two probability distributions P and Q:

**Standard KL Divergence:**
```
KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
```

**In PyTorch/LibTorch:**
- Input is expected to contain log-probabilities (log P)
- Target contains probabilities (Q)
- Formula: `target * (log(target) - input)`

When `logTarget=true`, target also contains log-probabilities.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create input tensor (log-probabilities)
set input [torch::tensor_create {-1.0986 -0.6931 -1.6094} float32 cpu false]

# Create target tensor (probabilities)
set target [torch::tensor_create {0.3333 0.3333 0.3333} float32 cpu false]

# Compute KL divergence loss
set loss [torch::kl_div_loss $input $target]
puts "KL Loss: [torch::tensor_item $loss]"
```

### Named Parameter Syntax
```tcl
# Same computation using named parameters
set loss [torch::kl_div_loss -input $input -target $target -reduction mean]

# With additional options
set loss_detailed [torch::kl_div_loss \
    -input $input \
    -target $target \
    -reduction none \
    -logTarget false]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::klDivLoss $input $target]

# With named parameters
set loss [torch::klDivLoss -input $input -target $target -reduction sum]
```

### Different Reduction Methods
```tcl
# No reduction - return per-element loss
set loss_none [torch::kl_div_loss $input $target 0]

# Mean reduction (default)
set loss_mean [torch::kl_div_loss $input $target 1]

# Sum reduction
set loss_sum [torch::kl_div_loss $input $target 2]

# Using string reduction values
set loss_mean_str [torch::kl_div_loss -input $input -target $target -reduction mean]
```

### Log-Target Mode
```tcl
# When both input and target are in log-space
set log_input [torch::tensor_create {-1.0986 -0.6931 -1.6094} float32 cpu false]
set log_target [torch::tensor_create {-1.0986 -0.6931 -1.6094} float32 cpu false]

set loss [torch::kl_div_loss $log_input $log_target 1 1]
# Or with named parameters
set loss [torch::kl_div_loss -input $log_input -target $log_target -logTarget true]
```

### Multi-dimensional Tensors
```tcl
# Create 2D tensors for batch processing
set input_2d [torch::randn {2 3} cpu float32]
set target_2d [torch::rand {2 3} cpu float32]

# Apply softmax to get proper probabilities
set target_prob [torch::softmax $target_2d 1]

# Compute KL divergence loss
set batch_loss [torch::kl_div_loss $input_2d $target_prob]
```

### Classification Example
```tcl
# Example: 3-class classification with batch size 2
set predictions [torch::tensor_create {{-2.3 -1.5 -0.2} {-1.8 -0.9 -1.1}} float32 cpu false]
set true_probs [torch::tensor_create {{0.0 0.0 1.0} {0.0 1.0 0.0}} float32 cpu false]

# Compute loss
set classification_loss [torch::klDivLoss \
    -input $predictions \
    -target $true_probs \
    -reduction mean]

puts "Classification KL Loss: [torch::tensor_item $classification_loss]"
```

## Use Cases

### 1. Classification Tasks
- **Purpose**: Measure divergence between predicted and true class distributions
- **Input**: Log-probabilities from model (after log_softmax)
- **Target**: One-hot encoded ground truth or soft labels

### 2. Generative Models
- **Purpose**: Training generative models like VAEs
- **Input**: Predicted distribution parameters
- **Target**: True data distribution

### 3. Distribution Matching
- **Purpose**: Force model output to match target distribution
- **Input**: Model predictions in log-space
- **Target**: Reference distribution

### 4. Knowledge Distillation
- **Purpose**: Transfer knowledge from teacher to student model
- **Input**: Student model predictions
- **Target**: Teacher model soft predictions

## Error Handling

```tcl
# Handle invalid tensor names
if {[catch {torch::kl_div_loss invalid_tensor $target} error]} {
    puts "Error: $error"
}

# Handle invalid parameters
if {[catch {torch::kl_div_loss -input $input -invalid param} error]} {
    puts "Error: $error"
}

# Handle invalid reduction values
if {[catch {torch::kl_div_loss -input $input -target $target -reduction invalid} error]} {
    puts "Error: $error"
}
```

## Performance Considerations

1. **Tensor Size**: Larger tensors require more computation time
2. **Reduction**: "none" reduction returns larger tensors
3. **Memory**: Consider memory usage for large batch sizes
4. **Numerical Stability**: Ensure input is in proper log-space to avoid numerical issues

## Backward Compatibility

All existing code using the positional syntax will continue to work:

```tcl
# Old code still works
set loss [torch::kl_div_loss $input $target]
set loss [torch::kl_div_loss $input $target 1]
set loss [torch::kl_div_loss $input $target 1 0]
```

## Migration Guide

### From Positional to Named Parameters

**Old:**
```tcl
set loss [torch::kl_div_loss $input $target 1 0]
```

**New:**
```tcl
set loss [torch::kl_div_loss -input $input -target $target -reduction mean -logTarget false]
# Or using camelCase
set loss [torch::klDivLoss -input $input -target $target -reduction mean -logTarget false]
```

### Parameter Mapping
- Position 1: `input` → `-input`
- Position 2: `target` → `-target`  
- Position 3: `reduction` (0/1/2) → `-reduction` ("none"/"mean"/"sum")
- Position 4: `log_target` (0/1) → `-logTarget` (false/true)

## Related Commands

- `torch::cross_entropy_loss` - Cross-entropy loss for classification
- `torch::nll_loss` - Negative log-likelihood loss
- `torch::mse_loss` - Mean squared error loss
- `torch::softmax` - Softmax activation function
- `torch::log_softmax` - Log-softmax activation function

## See Also

- [Loss Functions Documentation](../loss_functions.md)
- [Tensor Operations](../tensor_operations.md)
- [Mathematical Functions](../math_functions.md) 