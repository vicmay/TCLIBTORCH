# torch::nll_loss

Computes the Negative Log Likelihood (NLL) loss between input and target tensors.

## Syntax

### Positional Parameters (Original)
```tcl
torch::nll_loss input target ?weight? ?reduction?
```

### Named Parameters (New)
```tcl
torch::nll_loss -input tensor -target tensor ?-weight tensor? ?-reduction string?
```

### CamelCase Alias
```tcl
torch::nllLoss -input tensor -target tensor ?-weight tensor? ?-reduction string?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | tensor | Yes | Input tensor containing log probabilities (2D: [N, C]) |
| `target` | tensor | Yes | Target tensor containing class indices (1D: [N]) |
| `weight` | tensor | No | Optional weight tensor for each class (1D: [C]) |
| `reduction` | string | No | Reduction mode: "mean" (default), "sum", or "none" |

## Mathematical Definition

The Negative Log Likelihood loss is defined as:

For input tensor `x` and target tensor `y`:

**Individual Loss:**
```
loss(x, y) = -x[y]
```

**With Class Weights:**
```
loss(x, y) = -weight[y] * x[y]
```

**Reduction Modes:**
- `none`: No reduction, return raw losses for each sample
- `mean`: Average loss across all samples
- `sum`: Sum of all losses

## Key Properties

1. **Input Requirements**: Input must contain log probabilities (usually from log_softmax)
2. **Target Format**: Target contains class indices (not one-hot encoded)
3. **Shapes**: Input is 2D [batch_size, num_classes], target is 1D [batch_size]
4. **Loss Range**: [0, +∞) - higher values indicate worse predictions
5. **Gradient**: Provides sparse gradients only for the target class

## Examples

### Basic Usage (Positional)
```tcl
# Create log probability tensor for 2 samples, 3 classes
set input [torch::tensor_create {-1.0 -2.0 -3.0 -2.0 -1.0 -3.0} {2 3} float32]

# Create target with class indices [0, 1]
set target [torch::tensor_create {0 1} int64]

# Compute NLL loss (mean reduction by default)
set loss [torch::nll_loss $input $target]
puts "NLL Loss: [torch::tensor_item $loss]"
```

### Named Parameters with Options
```tcl
# Compute NLL loss with sum reduction
set loss [torch::nll_loss -input $input -target $target -reduction sum]

# NLL loss with class weights
set weight [torch::tensor_create {1.0 2.0 0.5} float32]
set weighted_loss [torch::nll_loss -input $input -target $target -weight $weight]
```

### No Reduction (Per-Sample Losses)
```tcl
# Get individual losses for each sample
set individual_losses [torch::nll_loss -input $input -target $target -reduction none]
set shape [torch::tensor_shape $individual_losses]  ; # Should be {2}
```

### CamelCase Alias
```tcl
# Using camelCase alias
set loss [torch::nllLoss -input $input -target $target -reduction mean]
```

## Reduction Modes

### Mean (Default)
```tcl
set loss_mean [torch::nll_loss -input $input -target $target -reduction mean]
# Returns scalar: average loss across batch
```

### Sum
```tcl
set loss_sum [torch::nll_loss -input $input -target $target -reduction sum]
# Returns scalar: total loss across batch
```

### None
```tcl
set loss_none [torch::nll_loss -input $input -target $target -reduction none]
# Returns tensor of shape [batch_size] with individual losses
```

## Use Cases

### 1. Multi-Class Classification
```tcl
# For 3-class classification with 4 samples
set logits [torch::tensor_create {-0.5 -1.2 -2.1 -1.0 -0.3 -1.8 -2.0 -1.1 -0.4 -1.5 -0.8 -1.3} {4 3} float32]
set targets [torch::tensor_create {0 1 2 1} int64]
set loss [torch::nll_loss -input $logits -target $targets]
```

### 2. Imbalanced Dataset with Class Weights
```tcl
# Weight rare classes more heavily
set class_weights [torch::tensor_create {0.5 2.0 3.0} float32]  ; # Rare class 2 gets 3x weight
set loss [torch::nll_loss -input $logits -target $targets -weight $class_weights]
```

### 3. Custom Training Loop
```tcl
# Training step
set predictions [torch::log_softmax $model_output 1]  ; # Convert to log probabilities
set loss [torch::nll_loss -input $predictions -target $true_labels]
```

## NLL Loss vs Other Loss Functions

### NLL Loss vs Cross-Entropy Loss
- **NLL Loss**: Expects log probabilities as input
- **Cross-Entropy Loss**: Expects raw logits and applies log_softmax internally
- **Relationship**: CrossEntropyLoss = LogSoftmax + NLLLoss

### NLL Loss vs MSE Loss
```tcl
# NLL Loss (for classification)
set nll [torch::nll_loss -input $log_probs -target $class_indices]

# MSE Loss (for regression)  
set mse [torch::mse_loss -input $predictions -target $continuous_targets]
```

## Performance Notes

1. **Memory Efficiency**: More memory efficient than one-hot encoding for sparse targets
2. **Gradient Computation**: Only computes gradients for the target class
3. **Numerical Stability**: Requires numerically stable log probabilities
4. **Class Weighting**: Minimal overhead when using class weights

## Common Patterns

### With PyTorch-style Training
```tcl
# Forward pass
set raw_output [torch::linear $input $weight $bias]
set log_probs [torch::log_softmax $raw_output 1]

# Loss computation
set loss [torch::nllLoss -input $log_probs -target $targets]

# Backward pass (loss backpropagation)
torch::backward $loss
```

### Multi-Target Classification
```tcl
# For batch processing
foreach {batch_input batch_target} $data_loader {
    set predictions [model_forward $batch_input]
    set loss [torch::nllLoss -input $predictions -target $batch_target]
    # Process loss...
}
```

## Error Handling

The function validates inputs and provides clear error messages:

```tcl
# Missing required parameter
catch {torch::nll_loss -input $tensor} error
# Error: Required parameters -input and -target must be provided

# Invalid parameter name
catch {torch::nll_loss -input $input -target $target -invalid param} error
# Error: Unknown parameter: -invalid

# Dimension mismatch
catch {torch::nll_loss $wrong_shape_input $target} error
# Error: size mismatch (got input: [shape], target: [shape])
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set loss [torch::nll_loss $input $target $weight "sum"]
```

**New (Named):**
```tcl
set loss [torch::nll_loss -input $input -target $target -weight $weight -reduction sum]
```

### Advantages of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Code is self-documenting
4. **Optional Parameters**: Easy to omit optional parameters

### Backward Compatibility

All existing positional syntax continues to work:
```tcl
# These all continue to work
set loss1 [torch::nll_loss $input $target]
set loss2 [torch::nll_loss $input $target $weight]
set loss3 [torch::nll_loss $input $target none "sum"]
```

## Mathematical Properties

1. **Non-Negativity**: NLL loss is always ≥ 0
2. **Perfect Prediction**: Loss = 0 when target class has probability 1 (log_prob = 0)
3. **Poor Prediction**: Loss → ∞ as target class probability → 0 (log_prob → -∞)
4. **Monotonicity**: Lower log probability for target class → higher loss
5. **Batch Independence**: Each sample contributes independently to total loss

## See Also

- `torch::cross_entropy_loss` - Combined softmax and NLL loss
- `torch::log_softmax` - Convert logits to log probabilities
- `torch::softmax` - Convert logits to probabilities
- `torch::mse_loss` - Mean squared error loss for regression
- `torch::l1_loss` - L1 loss for regression 