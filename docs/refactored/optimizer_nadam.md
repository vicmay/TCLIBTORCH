# torch::optimizer_nadam / torch::optimizerNadam

Creates a NAdam (Nesterov-accelerated Adaptive Moment Estimation) optimizer for neural network training. NAdam combines Adam with Nesterov momentum for potentially better convergence.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_nadam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value? ?-momentumDecay value?
torch::optimizerNadam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value? ?-momentumDecay value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_nadam parameters ?lr? ?betas? ?eps? ?weight_decay? ?momentum_decay?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.002 | Learning rate (NAdam typically uses higher default than Adam) |
| `-beta1` | double | 0.9 | First moment decay rate |
| `-beta2` | double | 0.999 | Second moment decay rate |
| `-eps` | double | 1e-8 | Term added for numerical stability |
| `-weightDecay` | double | 0.0 | Weight decay (L2 penalty) |
| `-momentumDecay` | double | 0.004 | Momentum decay rate for Nesterov momentum |

## Description

The NAdam optimizer is a variant of Adam that incorporates Nesterov momentum. It modifies the momentum term to look ahead in the direction of the accumulated gradient, which can lead to better convergence in some cases.

Key features:
- Combines Adam with Nesterov momentum
- Adaptive learning rates for each parameter
- Higher default learning rate than standard Adam
- Built-in momentum decay for Nesterov acceleration
- Suitable for non-convex optimization problems

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_nadam -parameters $tensor]

# Use in training loop
torch::optimizer_zero_grad $opt
# ... forward pass and loss calculation ...
torch::optimizer_step $opt
```

### Multiple Parameters with Custom Settings
```tcl
# Create multiple tensors
set t1 [torch::zeros {5 5} float32]
set t2 [torch::zeros {3 3} float32]
set params [list $t1 $t2]

# Create optimizer with custom settings
set opt [torch::optimizer_nadam \
    -parameters $params \
    -lr 0.01 \
    -beta1 0.8 \
    -beta2 0.9 \
    -eps 1e-6 \
    -weightDecay 0.01 \
    -momentumDecay 0.002]
```

### Using with Neural Network Module
```tcl
# Create a neural network module
set model [torch::sequential]
torch::add_module $model [torch::linear 784 256]
torch::add_module $model [torch::relu]
torch::add_module $model [torch::linear 256 10]

# Create optimizer for all module parameters
set opt [torch::optimizer_nadam -parameters $model -lr 0.002]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_adamw` - AdamW optimizer variant 