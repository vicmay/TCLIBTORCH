# torch::optimizer_rprop / torch::optimizerRprop

Creates a Rprop (Resilient backpropagation) optimizer for neural network training. Rprop is a gradient-based optimization algorithm that adapts the step size for each parameter based on the sign of the gradient.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_rprop -parameters value ?-lr value? ?-etaMinus value? ?-etaPlus value? ?-stepSizeMin value? ?-stepSizeMax value?
torch::optimizerRprop -parameters value ?-lr value? ?-etaMinus value? ?-etaPlus value? ?-stepSizeMin value? ?-stepSizeMax value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_rprop parameters ?lr? ?etas? ?step_sizes?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.01 | Initial learning rate |
| `-etaMinus` | double | 0.5 | Decrease factor for step size |
| `-etaPlus` | double | 1.2 | Increase factor for step size |
| `-stepSizeMin` | double | 1e-6 | Minimum step size |
| `-stepSizeMax` | double | 50.0 | Maximum step size |

## Description

Rprop (Resilient backpropagation) is a gradient-based optimization algorithm that adapts the step size for each parameter independently based on the sign of the gradient. It is particularly effective for full-batch training and can handle different scales of gradients well.

Key features:
- Individual step sizes for each parameter
- Sign-based gradient updates
- Robust to gradient scaling
- No momentum terms
- Suitable for full-batch training
- Works well with sparse gradients

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_rprop -parameters $tensor]

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
set opt [torch::optimizer_rprop \
    -parameters $params \
    -lr 0.01 \
    -etaMinus 0.5 \
    -etaPlus 1.2 \
    -stepSizeMin 1e-6 \
    -stepSizeMax 50.0]
```

### Using with Neural Network Module
```tcl
# Create a neural network module
set model [torch::sequential]
torch::add_module $model [torch::linear 784 256]
torch::add_module $model [torch::relu]
torch::add_module $model [torch::linear 256 10]

# Create optimizer for all module parameters
set opt [torch::optimizer_rprop -parameters $model -lr 0.01]
```

### Fine-tuning Step Size Bounds
```tcl
# Create optimizer with custom step size bounds
set opt [torch::optimizer_rprop \
    -parameters $tensor \
    -lr 0.01 \
    -etaMinus 0.4 \
    -etaPlus 1.5 \
    -stepSizeMin 1e-8 \
    -stepSizeMax 100.0]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_sgd` - Stochastic Gradient Descent optimizer 