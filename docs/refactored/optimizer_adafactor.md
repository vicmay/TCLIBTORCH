# torch::optimizer_adafactor / torch::optimizerAdafactor

Creates an Adafactor optimizer for neural network training. Adafactor is a memory-efficient variant of Adam that uses factored second moment estimates to reduce memory usage.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_adafactor -parameters value ?-lr value? ?-eps2 value? ?-clipThreshold value? ?-decayRate value? ?-beta1 value? ?-weightDecay value?
torch::optimizerAdafactor -parameters value ?-lr value? ?-eps2 value? ?-clipThreshold value? ?-decayRate value? ?-beta1 value? ?-weightDecay value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_adafactor parameters ?lr? ?eps2? ?cliping_threshold? ?decay_rate? ?beta1? ?weight_decay?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.8 | Learning rate |
| `-eps2` | double | 1e-30 | Second moment epsilon for numerical stability |
| `-clipThreshold` | double | 1.0 | Gradient clipping threshold |
| `-decayRate` | double | -1.0 | Learning rate decay rate (-1.0 for default schedule) |
| `-beta1` | double | -1.0 | First moment decay rate (-1.0 to disable momentum) |
| `-weightDecay` | double | 0.0 | Weight decay (L2 penalty) |

## Description

Adafactor is a memory-efficient optimization algorithm designed for large-scale machine learning models. It reduces memory usage compared to Adam by:
- Using factored second moment estimates
- Supporting relative step sizes
- Incorporating built-in learning rate scheduling

Key features:
- Memory-efficient second moment estimation
- Automatic learning rate scaling
- Optional momentum with beta1
- Built-in gradient clipping
- Configurable weight decay
- Suitable for training large language models

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_adafactor -parameters $tensor]

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
set opt [torch::optimizer_adafactor \
    -parameters $params \
    -lr 0.8 \
    -eps2 1e-30 \
    -clipThreshold 1.0 \
    -decayRate 0.8 \
    -beta1 0.9 \
    -weightDecay 0.01]
```

### Using with Neural Network Module
```tcl
# Create a neural network module
set model [torch::sequential]
torch::add_module $model [torch::linear 784 256]
torch::add_module $model [torch::relu]
torch::add_module $model [torch::linear 256 10]

# Create optimizer for all module parameters
set opt [torch::optimizer_adafactor -parameters $model -lr 0.8]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_adamw` - AdamW optimizer variant 