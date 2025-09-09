# torch::optimizer_radam / torch::optimizerRAdam

Creates a RAdam (Rectified Adam) optimizer for neural network training. RAdam is a variant of Adam that rectifies the variance of the adaptive learning rate, which can lead to better convergence.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_radam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
torch::optimizerRAdam -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_radam parameters ?lr? ?betas? ?eps? ?weight_decay?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.001 | Learning rate |
| `-beta1` | double | 0.9 | First moment decay rate |
| `-beta2` | double | 0.999 | Second moment decay rate |
| `-eps` | double | 1e-8 | Term added for numerical stability |
| `-weightDecay` | double | 0.0 | Weight decay (L2 penalty) |

## Description

The RAdam optimizer is a variant of Adam that addresses the warm-up period issue in adaptive learning rate methods. It rectifies the variance of the adaptive learning rate to provide better convergence.

Key features:
- Rectifies the variance of the adaptive learning rate
- Eliminates the need for learning rate warm-up
- Maintains good convergence properties of Adam
- Suitable for a wide range of deep learning tasks
- Particularly effective for training transformers and large models

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_radam -parameters $tensor]

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
set opt [torch::optimizer_radam \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.8 \
    -beta2 0.9 \
    -eps 1e-6 \
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
set opt [torch::optimizer_radam -parameters $model -lr 0.001]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_nadam` - NAdam optimizer variant 