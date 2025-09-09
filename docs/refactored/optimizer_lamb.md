# torch::optimizer_lamb / torch::optimizerLamb

Creates a LAMB (Layer-wise Adaptive Moments for Batch training) optimizer for neural network training. LAMB is a layerwise adaptive large batch optimization technique that helps scale deep learning training to larger batch sizes.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_lamb -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
torch::optimizerLamb -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_lamb parameters ?lr? ?betas? ?eps? ?weight_decay?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.001 | Learning rate |
| `-beta1` | double | 0.9 | First moment decay rate |
| `-beta2` | double | 0.999 | Second moment decay rate |
| `-eps` | double | 1e-6 | Epsilon for numerical stability (LAMB specific default) |
| `-weightDecay` | double | 0.01 | Weight decay (L2 penalty) |

## Description

LAMB (Layer-wise Adaptive Moments for Batch training) is an optimization algorithm designed to help train deep neural networks with large batch sizes. It extends Adam by incorporating the following key features:

- Layer-wise adaptive learning rates
- Trust ratio clipping
- Global gradient norm correction
- Effective weight decay

Key features:
- Supports training with large batch sizes
- Layer-wise learning rate adaptation
- Built-in trust ratio computation
- Automatic gradient norm correction
- Configurable weight decay
- Suitable for training large language models and transformers

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_lamb -parameters $tensor]

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
set opt [torch::optimizer_lamb \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.9 \
    -beta2 0.999 \
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
set opt [torch::optimizer_lamb -parameters $model -lr 0.001]
```

### Large Batch Training
```tcl
# Create a model with large batch size
set model [torch::sequential]
torch::add_module $model [torch::linear 1024 512]
torch::add_module $model [torch::relu]
torch::add_module $model [torch::linear 512 256]

# LAMB works well with large batch sizes and higher learning rates
set opt [torch::optimizer_lamb \
    -parameters $model \
    -lr 0.01 \
    -beta1 0.9 \
    -beta2 0.999 \
    -eps 1e-6 \
    -weightDecay 0.01]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_adamw` - AdamW optimizer variant
