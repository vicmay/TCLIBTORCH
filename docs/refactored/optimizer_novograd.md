# torch::optimizer_novograd / torch::optimizerNovograd

Creates a NovoGrad optimizer for neural network training. NovoGrad is a layer-wise adaptive optimizer that combines elements of Adam and Layer-wise Adaptive Rate Scaling (LARS).

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_novograd -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
torch::optimizerNovograd -parameters value ?-lr value? ?-beta1 value? ?-beta2 value? ?-eps value? ?-weightDecay value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_novograd parameters ?lr? ?betas? ?eps? ?weight_decay?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 0.001 | Learning rate |
| `-beta1` | double | 0.95 | Coefficient for computing running averages of gradient |
| `-beta2` | double | 0.98 | Coefficient for computing running averages of squared gradient |
| `-eps` | double | 1e-8 | Term added to denominator for numerical stability |
| `-weightDecay` | double | 0.0 | Weight decay (L2 penalty) |

## Description

NovoGrad is a layer-wise adaptive optimizer that combines the best features of Adam and LARS. It computes second-order moments at the layer level, which can lead to better convergence in some cases, particularly for large batch training.

Key features:
- Layer-wise gradient normalization
- Adaptive learning rates per layer
- Momentum using running averages
- Efficient memory usage
- Good performance with large batch sizes
- Suitable for training deep neural networks

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_novograd -parameters $tensor]

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
set opt [torch::optimizer_novograd \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.95 \
    -beta2 0.98 \
    -eps 1e-8 \
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
set opt [torch::optimizer_novograd -parameters $model -lr 0.001]
```

### Fine-tuning Learning Rate and Weight Decay
```tcl
# Create optimizer with custom learning rate and weight decay
set opt [torch::optimizer_novograd \
    -parameters $tensor \
    -lr 0.0005 \
    -beta1 0.92 \
    -beta2 0.99 \
    -eps 1e-10 \
    -weightDecay 0.005]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_sgd` - Stochastic Gradient Descent optimizer 