# torch::optimizer_sgd

Creates a Stochastic Gradient Descent (SGD) optimizer for neural network training, optionally with momentum.

## Syntax

### Positional Syntax (Original)
```tcl
torch::optimizer_sgd parameters lr ?momentum? ?dampening? ?weight_decay? ?nesterov?
```

### Named Parameter Syntax (New)
```tcl
torch::optimizer_sgd -parameters params -lr value ?-momentum value? ?-dampening value? ?-weightDecay value? ?-nesterov bool?
```

### CamelCase Alias
```tcl
torch::optimizerSgd -parameters params -lr value ?-momentum value? ?-dampening value? ?-weightDecay value? ?-nesterov bool?
```

## Parameters

### Required Parameters
- **parameters** (`list`): List of tensor handles representing the parameters to optimize

### Optional Parameters
- **lr** (`float`): Learning rate (default: 0.01)
- **momentum** (`float`): Momentum factor (default: 0.0, range: [0.0, ∞))
- **dampening** (`float`): Dampening for momentum (default: 0.0, range: [0.0, ∞))
- **weightDecay** (`float`): Weight decay (L2 penalty) coefficient (default: 0.0, range: [0.0, ∞))
- **nesterov** (`boolean`): Enables Nesterov momentum (default: false)

### Parameter Aliases
- `-parameters` can also be written as `-params`
- `-lr` can also be written as `-learningRate`
- `-weightDecay` can also be written as `-weight_decay`

## Returns
Returns an optimizer handle (string) that can be used with other optimizer functions.

## Algorithm Details

### Update Rule
SGD uses the following update rules:

**Without momentum (momentum = 0.0):**
```
param = param - lr * gradient
```

**With momentum (momentum > 0.0):**
```
v = momentum * v + (1 - dampening) * gradient
if nesterov:
    param = param - lr * (gradient + momentum * v)
else:
    param = param - lr * v
```

### Constraints
- When `nesterov=true`, `momentum` must be > 0.0 and `dampening` must be = 0.0
- All numeric parameters must be non-negative
- Learning rate must be > 0.0

## Examples

### Basic Usage
```tcl
# Create some parameters (tensors with gradients)
set param1 [torch::tensor_create -values {1.0 2.0 3.0} -requires_grad true]
set param2 [torch::tensor_create -values {4.0 5.0 6.0} -requires_grad true]
set params [list $param1 $param2]

# Create basic SGD optimizer
set optimizer [torch::optimizer_sgd $params 0.01]
```

### With Momentum (Positional Syntax)
```tcl
# SGD with momentum
set optimizer [torch::optimizer_sgd $params 0.01 0.9]

# SGD with momentum and weight decay
set optimizer [torch::optimizer_sgd $params 0.01 0.9 0.0 1e-4]
```

### Named Parameter Syntax
```tcl
# Basic named parameter usage
set optimizer [torch::optimizer_sgd -parameters $params -lr 0.01]

# With momentum
set optimizer [torch::optimizer_sgd -parameters $params -lr 0.01 -momentum 0.9]

# Full configuration
set optimizer [torch::optimizer_sgd \
    -parameters $params \
    -lr 0.01 \
    -momentum 0.9 \
    -dampening 0.0 \
    -weightDecay 1e-4 \
    -nesterov false]
```

### Nesterov Momentum
```tcl
# Nesterov momentum (requires momentum > 0 and dampening = 0)
set optimizer [torch::optimizer_sgd \
    -parameters $params \
    -lr 0.01 \
    -momentum 0.9 \
    -dampening 0.0 \
    -nesterov true]
```

### CamelCase Alias
```tcl
# Using the camelCase alias
set optimizer [torch::optimizerSgd -parameters $params -lr 0.01 -momentum 0.9]
```

### Training Loop Example
```tcl
# Create model parameters
set model_params [get_model_parameters $my_model]

# Create optimizer
set optimizer [torch::optimizer_sgd $model_params 0.01 0.9 0.0 1e-4]

# Training loop
for {set epoch 0} {$epoch < 100} {incr epoch} {
    foreach {input target} $training_data {
        # Forward pass
        set output [model_forward $my_model $input]
        set loss [torch::mse_loss $output $target]
        
        # Backward pass
        torch::backward $loss
        
        # Update parameters
        torch::optimizer_step $optimizer
        
        # Clear gradients
        torch::optimizer_zero_grad $optimizer
    }
}
```

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax:
set optimizer [torch::optimizer_sgd $params 0.01 0.9 0.0 1e-4 false]

# New named parameter syntax:
set optimizer [torch::optimizer_sgd \
    -parameters $params \
    -lr 0.01 \
    -momentum 0.9 \
    -dampening 0.0 \
    -weightDecay 1e-4 \
    -nesterov false]
```

## Error Handling

The command validates all parameters and will raise an error if:
- Required parameters are missing
- Learning rate is not positive
- Momentum, dampening, or weight decay are negative
- Nesterov is enabled without proper momentum/dampening configuration
- Parameter tensor handles are invalid

## Performance Notes

- SGD is generally faster than adaptive optimizers like Adam
- Momentum can significantly improve convergence for many problems
- Nesterov momentum often provides better convergence than standard momentum
- Weight decay helps prevent overfitting

## See Also

- [torch::optimizer_adam](optimizer_adam.md) - Adaptive moment estimation optimizer
- [torch::optimizer_step](optimizer_step.md) - Execute optimizer step
- [torch::optimizer_zero_grad](optimizer_zero_grad.md) - Clear gradients
- [torch::optimizerRprop](optimizer_rprop.md) - Resilient backpropagation optimizer 