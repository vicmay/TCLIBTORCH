# torch::optimizer_lbfgs / torch::optimizerLbfgs

Creates an L-BFGS (Limited-memory BFGS) optimizer for neural network training. L-BFGS is a quasi-Newton method that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using a limited amount of memory.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::optimizer_lbfgs -parameters value ?-lr value? ?-maxIter value? ?-maxEval value? ?-toleranceGrad value? ?-toleranceChange value?
torch::optimizerLbfgs -parameters value ?-lr value? ?-maxIter value? ?-maxEval value? ?-toleranceGrad value? ?-toleranceChange value?
```

### Positional Parameters (Legacy)
```tcl
torch::optimizer_lbfgs parameters ?lr? ?max_iter? ?max_eval? ?tolerance_grad? ?tolerance_change?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-parameters` | list/handle | Required | List of tensor handles or a module handle |
| `-lr` | double | 1.0 | Learning rate (L-BFGS specific default) |
| `-maxIter` | int | 20 | Maximum number of iterations per optimization step |
| `-maxEval` | int | 25 | Maximum number of function evaluations per optimization step |
| `-toleranceGrad` | double | 1e-7 | Gradient convergence tolerance |
| `-toleranceChange` | double | 1e-9 | Parameter value convergence tolerance |

## Description

L-BFGS (Limited-memory BFGS) is a quasi-Newton method that approximates the Broyden–Fletcher–Goldfarb–Shanno (BFGS) algorithm using a limited amount of memory. It is particularly useful for optimizing high-dimensional functions.

Key features:
- Second-order optimization method
- Memory-efficient approximation of BFGS
- Line search for step size selection
- Adaptive learning rate based on curvature
- Suitable for both convex and non-convex optimization
- Particularly effective for smooth optimization problems

## Return Value

Returns an optimizer handle that can be used with other optimizer commands like `torch::optimizer_step` and `torch::optimizer_zero_grad`.

## Examples

### Basic Usage with Single Tensor
```tcl
# Create a tensor
set tensor [torch::zeros {5 5} float32]

# Create optimizer with default parameters
set opt [torch::optimizer_lbfgs -parameters $tensor]

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
set opt [torch::optimizer_lbfgs \
    -parameters $params \
    -lr 1.0 \
    -maxIter 20 \
    -maxEval 25 \
    -toleranceGrad 1e-7 \
    -toleranceChange 1e-9]
```

### Using with Neural Network Module
```tcl
# Create a neural network module
set model [torch::sequential]
torch::add_module $model [torch::linear 784 256]
torch::add_module $model [torch::relu]
torch::add_module $model [torch::linear 256 10]

# Create optimizer for all module parameters
set opt [torch::optimizer_lbfgs -parameters $model -lr 1.0]
```

### Fine-tuning Convergence
```tcl
# Create optimizer with tighter convergence criteria
set opt [torch::optimizer_lbfgs \
    -parameters $tensor \
    -lr 1.0 \
    -maxIter 50 \
    -maxEval 60 \
    -toleranceGrad 1e-10 \
    -toleranceChange 1e-12]
```

## See Also

- `torch::optimizer_step` - Performs a single optimization step
- `torch::optimizer_zero_grad` - Zeros out parameter gradients
- `torch::optimizer_adam` - Standard Adam optimizer
- `torch::optimizer_sgd` - Stochastic Gradient Descent optimizer 