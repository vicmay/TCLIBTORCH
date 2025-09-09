# torch::optimizer_adagrad

Creates an Adagrad optimizer with support for both legacy positional and modern named parameter syntax.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::optimizer_adagrad -parameters $paramList -lr $learningRate ?-eps $epsilon?
torch::optimizerAdagrad -parameters $paramList -lr $learningRate ?-eps $epsilon?
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::optimizer_adagrad $paramList $learningRate ?$epsilon?
```

## Parameters

### Named Parameters
- **-parameters** | **-params** (required): List of tensor handles representing model parameters
- **-lr** | **-learningRate** (required): Learning rate (positive float)
- **-eps** | **-epsilon** (optional): Small constant for numerical stability (default: 1e-10)

### Positional Parameters
1. **paramList** (required): List of tensor handles representing model parameters
2. **learningRate** (required): Learning rate (positive float)  
3. **epsilon** (optional): Small constant for numerical stability (default: 1e-10)

## Returns

Returns a string handle that can be used to reference the optimizer in subsequent operations.

## Description

Adagrad is an adaptive learning rate optimization algorithm that adapts the learning rate to the parameters, performing larger updates for infrequent parameters and smaller updates for frequent parameters. It's particularly well-suited for dealing with sparse data.

The Adagrad algorithm maintains a running sum of squared gradients and uses this to normalize the learning rate. The update rule is:

```
accumulated_grad = accumulated_grad + grad^2
param = param - (lr / sqrt(accumulated_grad + eps)) * grad
```

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create model parameters
set weight [torch::randn {784 128} float32 cpu]
set bias [torch::randn {128} float32 cpu]
set params [list $weight $bias]

# Create Adagrad optimizer with default eps
set optimizer [torch::optimizer_adagrad -parameters $params -lr 0.01]
```

#### With Custom Epsilon
```tcl
# Create optimizer with custom epsilon
set optimizer [torch::optimizer_adagrad \
    -parameters $params \
    -lr 0.001 \
    -eps 1e-8]
```

#### Using Alternative Parameter Names
```tcl
# Using shorter parameter names
set optimizer [torch::optimizer_adagrad \
    -params $params \
    -learningRate 0.005 \
    -epsilon 1e-9]
```

#### CamelCase Alias
```tcl
# Using camelCase command name
set optimizer [torch::optimizerAdagrad \
    -parameters $params \
    -lr 0.01 \
    -eps 1e-8]
```

### Legacy Positional Syntax

#### Basic Usage
```tcl
# Create optimizer with default epsilon
set optimizer [torch::optimizer_adagrad $params 0.01]
```

#### With Custom Epsilon
```tcl
# Create optimizer with custom epsilon
set optimizer [torch::optimizer_adagrad $params 0.01 1e-8]
```

## Typical Workflow

```tcl
# 1. Create model parameters
set conv_weight [torch::randn {32 3 3 3} float32 cpu]
set conv_bias [torch::randn {32} float32 cpu]
set fc_weight [torch::randn {10 128} float32 cpu]
set fc_bias [torch::randn {10} float32 cpu]

set all_params [list $conv_weight $conv_bias $fc_weight $fc_bias]

# 2. Create Adagrad optimizer
set optimizer [torch::optimizer_adagrad \
    -parameters $all_params \
    -lr 0.01 \
    -eps 1e-10]

# 3. Training loop would use this optimizer
# (optimizer step, zero grad, etc.)
```

## Algorithm Details

### Advantages
- **Adaptive learning rates**: Automatically adjusts learning rate per parameter
- **Good for sparse data**: Works well with sparse gradients and embeddings
- **No manual learning rate scheduling**: Built-in learning rate decay
- **Simple implementation**: Few hyperparameters to tune

### Disadvantages
- **Aggressive learning rate decay**: Learning rate can become too small too quickly
- **May stop learning**: In long training runs, accumulated gradients can become very large
- **Not suitable for all problems**: May perform poorly on some non-convex optimization problems

### When to Use
- Training with sparse features (e.g., NLP with large vocabularies)
- When you want adaptive learning rates without manual tuning
- As a baseline optimizer for comparison
- When computational efficiency is important

## Parameter Guidelines

### Learning Rate (-lr)
- **Typical range**: 0.001 to 0.1
- **Default suggestion**: 0.01
- **Higher values**: For faster convergence but risk instability
- **Lower values**: For more stable but slower training

### Epsilon (-eps)
- **Typical range**: 1e-12 to 1e-6
- **Default**: 1e-10
- **Purpose**: Prevents division by zero and provides numerical stability
- **Smaller values**: More aggressive adaptation
- **Larger values**: More conservative adaptation

## Error Handling

The command provides comprehensive error checking:

```tcl
# Invalid learning rate
catch {torch::optimizer_adagrad -parameters $params -lr -0.01} error
# Error: Required parameters missing or invalid (parameters and positive learning rate required)

# Missing required parameters
catch {torch::optimizer_adagrad -lr 0.01} error
# Error: Required parameters missing or invalid (parameters and positive learning rate required)

# Invalid parameter name
catch {torch::optimizer_adagrad -parameters $params -lr 0.01 -invalid value} error
# Error: Unknown parameter: -invalid

# Invalid tensor reference
catch {torch::optimizer_adagrad {invalid_tensor} 0.01} error
# Error: Invalid parameter tensor: invalid_tensor
```

## Compatibility

- **Backward Compatible**: All existing code using positional syntax continues to work
- **Forward Compatible**: New code can use modern named parameter syntax
- **Interchangeable**: Both syntaxes produce identical optimizers
- **Consistent**: Same validation and error handling for both syntaxes

## See Also

- [torch::optimizer_adam](optimizer_adam.md) - Adam optimizer with momentum
- [torch::optimizer_sgd](optimizer_sgd.md) - Stochastic Gradient Descent
- [torch::optimizer_rmsprop](optimizer_rmsprop.md) - RMSprop optimizer
- [Optimizer Operations](../optimizers.md) - Common optimizer operations

## Migration Guide

### From Legacy to Named Parameters

```tcl
# Old positional syntax
set optimizer [torch::optimizer_adagrad $params 0.01 1e-8]

# New named parameter syntax  
set optimizer [torch::optimizer_adagrad \
    -parameters $params \
    -lr 0.01 \
    -eps 1e-8]
```

### Benefits of Named Parameters
1. **Self-documenting**: Parameter names make code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Error prevention**: Less likely to pass parameters in wrong order
5. **IDE support**: Better auto-completion and parameter hints 