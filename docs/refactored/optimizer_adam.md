# torch::optimizer_adam

Creates an Adam optimizer with support for both legacy positional and modern named parameter syntax.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::optimizer_adam -parameters $paramList -lr $learningRate ?-beta1 $beta1? ?-beta2 $beta2? ?-weightDecay $weightDecay?
torch::optimizerAdam -parameters $paramList -lr $learningRate ?-beta1 $beta1? ?-beta2 $beta2? ?-weightDecay $weightDecay?
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::optimizer_adam $paramList $learningRate ?$beta1? ?$beta2? ?$weightDecay?
```

## Parameters

### Named Parameters
- **-parameters** | **-params** (required): List of tensor handles representing model parameters
- **-lr** | **-learningRate** (required): Learning rate (positive float)
- **-beta1** (optional): First moment decay rate (default: 0.9, range: [0,1))
- **-beta2** (optional): Second moment decay rate (default: 0.999, range: [0,1))
- **-weightDecay** | **-weight_decay** (optional): Weight decay coefficient (default: 0.0, non-negative)

### Positional Parameters
1. **paramList** (required): List of tensor handles representing model parameters
2. **learningRate** (required): Learning rate (positive float)  
3. **beta1** (optional): First moment decay rate (default: 0.9)
4. **beta2** (optional): Second moment decay rate (default: 0.999)
5. **weightDecay** (optional): Weight decay coefficient (default: 0.0)

## Returns

Returns a string handle that can be used to reference the optimizer in subsequent operations.

## Description

Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSprop. Adam maintains per-parameter learning rates that are adapted based on the average of recent magnitudes of the gradients for the weights (first moment) and the average of the recent magnitudes of the gradients for the weights (second moment).

The Adam algorithm maintains running averages of both the gradients and the second moments of the gradients. The update rule is:

```
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)
```

Where:
- `m_t` and `v_t` are estimates of the first and second moments
- `m̂_t` and `v̂_t` are bias-corrected estimates
- `α` is the learning rate
- `β₁` and `β₂` are exponential decay rates
- `ε` is a small constant for numerical stability

## Examples

### Named Parameter Syntax

#### Basic Usage
```tcl
# Create model parameters
set weight [torch::randn {784 128} float32 cpu]
set bias [torch::randn {128} float32 cpu]
set params [list $weight $bias]

# Create Adam optimizer with default parameters
set optimizer [torch::optimizer_adam -parameters $params -lr 0.001]
```

#### With Custom Beta Values
```tcl
# Create optimizer with custom momentum parameters
set optimizer [torch::optimizer_adam \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.95 \
    -beta2 0.999]
```

#### With Weight Decay
```tcl
# Create optimizer with weight decay (L2 regularization)
set optimizer [torch::optimizer_adam \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.9 \
    -beta2 0.999 \
    -weightDecay 0.01]
```

#### Using Alternative Parameter Names
```tcl
# Using shorter/alternative parameter names
set optimizer [torch::optimizer_adam \
    -params $params \
    -learningRate 0.002 \
    -weight_decay 0.005]
```

#### Out-of-Order Parameters
```tcl
# Parameters can be specified in any order
set optimizer [torch::optimizer_adam \
    -beta1 0.85 \
    -parameters $params \
    -weightDecay 0.01 \
    -lr 0.001 \
    -beta2 0.995]
```

#### CamelCase Alias
```tcl
# Using camelCase command name
set optimizer [torch::optimizerAdam \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.9 \
    -beta2 0.999]
```

### Legacy Positional Syntax

#### Basic Usage
```tcl
# Create optimizer with default beta values and no weight decay
set optimizer [torch::optimizer_adam $params 0.001]
```

#### With Custom Beta1
```tcl
# Create optimizer with custom first moment decay
set optimizer [torch::optimizer_adam $params 0.001 0.95]
```

#### With Both Beta Values
```tcl
# Create optimizer with custom momentum parameters
set optimizer [torch::optimizer_adam $params 0.001 0.95 0.999]
```

#### All Parameters
```tcl
# Create optimizer with all parameters specified
set optimizer [torch::optimizer_adam $params 0.001 0.9 0.999 0.01]
```

## Typical Workflow

```tcl
# 1. Create model parameters
set conv_weight [torch::randn {32 3 3 3} float32 cpu]
set conv_bias [torch::randn {32} float32 cpu]
set fc_weight [torch::randn {10 128} float32 cpu]
set fc_bias [torch::randn {10} float32 cpu]

set all_params [list $conv_weight $conv_bias $fc_weight $fc_bias]

# 2. Create Adam optimizer
set optimizer [torch::optimizer_adam \
    -parameters $all_params \
    -lr 0.001 \
    -beta1 0.9 \
    -beta2 0.999 \
    -weightDecay 0.0001]

# 3. Training loop would use this optimizer
# (optimizer step, zero grad, etc.)
```

## Algorithm Details

### Advantages
- **Adaptive learning rates**: Different learning rates for each parameter
- **Momentum**: Uses both first and second moment estimates
- **Bias correction**: Corrects for initialization bias in moment estimates
- **Robust to hyperparameters**: Generally works well with default settings
- **Efficient**: Computationally efficient and has little memory requirements
- **Scale invariant**: Invariant to diagonal rescaling of the gradients

### Disadvantages
- **Memory overhead**: Requires storage of momentum terms (2x parameter storage)
- **May not converge**: Can fail to converge to optimal solution in some cases
- **Learning rate scheduling**: May still benefit from learning rate scheduling
- **Hyperparameter sensitivity**: Performance can be sensitive to β₂ parameter

### When to Use
- **Default choice**: Often a good first optimizer to try
- **Deep learning**: Particularly effective for training neural networks
- **Sparse gradients**: Works well with sparse gradients (e.g., NLP applications)
- **Non-stationary objectives**: Handles changing objectives better than SGD
- **Large parameter spaces**: Effective for models with many parameters

## Parameter Guidelines

### Learning Rate (-lr)
- **Typical range**: 0.0001 to 0.01
- **Default suggestion**: 0.001
- **Higher values**: For faster convergence but risk instability
- **Lower values**: For more stable but slower training
- **Common values**: 0.001, 0.0001, 0.01

### Beta1 (-beta1)
- **Typical range**: 0.8 to 0.99
- **Default**: 0.9
- **Higher values**: More momentum, smoother updates
- **Lower values**: Less momentum, more responsive to recent gradients
- **Rarely changed**: Usually left at default

### Beta2 (-beta2)
- **Typical range**: 0.99 to 0.9999
- **Default**: 0.999
- **Higher values**: More stable second moment estimates
- **Lower values**: More responsive to recent gradient magnitudes
- **Sparse gradients**: Sometimes increased to 0.9999 for sparse problems

### Weight Decay (-weightDecay)
- **Typical range**: 0.0 to 0.01
- **Default**: 0.0 (no regularization)
- **Common values**: 0.0001, 0.001, 0.01
- **Purpose**: L2 regularization to prevent overfitting
- **Higher values**: Stronger regularization

## Error Handling

The command provides comprehensive error checking:

```tcl
# Invalid learning rate
catch {torch::optimizer_adam -parameters $params -lr -0.001} error
# Error: Required parameters missing or invalid (parameters and positive learning rate required, beta values must be in [0,1), weight_decay must be non-negative)

# Missing required parameters
catch {torch::optimizer_adam -lr 0.001} error
# Error: Required parameters missing or invalid (parameters and positive learning rate required, beta values must be in [0,1), weight_decay must be non-negative)

# Invalid beta values
catch {torch::optimizer_adam -parameters $params -lr 0.001 -beta1 1.5} error
# Error: Required parameters missing or invalid (parameters and positive learning rate required, beta values must be in [0,1), weight_decay must be non-negative)

# Invalid parameter name
catch {torch::optimizer_adam -parameters $params -lr 0.001 -invalid value} error
# Error: Unknown parameter: -invalid

# Invalid tensor reference
catch {torch::optimizer_adam {invalid_tensor} 0.001} error
# Error: Invalid parameter tensor: invalid_tensor
```

## Comparison with Other Optimizers

| Optimizer | Learning Rate Adaptation | Momentum | Memory Overhead | Best For |
|-----------|---------------------------|----------|-----------------|----------|
| **Adam** | ✅ Per-parameter | ✅ First & Second | High | General purpose, deep learning |
| **SGD** | ❌ Fixed | ❌ Optional | Low | Simple problems, fine-tuning |
| **Adagrad** | ✅ Per-parameter | ❌ None | Medium | Sparse data, online learning |
| **RMSprop** | ✅ Per-parameter | ❌ Second only | Medium | RNNs, non-stationary objectives |

## Compatibility

- **Backward Compatible**: All existing code using positional syntax continues to work
- **Forward Compatible**: New code can use modern named parameter syntax
- **Interchangeable**: Both syntaxes produce identical optimizers
- **Consistent**: Same validation and error handling for both syntaxes

## See Also

- [torch::optimizer_adamw](optimizer_adamw.md) - Adam with decoupled weight decay
- [torch::optimizer_sgd](optimizer_sgd.md) - Stochastic Gradient Descent
- [torch::optimizer_rmsprop](optimizer_rmsprop.md) - RMSprop optimizer
- [torch::optimizer_adagrad](optimizer_adagrad.md) - Adagrad optimizer
- [Optimizer Operations](../optimizers.md) - Common optimizer operations

## Migration Guide

### From Legacy to Named Parameters

```tcl
# Old positional syntax
set optimizer [torch::optimizer_adam $params 0.001 0.9 0.999 0.01]

# New named parameter syntax  
set optimizer [torch::optimizer_adam \
    -parameters $params \
    -lr 0.001 \
    -beta1 0.9 \
    -beta2 0.999 \
    -weightDecay 0.01]
```

### Benefits of Named Parameters
1. **Self-documenting**: Parameter names make code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional parameters**: Easier to specify only needed parameters
4. **Error prevention**: Less likely to pass parameters in wrong order
5. **IDE support**: Better auto-completion and parameter hints
6. **Maintainability**: Easier to modify and understand code

## Performance Notes

- Adam typically converges faster than SGD in the early stages of training
- May require learning rate scheduling for optimal convergence
- Memory usage is approximately 2x the model parameters (for momentum terms)
- Computational overhead is minimal compared to the forward/backward pass 