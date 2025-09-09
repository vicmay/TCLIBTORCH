# torch::optimizer_momentum_sgd

Creates a Stochastic Gradient Descent (SGD) optimizer with momentum support. The command now supports both the original positional syntax and a modern named-parameter syntax while maintaining full backward compatibility.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
# snake_case command
torch::optimizer_momentum_sgd \
    -parameters $paramList \
    -lr 0.01 \
    -momentum 0.9 ?-weightDecay 0.0005?

# camelCase alias
torch::optimizerMomentumSgd \
    -parameters $paramList \
    -lr 0.01 \
    -momentum 0.9 ?-weightDecay 0.0005?
```

### Legacy Positional Syntax (Backward Compatibility)
```tcl
torch::optimizer_momentum_sgd $paramList $learningRate $momentum ?$weightDecay?
```

## Parameters

| Named Option | Positional | Description | Default |
|--------------|------------|-------------|---------|
| `-parameters` / `-params` | 1 | List of tensor handles representing model parameters | — |
| `-lr` / `-learningRate` | 2 | Learning rate (positive float) | — |
| `-momentum` | 3 | Momentum factor (non-negative float) | — |
| `-weightDecay` / `-weight_decay` | 4 | Weight-decay (L2 penalty) coefficient | `0.0` |

## Returns

A string handle that can be passed to other optimizer operations.

## Examples

### Named Parameter Syntax
```tcl
set w [torch::randn {10 10} float32 cpu]
set b [torch::zeros {10} float32 cpu]
set params [list $w $b]

# Basic optimizer (recommended syntax)
set opt [torch::optimizer_momentum_sgd \
            -parameters $params \
            -lr 0.05 \
            -momentum 0.9]
```

### Legacy Positional Syntax
```tcl
set opt [torch::optimizer_momentum_sgd $params 0.05 0.9]
```

### CamelCase Alias
```tcl
set opt [torch::optimizerMomentumSgd -parameters $params -lr 0.01 -momentum 0.9]
```

## Error Handling

```tcl
# Missing required options
catch {torch::optimizer_momentum_sgd -lr 0.01 -momentum 0.9} err
puts $err  ;# => Required parameters missing or invalid

# Invalid learning rate
catch {torch::optimizer_momentum_sgd -parameters $params -lr -0.1 -momentum 0.9} err
```

## Compatibility

* **Backward Compatible:** Existing positional calls remain valid.
* **Forward Compatible:** Named-parameter syntax provides clearer, self-documenting code.
* **camelCase Alias:** `torch::optimizerMomentumSgd` offers a modern command name.

## See Also

* [torch::optimizer_sgd](optimizer_sgd.md) – Plain SGD without momentum
* [torch::optimizer_adam](optimizer_adam.md) – Adam optimizer
* [Optimizer Operations](../optimizers.md) 