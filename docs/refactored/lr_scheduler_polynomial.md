# torch::lr_scheduler_polynomial

Creates a polynomial learning rate scheduler that decays the learning rate using a polynomial function of the training progress.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_polynomial -optimizer OPTIMIZER_HANDLE -totalIters TOTAL_ITERATIONS [OPTIONS]
torch::lrSchedulerPolynomial -optimizer OPTIMIZER_HANDLE -totalIters TOTAL_ITERATIONS [OPTIONS]  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_polynomial OPTIMIZER_HANDLE TOTAL_ITERATIONS [POWER] [LAST_EPOCH]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | Yes | - | Handle to the optimizer to modify |
| `-totalIters` | integer | Yes | - | Total number of training iterations |
| `-power` | double | No | 1.0 | Polynomial power for decay rate |
| `-lastEpoch` | integer | No | -1 | Index of the last epoch |

### Parameter Details

- **`-optimizer`**: The optimizer handle whose learning rate will be adjusted
- **`-totalIters`** (aliases: `-total_iters`): Total number of training iterations over which to decay
- **`-power`**: The polynomial power that controls the decay rate:
  - `power = 1.0`: Linear decay
  - `power > 1.0`: Faster initial decay, slower later
  - `power < 1.0`: Slower initial decay, faster later
  - `power = 0.0`: Constant learning rate
- **`-lastEpoch`** (aliases: `-last_epoch`): Index of the last completed epoch (-1 means start from beginning)

## Return Value

Returns a string handle to the created learning rate scheduler that can be used with other scheduler operations.

## Mathematical Formula

The polynomial learning rate scheduler follows this formula:

```
lr = initial_lr * (1 - current_iter / total_iters)^power
```

Where:
- `initial_lr` is the optimizer's initial learning rate
- `current_iter` is the current iteration number
- `total_iters` is the total number of iterations
- `power` is the polynomial power parameter

## Examples

### Basic Usage

```tcl
# Create optimizer
set param_tensor [torch::tensor_create -data {0.5 0.3} -dtype float32]
set optimizer [torch::optimizer_sgd [list $param_tensor] 0.01]

# Create polynomial scheduler with linear decay (power=1.0)
set scheduler [torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 1000]

# Create polynomial scheduler with custom power
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 500 \
    -power 2.0]
```

### Advanced Configuration

```tcl
# Quadratic decay (faster initial reduction)
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 2000 \
    -power 2.0 \
    -lastEpoch 100]

# Square root decay (slower initial reduction)
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 1500 \
    -power 0.5]

# Constant learning rate (no decay)
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 1000 \
    -power 0.0]
```

### Parameter Order Flexibility

```tcl
# Parameters can be specified in any order
set scheduler [torch::lr_scheduler_polynomial \
    -power 1.5 \
    -lastEpoch 50 \
    -optimizer $optimizer \
    -totalIters 800]
```

### Using camelCase Alias

```tcl
# camelCase version works identically
set scheduler [torch::lrSchedulerPolynomial \
    -optimizer $optimizer \
    -totalIters 1200 \
    -power 3.0]
```

### Using Snake Case Aliases

```tcl
# Snake case aliases are also supported
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -total_iters 1000 \
    -last_epoch 25]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_polynomial $optimizer 1000 2.0 50]

# New named parameter syntax (equivalent)
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 1000 \
    -power 2.0 \
    -lastEpoch 50]

# Using defaults (optimizer and totalIters only)
# Old: torch::lr_scheduler_polynomial $optimizer 1000
# New: torch::lr_scheduler_polynomial -optimizer $optimizer -totalIters 1000
```

### Common Migration Patterns

```tcl
# Pattern 1: Basic scheduler with totalIters
# Old: torch::lr_scheduler_polynomial $opt 500
# New: torch::lr_scheduler_polynomial -optimizer $opt -totalIters 500

# Pattern 2: With custom power
# Old: torch::lr_scheduler_polynomial $opt 1000 1.5
# New: torch::lr_scheduler_polynomial -optimizer $opt -totalIters 1000 -power 1.5

# Pattern 3: All parameters
# Old: torch::lr_scheduler_polynomial $opt 800 2.0 25
# New: torch::lr_scheduler_polynomial -optimizer $opt -totalIters 800 -power 2.0 -lastEpoch 25
```

## Use Cases

### Linear Decay Training

```tcl
# Standard linear decay over training
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 5000 \
    -power 1.0]
```

### Aggressive Early Decay

```tcl
# Quadratic decay - reduces learning rate quickly early on
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 3000 \
    -power 2.0]
```

### Gentle Early Decay

```tcl
# Square root decay - maintains higher learning rate longer
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 4000 \
    -power 0.5]
```

### Fine-tuning Scenario

```tcl
# Start from a specific epoch for fine-tuning
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 1000 \
    -power 1.5 \
    -lastEpoch 200]
```

### Constant Learning Rate Phase

```tcl
# Use power=0 for constant learning rate (no decay)
set scheduler [torch::lr_scheduler_polynomial \
    -optimizer $optimizer \
    -totalIters 2000 \
    -power 0.0]
```

## Decay Behavior Examples

### Different Power Values

```tcl
# Linear decay (power = 1.0)
# LR drops uniformly: 100% -> 75% -> 50% -> 25% -> 0%
set scheduler [torch::lr_scheduler_polynomial -optimizer $opt -totalIters 1000 -power 1.0]

# Quadratic decay (power = 2.0) 
# LR drops faster early: 100% -> 56% -> 25% -> 6% -> 0%
set scheduler [torch::lr_scheduler_polynomial -optimizer $opt -totalIters 1000 -power 2.0]

# Square root decay (power = 0.5)
# LR drops slower early: 100% -> 87% -> 71% -> 50% -> 0%
set scheduler [torch::lr_scheduler_polynomial -optimizer $opt -totalIters 1000 -power 0.5]
```

## Error Handling

The command performs comprehensive parameter validation:

```tcl
# These will raise errors:
torch::lr_scheduler_polynomial  # Missing optimizer and totalIters
torch::lr_scheduler_polynomial -optimizer $opt  # Missing totalIters
torch::lr_scheduler_polynomial -optimizer "invalid" -totalIters 100  # Invalid optimizer
torch::lr_scheduler_polynomial -optimizer $opt -totalIters 0  # Zero totalIters
torch::lr_scheduler_polynomial -optimizer $opt -totalIters -100  # Negative totalIters
torch::lr_scheduler_polynomial -optimizer $opt -totalIters 100 -power -1.0  # Negative power
```

## Implementation Notes

- The scheduler calculates learning rate decay based on polynomial functions
- The `totalIters` parameter must be a positive integer
- The `power` parameter must be non-negative (≥ 0.0)
- `lastEpoch` can be negative, zero, or positive
- The scheduler maintains backward compatibility with the legacy positional syntax
- Snake case aliases (`-total_iters`, `-last_epoch`) are supported for convenience

## Mathematical Properties

### Power Parameter Effects

- **power = 0**: Constant learning rate (no decay)
- **0 < power < 1**: Slow initial decay, faster later (concave down)
- **power = 1**: Linear decay
- **power > 1**: Fast initial decay, slower later (concave up)

### Training Duration Impact

- Shorter `totalIters`: Faster overall decay
- Longer `totalIters`: More gradual decay
- The final learning rate approaches zero as iterations approach `totalIters`

## Related Commands

- `torch::lr_scheduler_step` - Step-based learning rate scheduling
- `torch::lr_scheduler_exponential` - Exponential decay scheduling
- `torch::lr_scheduler_cosine` - Cosine annealing scheduling
- `torch::lr_scheduler_plateau` - Plateau-based learning rate reduction
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## See Also

- [Learning Rate Scheduling Guide](../guides/learning_rate_scheduling.md)
- [Optimizer Documentation](../optimizers/README.md)
- [Training Best Practices](../guides/training_best_practices.md)
- [Polynomial Decay Theory](../theory/polynomial_decay.md) 