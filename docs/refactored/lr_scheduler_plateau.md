# torch::lr_scheduler_plateau

Creates a learning rate scheduler that reduces the learning rate when a monitored metric has stopped improving (reached a plateau).

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_plateau -optimizer OPTIMIZER_HANDLE [OPTIONS]
torch::lrSchedulerPlateau -optimizer OPTIMIZER_HANDLE [OPTIONS]  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_plateau OPTIMIZER_HANDLE [MODE] [FACTOR] [PATIENCE]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | Yes | - | Handle to the optimizer to modify |
| `-mode` | string | No | "min" | Mode for monitoring metrics ("min" or "max") |
| `-factor` | double | No | 0.1 | Factor to reduce learning rate (0 < factor ≤ 1) |
| `-patience` | integer | No | 10 | Number of epochs to wait before reducing LR |

### Parameter Details

- **`-optimizer`**: The optimizer handle whose learning rate will be adjusted
- **`-mode`**: 
  - `"min"`: Reduce LR when metric stops decreasing
  - `"max"`: Reduce LR when metric stops increasing
- **`-factor`**: Multiplication factor for learning rate reduction (new_lr = lr * factor)
- **`-patience`**: Number of epochs with no improvement before reducing learning rate

## Return Value

Returns a string handle to the created learning rate scheduler that can be used with other scheduler operations.

## Examples

### Basic Usage

```tcl
# Create optimizer
set param_tensor [torch::tensor_create -data {0.5 0.3} -dtype float32]
set optimizer [torch::optimizer_sgd [list $param_tensor] 0.01]

# Create plateau scheduler with defaults (min mode, factor=0.1, patience=10)
set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer]

# Create plateau scheduler for maximizing metrics
set scheduler [torch::lr_scheduler_plateau -optimizer $optimizer -mode "max"]
```

### Advanced Configuration

```tcl
# Create plateau scheduler with custom parameters
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.5 \
    -patience 5]

# More aggressive reduction
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -factor 0.2 \
    -patience 3]
```

### Parameter Order Flexibility

```tcl
# Parameters can be specified in any order
set scheduler [torch::lr_scheduler_plateau \
    -patience 15 \
    -factor 0.3 \
    -optimizer $optimizer \
    -mode "max"]
```

### Using camelCase Alias

```tcl
# camelCase version works identically
set scheduler [torch::lrSchedulerPlateau -optimizer $optimizer -factor 0.5]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_plateau $optimizer "min" 0.1 10]

# New named parameter syntax (equivalent)
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.1 \
    -patience 10]

# Using defaults (optimizer only)
# Old: torch::lr_scheduler_plateau $optimizer
# New: torch::lr_scheduler_plateau -optimizer $optimizer
```

### Common Migration Patterns

```tcl
# Pattern 1: Basic scheduler
# Old: torch::lr_scheduler_plateau $opt
# New: torch::lr_scheduler_plateau -optimizer $opt

# Pattern 2: With mode
# Old: torch::lr_scheduler_plateau $opt "max"  
# New: torch::lr_scheduler_plateau -optimizer $opt -mode "max"

# Pattern 3: All parameters
# Old: torch::lr_scheduler_plateau $opt "min" 0.5 20
# New: torch::lr_scheduler_plateau -optimizer $opt -mode "min" -factor 0.5 -patience 20
```

## Use Cases

### Training Loss Monitoring

```tcl
# For monitoring training loss (minimize)
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.1 \
    -patience 10]
```

### Validation Accuracy Monitoring

```tcl
# For monitoring validation accuracy (maximize)
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -mode "max" \
    -factor 0.5 \
    -patience 5]
```

### Conservative Learning Rate Reduction

```tcl
# Conservative approach - wait longer, reduce less
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -factor 0.8 \
    -patience 20]
```

### Aggressive Learning Rate Reduction

```tcl
# Aggressive approach - reduce quickly and significantly
set scheduler [torch::lr_scheduler_plateau \
    -optimizer $optimizer \
    -factor 0.2 \
    -patience 3]
```

## Error Handling

The command performs comprehensive parameter validation:

```tcl
# These will raise errors:
torch::lr_scheduler_plateau  # Missing optimizer
torch::lr_scheduler_plateau -optimizer "invalid"  # Invalid optimizer handle
torch::lr_scheduler_plateau -optimizer $opt -mode "invalid"  # Invalid mode
torch::lr_scheduler_plateau -optimizer $opt -factor -0.1  # Negative factor
torch::lr_scheduler_plateau -optimizer $opt -factor 1.5   # Factor > 1
torch::lr_scheduler_plateau -optimizer $opt -patience 0   # Zero patience
torch::lr_scheduler_plateau -optimizer $opt -patience -5  # Negative patience
```

## Implementation Notes

- The scheduler monitors a metric externally and reduces learning rate when no improvement is detected
- The `factor` parameter must be between 0 and 1 (exclusive of 0, inclusive of 1)
- `patience` must be a positive integer
- Mode must be either "min" (for metrics to minimize) or "max" (for metrics to maximize)
- The scheduler maintains backward compatibility with the legacy positional syntax

## Related Commands

- `torch::lr_scheduler_step` - Step-based learning rate scheduling
- `torch::lr_scheduler_exponential` - Exponential decay scheduling
- `torch::lr_scheduler_cosine` - Cosine annealing scheduling
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## See Also

- [Learning Rate Scheduling Guide](../guides/learning_rate_scheduling.md)
- [Optimizer Documentation](../optimizers/README.md)
- [Training Best Practices](../guides/training_best_practices.md) 