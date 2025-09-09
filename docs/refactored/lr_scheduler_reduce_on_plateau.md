# torch::lr_scheduler_reduce_on_plateau

Creates a learning rate scheduler that reduces the learning rate when a monitored metric has stopped improving (reached a plateau). This scheduler provides more advanced control over plateau detection compared to the basic plateau scheduler.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_reduce_on_plateau -optimizer OPTIMIZER_HANDLE [OPTIONS]
torch::lrSchedulerReduceOnPlateau -optimizer OPTIMIZER_HANDLE [OPTIONS]  # camelCase alias
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_reduce_on_plateau OPTIMIZER_HANDLE [MODE] [FACTOR] [PATIENCE] [THRESHOLD] [THRESHOLD_MODE] [MIN_LR]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-optimizer` | string | Yes | - | Handle to the optimizer to modify |
| `-mode` | string | No | "min" | Mode for monitoring metrics ("min" or "max") |
| `-factor` | double | No | 0.1 | Factor to reduce learning rate (0 < factor ≤ 1) |
| `-patience` | integer | No | 10 | Number of epochs to wait before reducing LR |
| `-threshold` | double | No | 1e-4 | Threshold for measuring improvement |
| `-thresholdMode` | string | No | "rel" | Threshold mode ("rel" or "abs") |
| `-minLr` | double | No | 0.0 | Minimum learning rate |

### Parameter Details

- **`-optimizer`**: The optimizer handle whose learning rate will be adjusted
- **`-mode`**: 
  - `"min"`: Reduce LR when metric stops decreasing
  - `"max"`: Reduce LR when metric stops increasing
- **`-factor`**: Multiplication factor for learning rate reduction (new_lr = lr * factor)
- **`-patience`**: Number of epochs with no improvement before reducing learning rate
- **`-threshold`**: Threshold for measuring improvement in the monitored metric
- **`-thresholdMode`** (aliases: `-threshold_mode`): Mode for using the threshold:
  - `"rel"`: Relative improvement threshold
  - `"abs"`: Absolute improvement threshold
- **`-minLr`** (aliases: `-min_lr`): Minimum learning rate (LR won't be reduced below this value)

## Return Value

Returns a string handle to the created learning rate scheduler that can be used with other scheduler operations.

## Examples

### Basic Usage

```tcl
# Create optimizer
set param_tensor [torch::tensor_create -data {0.5 0.3} -dtype float32]
set optimizer [torch::optimizer_sgd [list $param_tensor] 0.01]

# Create reduce on plateau scheduler with defaults
set scheduler [torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer]

# Create scheduler for maximizing metrics (e.g., accuracy)
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "max"]
```

### Advanced Configuration

```tcl
# Aggressive reduction with custom parameters
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.5 \
    -patience 5 \
    -threshold 1e-3 \
    -thresholdMode "abs" \
    -minLr 1e-6]

# Conservative reduction for fine-tuning
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -factor 0.8 \
    -patience 20 \
    -threshold 1e-5 \
    -minLr 1e-7]
```

### Parameter Order Flexibility

```tcl
# Parameters can be specified in any order
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -patience 15 \
    -factor 0.3 \
    -threshold 1e-4 \
    -optimizer $optimizer \
    -mode "max" \
    -minLr 1e-8 \
    -thresholdMode "rel"]
```

### Using camelCase Alias

```tcl
# camelCase version works identically
set scheduler [torch::lrSchedulerReduceOnPlateau \
    -optimizer $optimizer \
    -factor 0.2 \
    -patience 8]
```

### Using Snake Case Aliases

```tcl
# Snake case aliases are also supported
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -threshold_mode "abs" \
    -min_lr 1e-6]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_reduce_on_plateau $optimizer "min" 0.1 10 1e-4 "rel" 0.0]

# New named parameter syntax (equivalent)
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.1 \
    -patience 10 \
    -threshold 1e-4 \
    -thresholdMode "rel" \
    -minLr 0.0]

# Using defaults (optimizer only)
# Old: torch::lr_scheduler_reduce_on_plateau $optimizer
# New: torch::lr_scheduler_reduce_on_plateau -optimizer $optimizer
```

### Common Migration Patterns

```tcl
# Pattern 1: Basic scheduler
# Old: torch::lr_scheduler_reduce_on_plateau $opt
# New: torch::lr_scheduler_reduce_on_plateau -optimizer $opt

# Pattern 2: With mode
# Old: torch::lr_scheduler_reduce_on_plateau $opt "max"  
# New: torch::lr_scheduler_reduce_on_plateau -optimizer $opt -mode "max"

# Pattern 3: With multiple parameters
# Old: torch::lr_scheduler_reduce_on_plateau $opt "min" 0.5 20
# New: torch::lr_scheduler_reduce_on_plateau -optimizer $opt -mode "min" -factor 0.5 -patience 20

# Pattern 4: All parameters
# Old: torch::lr_scheduler_reduce_on_plateau $opt "max" 0.2 15 1e-3 "abs" 1e-6
# New: torch::lr_scheduler_reduce_on_plateau -optimizer $opt -mode "max" -factor 0.2 -patience 15 -threshold 1e-3 -thresholdMode "abs" -minLr 1e-6
```

## Use Cases

### Training Loss Monitoring

```tcl
# Standard setup for monitoring training loss
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.1 \
    -patience 10 \
    -threshold 1e-4]
```

### Validation Accuracy Monitoring

```tcl
# Setup for monitoring validation accuracy
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "max" \
    -factor 0.5 \
    -patience 5 \
    -threshold 1e-3]
```

### Fine-tuning Scenarios

```tcl
# Conservative approach for fine-tuning
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -factor 0.9 \
    -patience 15 \
    -threshold 1e-5 \
    -minLr 1e-7]
```

### Aggressive Training

```tcl
# Aggressive reduction for quick convergence
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -factor 0.2 \
    -patience 3 \
    -threshold 1e-3 \
    -thresholdMode "abs"]
```

### Different Threshold Modes

```tcl
# Relative threshold (default) - percentage improvement
set scheduler_rel [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -threshold 0.01 \
    -thresholdMode "rel"]  # 1% relative improvement

# Absolute threshold - absolute value improvement
set scheduler_abs [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -threshold 0.001 \
    -thresholdMode "abs"]  # 0.001 absolute improvement
```

## Threshold Mode Examples

### Relative Threshold Mode ("rel")

```tcl
# With relative threshold, improvement is calculated as:
# improvement = (best_value - current_value) / max(abs(best_value), threshold)
# For minimization: improvement = (best_value - current_value) / max(abs(best_value), 1e-4)

set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -threshold 0.01 \
    -thresholdMode "rel"]  # Requires 1% relative improvement
```

### Absolute Threshold Mode ("abs")

```tcl
# With absolute threshold, improvement is calculated as:
# improvement = best_value - current_value (for minimization)
# improvement = current_value - best_value (for maximization)

set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -threshold 0.001 \
    -thresholdMode "abs"]  # Requires 0.001 absolute improvement
```

## Monitoring Different Metrics

### Loss Monitoring

```tcl
# Monitor training or validation loss (minimize)
set loss_scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "min" \
    -factor 0.1 \
    -patience 10]
```

### Accuracy Monitoring

```tcl
# Monitor validation accuracy (maximize)
set acc_scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "max" \
    -factor 0.5 \
    -patience 8]
```

### Custom Metric Monitoring

```tcl
# Monitor F1 score, AUC, or other metrics (maximize)
set metric_scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -mode "max" \
    -factor 0.3 \
    -patience 12 \
    -threshold 1e-3 \
    -thresholdMode "abs"]
```

## Error Handling

The command performs comprehensive parameter validation:

```tcl
# These will raise errors:
torch::lr_scheduler_reduce_on_plateau  # Missing optimizer
torch::lr_scheduler_reduce_on_plateau -optimizer "invalid"  # Invalid optimizer handle
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -mode "invalid"  # Invalid mode
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -factor -0.1  # Negative factor
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -factor 1.5   # Factor > 1
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -patience 0   # Zero patience
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -patience -5  # Negative patience
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -threshold -1e-4  # Negative threshold
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -thresholdMode "invalid"  # Invalid threshold mode
torch::lr_scheduler_reduce_on_plateau -optimizer $opt -minLr -1e-6  # Negative minLr
```

## Implementation Notes

- The scheduler monitors a metric externally and reduces learning rate when no improvement is detected
- The `factor` parameter must be between 0 and 1 (exclusive of 0, inclusive of 1)
- `patience` must be a positive integer
- `threshold` must be non-negative
- `minLr` must be non-negative
- Mode must be either "min" (for metrics to minimize) or "max" (for metrics to maximize)
- Threshold mode must be either "rel" (relative) or "abs" (absolute)
- The scheduler maintains backward compatibility with the legacy positional syntax
- Snake case aliases (`-threshold_mode`, `-min_lr`) are supported for convenience

## Advanced Features

### Multiple Threshold Modes

The scheduler supports two threshold modes for determining improvement:

1. **Relative mode (`"rel"`)**: Measures improvement as a percentage of the current best value
2. **Absolute mode (`"abs"`)**: Measures improvement as an absolute difference

### Minimum Learning Rate Protection

The `minLr` parameter prevents the learning rate from being reduced below a specified minimum:

```tcl
# Learning rate won't go below 1e-6
set scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -factor 0.1 \
    -minLr 1e-6]
```

### Patience Configuration

The patience parameter controls how many epochs to wait before reducing the learning rate:

```tcl
# Very patient - wait 50 epochs before reducing
set patient_scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -patience 50]

# Impatient - reduce after just 2 epochs
set impatient_scheduler [torch::lr_scheduler_reduce_on_plateau \
    -optimizer $optimizer \
    -patience 2]
```

## Related Commands

- `torch::lr_scheduler_plateau` - Basic plateau-based learning rate reduction
- `torch::lr_scheduler_step` - Step-based learning rate scheduling
- `torch::lr_scheduler_exponential` - Exponential decay scheduling
- `torch::lr_scheduler_polynomial` - Polynomial decay scheduling
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## See Also

- [Learning Rate Scheduling Guide](../guides/learning_rate_scheduling.md)
- [Optimizer Documentation](../optimizers/README.md)
- [Training Best Practices](../guides/training_best_practices.md)
- [Plateau Detection Theory](../theory/plateau_detection.md) 