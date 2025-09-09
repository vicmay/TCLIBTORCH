# torch::lr_scheduler_exponential

Creates an exponential learning rate scheduler that decays the learning rate by a constant factor at each epoch.

## Syntax

### Current (Positional) - Backward Compatible
```tcl
torch::lr_scheduler_exponential optimizer ?gamma?
```

### New (Named Parameters) - Recommended
```tcl
torch::lr_scheduler_exponential -optimizer optimizer ?-gamma gamma?
torch::lr_scheduler_exponential -opt optimizer ?-decay gamma?
```

### camelCase Alias
```tcl
torch::lrSchedulerExponential -optimizer optimizer ?-gamma gamma?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-optimizer` or `-opt` | string | required | Handle of optimizer to schedule |
| `-gamma` or `-decay` | double | 0.95 | Multiplicative factor of learning rate decay |

**Note**: `gamma` should be a positive value less than or equal to 1.0 for decay.

## Returns

Returns a scheduler handle that can be used with `torch::lr_scheduler_step_update`.

## Description

The exponential learning rate scheduler multiplies the learning rate by `gamma` at each step:

```
lr_new = lr_initial × gamma^step_count
```

This provides exponential decay of the learning rate, which is useful for:
- Fine-tuning pre-trained models
- Stabilizing training in later epochs
- Gradual learning rate reduction

### Mathematical Properties

1. **Exponential Decay**: Learning rate decreases exponentially with each step
2. **Monotonic**: Always decreases (when gamma < 1.0)
3. **Continuous**: Smooth decay curve without sudden drops
4. **Configurable Rate**: Decay rate controlled by gamma parameter

## Examples

### Basic Usage

```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.1]

# Create exponential scheduler (positional syntax)
set scheduler [torch::lr_scheduler_exponential $optimizer 0.9]

# Create exponential scheduler (named syntax)
set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -gamma 0.9]

# Using camelCase alias
set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.9]
```

### Training Loop Integration

```tcl
# Setup
set model_params [torch::tensor_create {/* model parameters */} float32]
set optimizer [torch::optimizer_adam $model_params 0.001]
set scheduler [torch::lr_scheduler_exponential $optimizer 0.95]

# Training loop
for {set epoch 0} {$epoch < 100} {incr epoch} {
    # Training step
    # ... forward pass, loss computation, backward pass ...
    
    # Step optimizer
    torch::optimizer_step $optimizer
    
    # Step learning rate scheduler
    torch::lr_scheduler_step_update $scheduler
    
    # Monitor learning rate
    set current_lr [torch::get_lr $optimizer]
    puts "Epoch $epoch: LR = $current_lr"
}
```

### Different Decay Rates

```tcl
set optimizer [torch::optimizer_sgd $weights 0.1]

# Conservative decay (slow)
set slow_scheduler [torch::lr_scheduler_exponential $optimizer 0.99]

# Moderate decay
set medium_scheduler [torch::lr_scheduler_exponential $optimizer 0.95]

# Aggressive decay (fast)
set fast_scheduler [torch::lr_scheduler_exponential $optimizer 0.8]
```

### Fine-tuning Scenario

```tcl
# Pre-trained model with lower learning rate
set pretrained_params [torch::tensor_create {/* pre-trained weights */} float32]
set optimizer [torch::optimizer_adam $pretrained_params 0.0001]

# Very conservative decay for fine-tuning
set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -gamma 0.98]

# Fine-tuning loop
for {set epoch 0} {$epoch < 50} {incr epoch} {
    # Fine-tuning steps...
    torch::optimizer_step $optimizer
    torch::lr_scheduler_step_update $scheduler
}
```

## Common Use Cases

### 1. Model Fine-tuning
```tcl
# For fine-tuning pre-trained models
set scheduler [torch::lr_scheduler_exponential $optimizer 0.98]
```

### 2. Stabilizing Training
```tcl
# For stable convergence in later training phases
set scheduler [torch::lr_scheduler_exponential $optimizer 0.95]
```

### 3. Gradual Learning Rate Reduction
```tcl
# For smooth learning rate reduction
set scheduler [torch::lr_scheduler_exponential $optimizer 0.9]
```

## Learning Rate Progression

| Step | LR (γ=0.9) | LR (γ=0.95) | LR (γ=0.8) |
|------|------------|-------------|------------|
| 0    | 0.1000     | 0.1000      | 0.1000     |
| 1    | 0.0900     | 0.0950      | 0.0800     |
| 2    | 0.0810     | 0.0903      | 0.0640     |
| 5    | 0.0590     | 0.0774      | 0.0328     |
| 10   | 0.0349     | 0.0599      | 0.0107     |
| 20   | 0.0122     | 0.0358      | 0.0012     |

## Error Handling

```tcl
# Missing optimizer parameter
catch {torch::lr_scheduler_exponential} error
# Error: Required parameter missing: -optimizer

# Invalid optimizer handle
catch {torch::lr_scheduler_exponential invalid_optimizer} error
# Error: Invalid optimizer name

# Invalid gamma value
catch {torch::lr_scheduler_exponential $optimizer -0.1} error
# Error: Invalid gamma parameter (must be positive)

# Zero gamma (no decay)
catch {torch::lr_scheduler_exponential $optimizer 0.0} error
# Error: Invalid gamma parameter (must be positive)

# Missing value for named parameter
catch {torch::lr_scheduler_exponential -optimizer} error
# Error: Missing value for parameter -optimizer
```

## Parameter Validation

- **Optimizer**: Must be a valid optimizer handle from `torch::optimizer_*`
- **Gamma**: Must be positive (> 0.0)
  - Values < 1.0: Decay (typical usage)
  - Value = 1.0: No change in learning rate
  - Values > 1.0: Increase (unusual but valid)

## Integration with Optimizers

### Supported Optimizers
```tcl
# SGD
set sgd_opt [torch::optimizer_sgd $params 0.1]
set scheduler [torch::lr_scheduler_exponential $sgd_opt 0.9]

# Adam
set adam_opt [torch::optimizer_adam $params 0.001]
set scheduler [torch::lr_scheduler_exponential $adam_opt 0.95]

# AdamW
set adamw_opt [torch::optimizer_adamw $params 0.001]
set scheduler [torch::lr_scheduler_exponential $adamw_opt 0.98]
```

### Scheduler Operations
```tcl
# Step the scheduler (updates learning rate)
torch::lr_scheduler_step_update $scheduler

# Get current learning rate
set current_lr [torch::get_lr $optimizer]

# Multiple schedulers (though typically not needed)
set scheduler1 [torch::lr_scheduler_exponential $optimizer 0.9]
set scheduler2 [torch::lr_scheduler_exponential $optimizer 0.8]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_exponential $optimizer]
set scheduler [torch::lr_scheduler_exponential $optimizer 0.9]

# New named parameter syntax
set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer]
set scheduler [torch::lr_scheduler_exponential -optimizer $optimizer -gamma 0.9]
```

### Parameter Name Flexibility

```tcl
# These are equivalent
torch::lr_scheduler_exponential -optimizer $opt -gamma 0.9
torch::lr_scheduler_exponential -opt $opt -decay 0.9
```

### camelCase Usage

```tcl
# snake_case (traditional)
torch::lr_scheduler_exponential -optimizer $opt -gamma 0.9

# camelCase (modern)  
torch::lrSchedulerExponential -optimizer $opt -gamma 0.9
```

## Mathematical Analysis

### Decay Rate Comparison

```tcl
# Compare different gamma values
set optimizer [torch::optimizer_sgd $weights 0.1]

# After 10 steps:
# γ = 0.99: LR = 0.1 × 0.99^10 ≈ 0.0905 (9.5% reduction)
# γ = 0.95: LR = 0.1 × 0.95^10 ≈ 0.0599 (40.1% reduction)  
# γ = 0.90: LR = 0.1 × 0.90^10 ≈ 0.0349 (65.1% reduction)
# γ = 0.80: LR = 0.1 × 0.80^10 ≈ 0.0107 (89.3% reduction)
```

### Half-life Calculation

The number of steps to reduce learning rate by half:
```
half_life = ln(0.5) / ln(gamma)

γ = 0.99: ~69 steps
γ = 0.95: ~14 steps  
γ = 0.90: ~7 steps
γ = 0.80: ~3 steps
```

## Performance Considerations

1. **Minimal Overhead**: Scheduler has negligible computational cost
2. **Memory Efficient**: Stores only gamma and step count
3. **Fast Updates**: O(1) learning rate computation
4. **Scalable**: Works with any optimizer size

## Best Practices

### Choosing Gamma Values

- **0.95-0.99**: Conservative decay for fine-tuning
- **0.9-0.95**: Moderate decay for standard training
- **0.8-0.9**: Aggressive decay for quick convergence
- **0.5-0.8**: Very aggressive decay (use with caution)

### Training Strategies

```tcl
# Strategy 1: Start aggressive, then conservative
set scheduler [torch::lr_scheduler_exponential $optimizer 0.8]
# ... train for 20 epochs ...
set scheduler [torch::lr_scheduler_exponential $optimizer 0.95]

# Strategy 2: Two-phase training
# Phase 1: Regular training
# Phase 2: Fine-tuning with exponential decay
set scheduler [torch::lr_scheduler_exponential $optimizer 0.98]
```

## Comparison with Other Schedulers

### vs. Step Scheduler
```tcl
# Step: Sudden drops at specific intervals
set step_sched [torch::lr_scheduler_step $optimizer 10 0.5]

# Exponential: Smooth continuous decay
set exp_sched [torch::lr_scheduler_exponential $optimizer 0.933]
```

### vs. Cosine Annealing
```tcl
# Cosine: Periodic annealing pattern
set cos_sched [torch::lr_scheduler_cosine $optimizer 50]

# Exponential: Monotonic decay
set exp_sched [torch::lr_scheduler_exponential $optimizer 0.95]
```

## Troubleshooting

### Common Issues

1. **Learning rate becomes too small**: Use larger gamma (closer to 1.0)
2. **Decay too slow**: Use smaller gamma
3. **Training becomes unstable**: Learning rate might be decaying too fast

### Monitoring

```tcl
# Log learning rate changes
proc log_lr {optimizer epoch} {
    set lr [torch::get_lr $optimizer]
    puts "Epoch $epoch: Learning Rate = [format "%.6f" $lr]"
}

# Use in training loop
torch::lr_scheduler_step_update $scheduler
log_lr $optimizer $epoch
```

## See Also

- [`torch::lr_scheduler_step`](lr_scheduler_step.md) - Step-based learning rate scheduler
- [`torch::lr_scheduler_cosine`](lr_scheduler_cosine.md) - Cosine annealing scheduler
- [`torch::optimizer_sgd`](optimizer_sgd.md) - SGD optimizer
- [`torch::optimizer_adam`](optimizer_adam.md) - Adam optimizer
- [`torch::get_lr`](get_lr.md) - Get current learning rate

## Implementation Details

- **Backend**: Custom implementation using LibTorch scheduler framework
- **Memory**: Stores optimizer reference, gamma, and step count
- **Thread Safety**: Safe for concurrent use with different schedulers
- **Precision**: Uses double precision for gamma calculations

---

*This documentation covers LibTorch TCL Extension v2.0+ with dual syntax support.* 