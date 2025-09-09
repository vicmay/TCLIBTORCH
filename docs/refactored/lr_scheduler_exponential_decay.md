# torch::lr_scheduler_exponential_decay

Creates an exponential learning rate scheduler that decays the learning rate by a multiplicative factor at each step.

## Syntax

### Current (Positional - Backward Compatible)
```tcl
torch::lr_scheduler_exponential_decay optimizer gamma
```

### New (Named Parameters)
```tcl
torch::lr_scheduler_exponential_decay -optimizer optimizer_handle -gamma decay_factor
```

### camelCase Alias
```tcl
torch::lrSchedulerExponentialDecay -optimizer optimizer_handle -gamma decay_factor
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-optimizer` | string | *required* | Handle to the optimizer to schedule |
| `-gamma` | double | 0.95 | Multiplicative factor for learning rate decay (0 < gamma ≤ 1) |

## Description

The exponential decay scheduler multiplies the learning rate by `gamma` at each step, resulting in exponential decay. This is one of the simplest and most commonly used learning rate scheduling strategies.

### Mathematical Formula
```
lr_new = lr_old * gamma
```

Where:
- `lr_old` is the current learning rate
- `gamma` is the decay factor
- `lr_new` is the new learning rate after decay

### Decay Behavior
- **gamma = 0.95**: Moderate decay (5% reduction per step)
- **gamma = 0.9**: Faster decay (10% reduction per step)
- **gamma = 0.99**: Slow decay (1% reduction per step)
- **gamma = 1.0**: No decay (learning rate remains constant)

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create model parameters and optimizer
set param1 [torch::rand {100 50}]
set param2 [torch::rand {50 10}]
set optimizer [torch::optimizer_sgd [list $param1 $param2] 0.01]

# Create exponential decay scheduler with 5% decay per step
set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.95]
```

### Named Parameter Syntax
```tcl
# Create exponential decay scheduler with named parameters
set scheduler [torch::lr_scheduler_exponential_decay \
    -optimizer $optimizer \
    -gamma 0.9]
```

### camelCase Syntax
```tcl
# Using the modern camelCase alias
set scheduler [torch::lrSchedulerExponentialDecay \
    -optimizer $optimizer \
    -gamma 0.95]
```

### Parameter Order Flexibility
```tcl
# Named parameters can be in any order
set scheduler [torch::lr_scheduler_exponential_decay \
    -gamma 0.85 \
    -optimizer $optimizer]
```

### Different Decay Rates
```tcl
# Slow decay (1% per step)
set slow_scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.99]

# Moderate decay (5% per step)  
set moderate_scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.95]

# Fast decay (20% per step)
set fast_scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.8]
```

## Learning Rate Evolution

Here's how the learning rate evolves with different gamma values:

| Step | γ=0.99 | γ=0.95 | γ=0.9 | γ=0.8 |
|------|--------|--------|-------|-------|
| 0    | 0.010  | 0.010  | 0.010 | 0.010 |
| 10   | 0.0095 | 0.0060 | 0.0035| 0.0011|
| 50   | 0.0061 | 0.0008 | 0.0001| 0.0000|
| 100  | 0.0037 | 0.0001 | 0.0000| 0.0000|

## Use Cases

### 1. Training Stabilization
```tcl
# Use slow decay to stabilize training near convergence
set stabilizing_scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.995]
```

### 2. Quick Decay for Fine-tuning
```tcl
# Fast decay for fine-tuning pre-trained models
set finetuning_scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.85]
```

### 3. Multiple Optimizers
```tcl
# Different decay rates for different optimizers
set main_optimizer [torch::optimizer_adam $main_params 0.001]
set aux_optimizer [torch::optimizer_sgd $aux_params 0.01]

set main_scheduler [torch::lr_scheduler_exponential_decay $main_optimizer 0.95]
set aux_scheduler [torch::lr_scheduler_exponential_decay $aux_optimizer 0.9]
```

## Integration with Training Loop

```tcl
# Training loop with exponential decay
set optimizer [torch::optimizer_adam $model_params 0.001]
set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.98]

for {set epoch 0} {$epoch < 100} {incr epoch} {
    # Training step
    foreach batch $training_data {
        # Forward pass, compute loss, backward pass
        # ...
        
        # Update parameters
        torch::optimizer_step $optimizer
    }
    
    # Update learning rate (call after each epoch or step)
    torch::lr_scheduler_step $scheduler
    
    # Optional: Get current learning rate for logging
    set current_lr [torch::get_lr $optimizer]
    puts "Epoch $epoch: LR = $current_lr"
}
```

## Error Handling

### Invalid Optimizer Handle
```tcl
# This will raise an error
catch {torch::lr_scheduler_exponential_decay "invalid_handle" 0.95} error
puts "Error: $error"
# Output: Invalid optimizer handle
```

### Invalid Gamma Values
```tcl
# Gamma must be between 0 and 1
catch {torch::lr_scheduler_exponential_decay $optimizer -0.1} error
puts "Error: $error"
# Output: Required parameters missing or invalid (gamma must be between 0 and 1)

catch {torch::lr_scheduler_exponential_decay $optimizer 1.5} error  
puts "Error: $error"
# Output: Required parameters missing or invalid (gamma must be between 0 and 1)
```

### Missing Parameters
```tcl
# Named syntax requires optimizer parameter
catch {torch::lr_scheduler_exponential_decay -gamma 0.95} error
puts "Error: $error"
# Output: Required parameters missing or invalid (optimizer handle required)
```

## Return Value

Returns a string handle for the created scheduler (e.g., "scheduler1", "scheduler2", etc.) that can be used with other scheduler functions like `torch::lr_scheduler_step`.

## Comparison with Other Schedulers

| Scheduler Type | Use Case | Decay Pattern |
|----------------|----------|---------------|
| Exponential Decay | Simple, predictable decay | Multiplicative |
| Step Decay | Plateau-based reduction | Step function |
| Cosine Annealing | Smooth oscillating decay | Cosine curve |
| Polynomial Decay | Smooth polynomial decay | Power function |

## Best Practices

1. **Start with gamma = 0.95**: A good default for most applications
2. **Monitor training loss**: Adjust gamma based on convergence behavior
3. **Use with warmup**: Combine with warmup for better training stability
4. **Log learning rates**: Track LR changes to understand training dynamics
5. **Validate on held-out data**: Ensure decay doesn't hurt generalization

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_exponential_decay $optimizer 0.95]

# New named parameter syntax (equivalent)
set scheduler [torch::lr_scheduler_exponential_decay -optimizer $optimizer -gamma 0.95]

# Modern camelCase alias
set scheduler [torch::lrSchedulerExponentialDecay -optimizer $optimizer -gamma 0.95]
```

### Advantages of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Order independent**: Parameters can be specified in any order
- **Future-proof**: New parameters can be added without breaking existing code
- **Error prevention**: Less likely to mix up parameter positions

## See Also

- `torch::lr_scheduler_step` - Step the scheduler
- `torch::lr_scheduler_exponential` - Standard exponential scheduler  
- `torch::lr_scheduler_cosine` - Cosine annealing scheduler
- `torch::get_lr` - Get current learning rate
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer 