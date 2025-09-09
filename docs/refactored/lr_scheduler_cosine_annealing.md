# torch::lr_scheduler_cosine_annealing

## Overview
Creates a standard cosine annealing learning rate scheduler that adjusts the learning rate following a cosine annealing schedule. This is similar to `torch::lr_scheduler_cosine` but follows the standard PyTorch CosineAnnealingLR implementation. The learning rate decreases following a cosine curve from the initial value to a minimum value (`eta_min`) over `T_max` iterations.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_cosine_annealing optimizer T_max ?eta_min?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_cosine_annealing -optimizer optimizer -tMax T_max ?-etaMin eta_min?
```

### camelCase Alias
```tcl
torch::lrSchedulerCosineAnnealing -optimizer optimizer -tMax T_max ?-etaMin eta_min?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `optimizer` | string | Yes | - | Handle to the optimizer whose learning rate will be scheduled |
| `T_max` | integer | Yes | - | Maximum number of iterations for the cosine annealing cycle |
| `eta_min` | double | No | 0.0 | Minimum learning rate value |

### Parameter Aliases

#### T_max Parameter
- `-tMax` (recommended)
- `-t_max` 
- `-T_max`

#### eta_min Parameter
- `-etaMin` (recommended)
- `-eta_min`

## Returns
Returns a scheduler handle string that can be used with `torch::lr_scheduler_step_update` to update the learning rate.

## Mathematical Formula
The cosine annealing learning rate is calculated using:

```
lr = eta_min + (initial_lr - eta_min) * (1 + cos(π * step_count / T_max)) / 2
```

Where:
- `initial_lr` is the learning rate from the optimizer when the scheduler is created
- `step_count` is the current number of steps (starts at 0)
- `T_max` is the maximum number of iterations for one cycle
- `eta_min` is the minimum learning rate

## Usage Examples

### Basic Usage with Positional Syntax
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.1]

# Create cosine annealing scheduler with T_max=100, eta_min=0.0 (default)
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100]

# Step the scheduler
torch::lr_scheduler_step_update $scheduler
set current_lr [torch::get_lr $optimizer]
puts "Learning rate: $current_lr"
```

### Named Parameter Syntax
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_adam $weights 0.01]

# Create cosine annealing scheduler with specific eta_min
set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 200 -etaMin 0.001]

# Step through training loop
for {set epoch 0} {$epoch < 50} {incr epoch} {
    # ... training code ...
    torch::lr_scheduler_step_update $scheduler
    set lr [torch::get_lr $optimizer]
    puts "Epoch $epoch, Learning rate: $lr"
}
```

### camelCase Alias
```tcl
# Using camelCase alias with named parameters
set scheduler [torch::lrSchedulerCosineAnnealing -optimizer $optimizer -tMax 150 -etaMin 0.005]
```

### Parameter Aliases
```tcl
# Different ways to specify T_max
set scheduler1 [torch::lr_scheduler_cosine_annealing -optimizer $opt -tMax 100]
set scheduler2 [torch::lr_scheduler_cosine_annealing -optimizer $opt -t_max 100]
set scheduler3 [torch::lr_scheduler_cosine_annealing -optimizer $opt -T_max 100]

# Different ways to specify eta_min
set scheduler4 [torch::lr_scheduler_cosine_annealing -optimizer $opt -tMax 100 -etaMin 0.01]
set scheduler5 [torch::lr_scheduler_cosine_annealing -optimizer $opt -tMax 100 -eta_min 0.01]
```

## Training Loop Integration
```tcl
# Complete training example with cosine annealing
set model [torch::linear 784 10]
set optimizer [torch::optimizer_sgd [list $model] 0.1]
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100 0.001]

for {set epoch 0} {$epoch < 100} {incr epoch} {
    # Forward pass
    set output [torch::linear_forward $model $input]
    set loss [torch::mse_loss $output $target]
    
    # Backward pass
    torch::optimizer_zero_grad $optimizer
    torch::tensor_backward $loss
    torch::optimizer_step $optimizer
    
    # Update learning rate
    torch::lr_scheduler_step_update $scheduler
    
    if {$epoch % 10 == 0} {
        set current_lr [torch::get_lr $optimizer]
        puts "Epoch $epoch: Loss = [torch::tensor_item $loss], LR = $current_lr"
    }
}
```

## Behavior Characteristics

### Learning Rate Pattern
- **Start**: Learning rate begins at the optimizer's initial learning rate
- **Decrease**: Follows a smooth cosine curve downward
- **Minimum**: Reaches `eta_min` at step `T_max/2`
- **Increase**: Continues cosine curve upward back toward initial learning rate
- **Cycle**: Completes one full cosine cycle at step `T_max`
- **Continuation**: After `T_max`, the pattern repeats

### Key Points
- At step 0: lr = initial_lr
- At step T_max/2: lr ≈ eta_min (minimum)
- At step T_max: lr = initial_lr (cycle completes)
- Beyond T_max: Pattern repeats cyclically

## Differences from torch::lr_scheduler_cosine

While both commands implement cosine annealing, there are subtle differences:

| Feature | torch::lr_scheduler_cosine | torch::lr_scheduler_cosine_annealing |
|---------|---------------------------|-------------------------------------|
| **Algorithm** | Custom cosine implementation | Standard PyTorch CosineAnnealingLR |
| **Cycle Behavior** | Configurable cycle behavior | Standard single cycle |
| **Implementation** | LibTorch TCL custom | Matches PyTorch behavior |
| **Use Case** | Experimental/custom schedules | Standard training workflows |

Both commands use the same mathematical formula and produce equivalent results.

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid optimizer
torch::lr_scheduler_cosine_annealing "invalid_optimizer" 100
# Error: Invalid optimizer name

# Missing required T_max
torch::lr_scheduler_cosine_annealing $optimizer
# Error: Invalid number of arguments

# Invalid T_max type
torch::lr_scheduler_cosine_annealing $optimizer "not_a_number"
# Error: Invalid T_max value

# Negative T_max
torch::lr_scheduler_cosine_annealing $optimizer -5
# Error: Required parameters missing or invalid

# Invalid eta_min type
torch::lr_scheduler_cosine_annealing $optimizer 100 "not_a_number"
# Error: Invalid eta_min value
```

## Compatibility

### Backward Compatibility
✅ **Fully Compatible**: All existing code using positional syntax continues to work unchanged.

```tcl
# Legacy code continues to work
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100]
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100 0.01]
```

### Migration Guide

#### From Positional to Named Parameters
```tcl
# Before (positional)
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100 0.001]

# After (named parameters)
set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 100 -etaMin 0.001]

# Or using camelCase
set scheduler [torch::lrSchedulerCosineAnnealing -optimizer $optimizer -tMax 100 -etaMin 0.001]
```

#### From torch::lr_scheduler_cosine
```tcl
# Old command (if migrating)
set scheduler [torch::lr_scheduler_cosine $optimizer 100 0.001]

# New standard command (equivalent behavior)
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100 0.001]

# Or with named parameters
set scheduler [torch::lr_scheduler_cosine_annealing -optimizer $optimizer -tMax 100 -etaMin 0.001]
```

## Related Commands
- `torch::lr_scheduler_step_update` - Updates the learning rate
- `torch::get_lr` - Gets current learning rate from optimizer
- `torch::lr_scheduler_cosine` - Alternative cosine scheduler
- `torch::lr_scheduler_exponential` - Exponential decay scheduler
- `torch::lr_scheduler_multi_step` - Multi-step scheduler
- `torch::lr_scheduler_cosine_annealing_warm_restarts` - Cosine annealing with warm restarts

## Performance Notes
- ✅ **Lightweight**: Minimal computational overhead
- ✅ **Memory Efficient**: Small memory footprint
- ✅ **Fast Updates**: O(1) learning rate calculations
- ✅ **Standard**: Follows PyTorch CosineAnnealingLR behavior

## Best Practices

### When to Use
- **Fine-tuning**: Excellent for fine-tuning pre-trained models
- **Final Training Phases**: Good for the last stages of training
- **Cyclic Training**: When you want periodic learning rate cycles
- **Standard Workflows**: When following established training recipes

### Parameter Selection
- **T_max**: Should match your training cycle length
- **eta_min**: Typically 1% to 10% of initial learning rate
- **Initial LR**: Set in optimizer, should be appropriate for your model

### Example Parameter Combinations
```tcl
# Conservative fine-tuning
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 50 0.001]

# Aggressive training
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 200 0.0]

# Standard deep learning
set scheduler [torch::lr_scheduler_cosine_annealing $optimizer 100 0.01]
``` 