# torch::lr_scheduler_lambda

Creates a Lambda learning rate scheduler that multiplies the learning rate by a constant factor.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::lr_scheduler_lambda optimizer ?multiplier?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_lambda -optimizer OPTIMIZER ?-multiplier MULTIPLIER?
```

### CamelCase Alias
```tcl
torch::lrSchedulerLambda -optimizer OPTIMIZER ?-multiplier MULTIPLIER?
```

## Parameters

| Parameter | Aliases | Type | Required | Default | Description |
|-----------|---------|------|----------|---------|-------------|
| optimizer | N/A | string | Yes | N/A | Handle to the optimizer |
| multiplier | -multiplier, -lambda | double | No | 1.0 | Factor to multiply the learning rate by |

## Description

The Lambda Learning Rate scheduler provides a simple way to scale the learning rate by a constant factor. This is useful for implementing custom learning rate schedules where you want to apply a fixed multiplier to the current learning rate at each scheduler step.

### Key Features

- **Constant Multiplier**: Applies the same factor at each step
- **Flexible Scaling**: Can increase (multiplier > 1.0) or decrease (multiplier < 1.0) learning rate
- **Simple Control**: Easy to understand and control
- **Custom Schedules**: Foundation for building more complex schedules

## Return Value

Returns a handle to the created learning rate scheduler that can be used with scheduler step operations.

## Examples

### Example 1: Basic Usage with Default Multiplier
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_sgd $tensor 0.01]

# Create lambda scheduler with default multiplier (1.0)
set scheduler [torch::lr_scheduler_lambda $optimizer]

puts "Lambda scheduler (identity): $scheduler"
```

### Example 2: Decay Schedule with Positional Syntax
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_adam $tensor 0.001]

# Create lambda scheduler that reduces LR by 5% each step
set scheduler [torch::lr_scheduler_lambda $optimizer 0.95]

puts "Lambda scheduler (decay): $scheduler"
```

### Example 3: Named Parameters
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_sgd $tensor 0.005]

# Create lambda scheduler with named parameters
set scheduler [torch::lr_scheduler_lambda \
    -optimizer $optimizer \
    -multiplier 0.90]

puts "Named parameter lambda scheduler: $scheduler"
```

### Example 4: CamelCase Alias with Lambda Parameter
```tcl
# Create optimizer
set optimizer [torch::optimizer_sgd $tensor 0.01]

# Using camelCase alias with -lambda parameter alias
set scheduler [torch::lrSchedulerLambda \
    -optimizer $optimizer \
    -lambda 0.98]

puts "CamelCase lambda scheduler: $scheduler"
```

### Example 5: Different Multiplier Scenarios
```tcl
# Setup
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} {4}]

# Decay schedule (reduce LR)
set opt1 [torch::optimizer_sgd $tensor 0.1]
set decay_sched [torch::lr_scheduler_lambda $opt1 0.9]

# Growth schedule (increase LR)
set opt2 [torch::optimizer_sgd $tensor 0.001]
set growth_sched [torch::lr_scheduler_lambda $opt2 1.1]

# Identity schedule (keep LR same)
set opt3 [torch::optimizer_sgd $tensor 0.01]
set identity_sched [torch::lr_scheduler_lambda $opt3 1.0]

# Aggressive decay
set opt4 [torch::optimizer_sgd $tensor 0.1]
set aggressive_sched [torch::lr_scheduler_lambda $opt4 0.5]

puts "Decay: $decay_sched"
puts "Growth: $growth_sched"
puts "Identity: $identity_sched"
puts "Aggressive: $aggressive_sched"
```

### Example 6: Training Loop Integration
```tcl
# Setup
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} {4}]
set optimizer [torch::optimizer_sgd $tensor 0.1]
set scheduler [torch::lr_scheduler_lambda $optimizer 0.99]

# Training loop with gradual decay
set epochs 50
for {set epoch 0} {$epoch < $epochs} {incr epoch} {
    # Training step would go here
    # ...
    
    # Apply lambda multiplier (hypothetical step function)
    # In practice, you'd call appropriate step function
    
    if {$epoch % 10 == 0} {
        puts "Epoch $epoch: Scheduler = $scheduler"
    }
}
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_lambda $optimizer 0.95]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_lambda \
    -optimizer $optimizer \
    -multiplier 0.95]
```

**Or using CamelCase:**
```tcl
set scheduler [torch::lrSchedulerLambda \
    -optimizer $optimizer \
    -lambda 0.95]
```

### Default Multiplier Migration
**Old:**
```tcl
set scheduler [torch::lr_scheduler_lambda $optimizer]
```

**New (all equivalent):**
```tcl
set scheduler [torch::lr_scheduler_lambda -optimizer $optimizer]
set scheduler [torch::lr_scheduler_lambda -optimizer $optimizer -multiplier 1.0]
set scheduler [torch::lrSchedulerLambda -optimizer $optimizer]
```

## Parameter Details

### multiplier (Lambda Factor)
- The constant factor to multiply the learning rate by
- **Default**: 1.0 (identity - no change)
- **Range**: Any real number (typically 0.1 to 2.0)
- **Effect**:
  - `< 1.0`: Decreases learning rate (decay)
  - `= 1.0`: Maintains learning rate (identity)
  - `> 1.0`: Increases learning rate (growth)

## Common Multiplier Values

### Decay Schedules
- **0.95**: Mild decay (5% reduction per step)
- **0.90**: Moderate decay (10% reduction per step)
- **0.80**: Aggressive decay (20% reduction per step)
- **0.50**: Very aggressive decay (50% reduction per step)

### Growth Schedules
- **1.05**: Mild growth (5% increase per step)
- **1.10**: Moderate growth (10% increase per step)
- **1.20**: Aggressive growth (20% increase per step)

### Special Values
- **1.0**: Identity (no change)
- **0.0**: Zero out learning rate
- **Negative values**: Reverse direction (typically not useful)

## Use Cases

### 1. Simple Exponential Decay
```tcl
# Decay LR by 2% each step
set scheduler [torch::lr_scheduler_lambda $optimizer 0.98]
```

### 2. Warming Up Learning Rate
```tcl
# Gradually increase LR during warmup
set scheduler [torch::lr_scheduler_lambda $optimizer 1.02]
```

### 3. Custom Schedule Building Block
```tcl
# Use as part of more complex scheduling logic
set scheduler [torch::lr_scheduler_lambda $optimizer $custom_multiplier]
```

### 4. Learning Rate Annealing
```tcl
# Different multipliers for different training phases
if {$epoch < 10} {
    set multiplier 1.0      ;# Maintain LR
} elseif {$epoch < 50} {
    set multiplier 0.95     ;# Gradual decay
} else {
    set multiplier 0.90     ;# Faster decay
}
set scheduler [torch::lr_scheduler_lambda $optimizer $multiplier]
```

## Mathematical Background

The lambda scheduler applies a simple multiplication:

```
lr_new = lr_current × multiplier
```

After n steps:
```
lr_n = lr_initial × multiplier^n
```

This creates an exponential schedule:
- **Decay** (multiplier < 1): Exponential decrease
- **Growth** (multiplier > 1): Exponential increase
- **Identity** (multiplier = 1): Constant learning rate

## Best Practices

1. **Start Conservative**: Use multipliers close to 1.0 initially
2. **Monitor Training**: Watch loss curves to ensure stability
3. **Consider Frequency**: How often you apply the multiplier matters
4. **Combine Wisely**: Can be part of more complex schedules
5. **Validate Range**: Ensure final LR stays in reasonable range

## Error Handling

The command will return an error in the following cases:

- **Invalid optimizer**: The optimizer handle doesn't exist
- **Invalid multiplier**: Non-numeric multiplier value
- **Missing parameters**: Required optimizer parameter missing
- **Parameter mismatch**: Incorrect parameter combinations

## Comparison with Other Schedulers

| Scheduler | Complexity | Use Case |
|-----------|------------|----------|
| Lambda | Simple | Constant factor scaling |
| Step | Medium | Discrete LR drops |
| Exponential | Medium | Smooth exponential decay |
| Cosine | High | Smooth annealing to minimum |
| Cyclic | High | Cyclical LR variations |

## Advanced Usage

### Dynamic Multiplier Selection
```tcl
# Adjust multiplier based on training progress
proc get_multiplier {epoch} {
    if {$epoch < 20} {
        return 1.0
    } elseif {$epoch < 100} {
        return 0.99
    } else {
        return 0.95
    }
}

set multiplier [get_multiplier $current_epoch]
set scheduler [torch::lr_scheduler_lambda $optimizer $multiplier]
```

### Combining with Other Schedulers
```tcl
# Use lambda as a fine-tuning step after other schedulers
set main_scheduler [torch::lr_scheduler_step $optimizer 10 0.5]
# ... apply main_scheduler ...

# Then apply fine adjustment
set fine_scheduler [torch::lr_scheduler_lambda $optimizer 0.98]
```

## See Also

- `torch::lr_scheduler_step` - Step-based LR scheduling
- `torch::lr_scheduler_exponential` - Exponential LR decay
- `torch::lr_scheduler_cosine` - Cosine annealing
- `torch::lr_scheduler_cyclic` - Cyclic LR schedule
- `torch::get_lr` - Get current learning rate
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## Notes

- The scheduler stores the multiplier but doesn't automatically apply it
- You need to implement the stepping mechanism to actually update learning rates
- The multiplier can be any real number, but typical values are between 0.1 and 2.0
- Negative multipliers are technically allowed but usually not useful
- Zero multiplier will effectively stop learning (LR becomes 0)
- The scheduler works with any PyTorch optimizer created through this extension 