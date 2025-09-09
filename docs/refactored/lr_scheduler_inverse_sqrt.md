# torch::lr_scheduler_inverse_sqrt

Creates an inverse square root learning rate scheduler that provides a warmup phase followed by inverse square root decay, commonly used in transformer training.

## Syntax

### Current (Positional - Backward Compatible)
```tcl
torch::lr_scheduler_inverse_sqrt optimizer warmup_steps ?decay_factor?
```

### New (Named Parameters)
```tcl
torch::lr_scheduler_inverse_sqrt -optimizer optimizer_handle -warmupSteps warmup_steps ?-decayFactor decay_factor?
```

### camelCase Alias
```tcl
torch::lrSchedulerInverseSqrt -optimizer optimizer_handle -warmupSteps warmup_steps ?-decayFactor decay_factor?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-optimizer` | string | *required* | Handle to the optimizer to schedule |
| `-warmupSteps` | integer | *required* | Number of warmup steps for linear increase |
| `-decayFactor` | double | 1.0 | Scaling factor for the inverse square root decay |

### Alternative Parameter Names
- `-warmup_steps` (snake_case alternative to `-warmupSteps`)
- `-decay_factor` (snake_case alternative to `-decayFactor`)

## Description

The inverse square root scheduler implements a learning rate schedule that:
1. **Warmup Phase**: Linearly increases learning rate from 0 to the initial learning rate over `warmup_steps`
2. **Decay Phase**: Applies inverse square root decay after warmup is complete

This scheduler is particularly popular in transformer models and was notably used in the "Attention Is All You Need" paper.

### Mathematical Formula

**Warmup Phase (step < warmup_steps):**
```
lr = initial_lr * (step / warmup_steps) * decay_factor
```

**Decay Phase (step ≥ warmup_steps):**
```
lr = initial_lr * decay_factor * sqrt(warmup_steps / step)
```

Where:
- `initial_lr` is the optimizer's initial learning rate
- `step` is the current training step (0-indexed)
- `warmup_steps` is the number of warmup steps
- `decay_factor` is the scaling factor

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create model parameters and optimizer
set param1 [torch::rand {512 256}]
set param2 [torch::rand {256 10}]
set optimizer [torch::optimizer_adam [list $param1 $param2] 0.001]

# Create inverse sqrt scheduler with 4000 warmup steps
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000]
```

### Named Parameter Syntax
```tcl
# Create inverse sqrt scheduler with named parameters
set scheduler [torch::lr_scheduler_inverse_sqrt \
    -optimizer $optimizer \
    -warmupSteps 8000 \
    -decayFactor 1.0]
```

### camelCase Syntax
```tcl
# Using the modern camelCase alias
set scheduler [torch::lrSchedulerInverseSqrt \
    -optimizer $optimizer \
    -warmupSteps 4000 \
    -decayFactor 0.8]
```

### Parameter Order Flexibility
```tcl
# Named parameters can be in any order
set scheduler [torch::lr_scheduler_inverse_sqrt \
    -decayFactor 1.2 \
    -warmupSteps 6000 \
    -optimizer $optimizer]
```

### Alternative Parameter Names
```tcl
# Using snake_case parameter names
set scheduler [torch::lr_scheduler_inverse_sqrt \
    -optimizer $optimizer \
    -warmup_steps 4000 \
    -decay_factor 0.9]
```

### Different Configurations
```tcl
# Short warmup for fine-tuning
set finetune_scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 1000 0.5]

# Long warmup for large models
set large_model_scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 16000 1.0]

# Custom scaling factor
set scaled_scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 2.0]
```

## Learning Rate Evolution

Here's how the learning rate evolves with different configurations (assuming initial_lr = 0.001):

### Warmup Phase (steps 0-4000, warmup_steps=4000, decay_factor=1.0)
| Step | Learning Rate | Formula |
|------|---------------|---------|
| 0    | 0.000         | 0.001 * (0/4000) * 1.0 |
| 1000 | 0.00025       | 0.001 * (1000/4000) * 1.0 |
| 2000 | 0.0005        | 0.001 * (2000/4000) * 1.0 |
| 4000 | 0.001         | 0.001 * (4000/4000) * 1.0 |

### Decay Phase (steps > 4000, warmup_steps=4000, decay_factor=1.0)
| Step | Learning Rate | Formula |
|------|---------------|---------|
| 4000 | 0.001         | 0.001 * 1.0 * sqrt(4000/4000) |
| 8000 | 0.000707      | 0.001 * 1.0 * sqrt(4000/8000) |
| 16000| 0.0005        | 0.001 * 1.0 * sqrt(4000/16000) |
| 32000| 0.000354      | 0.001 * 1.0 * sqrt(4000/32000) |

## Use Cases

### 1. Transformer Training
```tcl
# Standard transformer configuration (GPT, BERT, etc.)
set d_model 512
set warmup_steps [expr {$d_model * 4}]  ;# Common heuristic: 4 * d_model
set optimizer [torch::optimizer_adam $model_params 0.001]
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer $warmup_steps]
```

### 2. Large Language Models
```tcl
# Configuration for large models requiring longer warmup
set optimizer [torch::optimizer_adamw $model_params 0.0001]
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 10000 1.0]
```

### 3. Fine-tuning Pre-trained Models
```tcl
# Shorter warmup and scaled down learning rate for fine-tuning
set optimizer [torch::optimizer_adam $model_params 0.00005]
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 500 0.5]
```

### 4. Multi-stage Training
```tcl
# Different schedulers for different training phases
set pretrain_optimizer [torch::optimizer_adam $params 0.001]
set pretrain_scheduler [torch::lr_scheduler_inverse_sqrt $pretrain_optimizer 8000]

set finetune_optimizer [torch::optimizer_adam $params 0.0001]
set finetune_scheduler [torch::lr_scheduler_inverse_sqrt $finetune_optimizer 1000 0.3]
```

## Integration with Training Loop

```tcl
# Training loop with inverse sqrt scheduler
set optimizer [torch::optimizer_adam $model_params 0.001]
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000]

set global_step 0
for {set epoch 0} {$epoch < 100} {incr epoch} {
    foreach batch $training_data {
        # Forward pass, compute loss, backward pass
        # ...
        
        # Update parameters
        torch::optimizer_step $optimizer
        
        # Update learning rate every step
        torch::lr_scheduler_step $scheduler
        incr global_step
        
        # Optional: Log learning rate
        if {$global_step % 1000 == 0} {
            set current_lr [torch::get_lr $optimizer]
            puts "Step $global_step: LR = $current_lr"
        }
    }
}
```

## Error Handling

### Invalid Optimizer Handle
```tcl
# This will raise an error
catch {torch::lr_scheduler_inverse_sqrt "invalid_handle" 4000} error
puts "Error: $error"
# Output: Invalid optimizer name
```

### Missing Required Parameters
```tcl
# Missing warmup_steps in positional syntax
catch {torch::lr_scheduler_inverse_sqrt $optimizer} error
puts "Error: $error"
# Output: Usage: torch::lr_scheduler_inverse_sqrt optimizer warmup_steps ?decay_factor?

# Missing warmup_steps in named syntax
catch {torch::lr_scheduler_inverse_sqrt -optimizer $optimizer} error
puts "Error: $error"
# Output: Required parameters missing or invalid (optimizer handle and warmup_steps required...)
```

### Invalid Parameter Values
```tcl
# Negative warmup_steps
catch {torch::lr_scheduler_inverse_sqrt $optimizer -1000} error
puts "Error: $error"
# Output: Required parameters missing or invalid (warmup_steps must be positive...)

# Zero warmup_steps
catch {torch::lr_scheduler_inverse_sqrt $optimizer 0} error
puts "Error: $error"
# Output: Required parameters missing or invalid (warmup_steps must be positive...)

# Negative decay_factor
catch {torch::lr_scheduler_inverse_sqrt $optimizer 4000 -0.5} error
puts "Error: $error"
# Output: Required parameters missing or invalid (decay_factor must be positive...)
```

### Unknown Parameters
```tcl
# Invalid parameter name
catch {torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000 -invalid_param value} error
puts "Error: $error"
# Output: Unknown parameter: -invalid_param
```

## Return Value

Returns a string handle for the created scheduler (e.g., "inverse_sqrt_scheduler1", "inverse_sqrt_scheduler2", etc.) that can be used with other scheduler functions like `torch::lr_scheduler_step`.

## Comparison with Other Schedulers

| Scheduler Type | Warmup | Decay Pattern | Use Case |
|----------------|--------|---------------|----------|
| Inverse Sqrt | Yes | 1/√step | Transformers, language models |
| Exponential Decay | No | γ^step | Simple decay |
| Cosine Annealing | No | Cosine curve | Cyclic training |
| Step Decay | No | Step function | Milestone-based reduction |
| Noam | Yes | min(1/√step, step/warmup^1.5) | Original transformer paper |

## Best Practices

1. **Choose appropriate warmup steps**:
   - Small models: 1000-4000 steps
   - Large models: 8000-16000 steps
   - Rule of thumb: 4 × model_dimension

2. **Monitor training stability**:
   - Too short warmup can cause training instability
   - Too long warmup delays convergence

3. **Adjust decay_factor**:
   - decay_factor > 1.0: Higher peak learning rate
   - decay_factor < 1.0: Lower peak learning rate
   - decay_factor = 1.0: Standard behavior

4. **Coordinate with optimizer**:
   - Works well with Adam/AdamW
   - Consider lower initial learning rates for AdamW

5. **Log learning rates**:
   - Monitor LR schedule behavior
   - Ensure warmup completes as expected

## Mathematical Background

The inverse square root scheduler is based on the observation that optimal learning rates for attention mechanisms tend to follow an inverse square root relationship with training time. This was empirically discovered during transformer development.

### Theoretical Justification
- **Warmup prevents early optimization instability**
- **Inverse sqrt decay balances exploration vs exploitation**
- **Schedule adapts to changing gradient landscapes during training**

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_inverse_sqrt $optimizer 4000 1.0]

# New named parameter syntax (equivalent)
set scheduler [torch::lr_scheduler_inverse_sqrt -optimizer $optimizer -warmupSteps 4000 -decayFactor 1.0]

# Modern camelCase alias
set scheduler [torch::lrSchedulerInverseSqrt -optimizer $optimizer -warmupSteps 4000 -decayFactor 1.0]
```

### Advantages of Named Parameters
- **Clear parameter intent**: Explicitly shows warmup duration and scaling
- **Order independence**: Parameters can be specified in any order
- **Self-documenting**: Code is more readable and maintainable
- **Future-proof**: Easy to add new parameters without breaking changes

## See Also

- `torch::lr_scheduler_step` - Step the scheduler
- `torch::lr_scheduler_noam` - Noam scheduler (original transformer paper)
- `torch::lr_scheduler_exponential` - Simple exponential decay
- `torch::lr_scheduler_cosine` - Cosine annealing scheduler
- `torch::get_lr` - Get current learning rate
- `torch::optimizer_adam` - Adam optimizer (commonly used with this scheduler)
- `torch::optimizer_adamw` - AdamW optimizer (also commonly used) 