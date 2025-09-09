# torch::lr_scheduler_constant_with_warmup

Creates a constant learning rate scheduler with an initial warm-up phase. This scheduler linearly increases the learning rate from 0 to the initial learning rate during the warm-up period, then maintains a constant learning rate afterwards.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_constant_with_warmup -optimizer optimizer_handle -numWarmupSteps warmup_steps [-lastEpoch last_epoch]
torch::lrSchedulerConstantWithWarmup -optimizer optimizer_handle -numWarmupSteps warmup_steps [-lastEpoch last_epoch]
```

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_constant_with_warmup optimizer_handle warmup_steps [last_epoch]
torch::lrSchedulerConstantWithWarmup optimizer_handle warmup_steps [last_epoch]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| optimizer | string | required | Handle to the optimizer to schedule |
| numWarmupSteps | integer | required | Number of warmup steps for linear increase |
| lastEpoch | integer | -1 | Index of the last epoch when resuming training |

### Alternative Parameter Names
- `-num_warmup_steps` (snake_case alternative to `-numWarmupSteps`)
- `-last_epoch` (snake_case alternative to `-lastEpoch`)

## Output

Returns a scheduler handle for the created constant with warmup scheduler.

**Scheduler Handle Format**: `constant_warmup_scheduler_*`

## Description

The constant with warmup scheduler implements a two-phase learning rate schedule:

1. **Warmup Phase** (steps 0 to numWarmupSteps): Linearly increases learning rate from 0 to the initial learning rate
2. **Constant Phase** (steps > numWarmupSteps): Maintains the initial learning rate

This schedule is particularly useful for:
- Stabilizing training in the early phases
- Preventing gradient explosions with large learning rates
- Transformer models and other architectures sensitive to initial learning rate

### Mathematical Formula

**Warmup Phase (step ≤ numWarmupSteps):**
```
lr = initial_lr × (step / numWarmupSteps)
```

**Constant Phase (step > numWarmupSteps):**
```
lr = initial_lr
```

Where:
- `initial_lr` is the optimizer's initial learning rate
- `step` is the current training step (0-indexed)
- `numWarmupSteps` is the number of warmup steps

## Examples

### Basic Usage with Named Parameters
```tcl
# Create an optimizer
set param1 [torch::ones -shape {10 10} -dtype float32 -requires_grad 1]
set param2 [torch::ones -shape {5} -dtype float32 -requires_grad 1]
set optimizer [torch::optimizer_sgd [list $param1 $param2] 0.1]

# Create scheduler with 100 warmup steps
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 100]
```

### Basic Usage with Positional Syntax
```tcl
# Create optimizer and scheduler using positional syntax
set optimizer [torch::optimizer_adam [list $param1] 0.001]
set scheduler [torch::lr_scheduler_constant_with_warmup $optimizer 50]
```

### With Last Epoch (Resuming Training)
```tcl
# Resume training from epoch 25
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 100 \
    -lastEpoch 25]
```

### Snake Case Parameter Names
```tcl
# Using snake_case parameter names
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -num_warmup_steps 200 \
    -last_epoch 10]
```

### Using camelCase Alias
```tcl
# Using the camelCase alias
set scheduler [torch::lrSchedulerConstantWithWarmup \
    -optimizer $optimizer \
    -numWarmupSteps 150]
```

### Different Warmup Periods
```tcl
# Short warmup (10 steps)
set short_warmup [torch::lr_scheduler_constant_with_warmup $opt1 10]

# Medium warmup (100 steps)  
set medium_warmup [torch::lr_scheduler_constant_with_warmup $opt2 100]

# Long warmup (1000 steps)
set long_warmup [torch::lr_scheduler_constant_with_warmup $opt3 1000]
```

### Integration with Different Optimizers
```tcl
# With SGD optimizer
set sgd_params [list $param1 $param2]
set sgd_opt [torch::optimizer_sgd $sgd_params 0.1]
set sgd_scheduler [torch::lr_scheduler_constant_with_warmup $sgd_opt 100]

# With Adam optimizer
set adam_params [list $param3 $param4]
set adam_opt [torch::optimizer_adam $adam_params 0.001]
set adam_scheduler [torch::lr_scheduler_constant_with_warmup $adam_opt 50]
```

## Use Cases

### 1. Transformer Training
```tcl
# Common pattern for transformer models
set model [create_transformer_model]
set params [torch::parameters $model]
set optimizer [torch::optimizer_adam $params 0.0001]

# Warmup for 4000 steps (common in transformer training)
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 4000]
```

### 2. Large Learning Rate Stabilization
```tcl
# When using large learning rates, warmup prevents instability
set optimizer [torch::optimizer_sgd $params 1.0]  # Large LR
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 500]  # Gradual increase to 1.0
```

### 3. Fine-tuning Pretrained Models
```tcl
# Gentle warmup for fine-tuning
set pretrained_params [get_pretrained_parameters]
set optimizer [torch::optimizer_adam $pretrained_params 0.00001]
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 100]
```

### 4. Training Resumption
```tcl
# Resume training from a checkpoint
set saved_epoch 150
set optimizer [load_optimizer_checkpoint "checkpoint.pth"]
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 100 \
    -lastEpoch $saved_epoch]
```

### 5. Zero Warmup (Immediate Constant Rate)
```tcl
# No warmup, immediate constant learning rate
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 0]
```

## Learning Rate Schedule Visualization

The learning rate progression looks like this:

```
LR
 |
 |     ┌─────────────────────────── Constant Phase
 |    /│
 |   / │
 |  /  │
 | /   │
 |/    │
 └─────┼──────────────────────────── Steps
 0   warmup                    
```

### Example Schedule (initial_lr=0.1, warmup_steps=100):
- Step 0: lr = 0.0
- Step 25: lr = 0.025
- Step 50: lr = 0.05
- Step 75: lr = 0.075
- Step 100: lr = 0.1
- Step 200: lr = 0.1 (constant)
- Step 1000: lr = 0.1 (constant)

## Integration with Training Loop

```tcl
# Complete training example
set model [create_model]
set params [torch::parameters $model]
set optimizer [torch::optimizer_adam $params 0.001]
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 1000]

set total_steps 10000
for {set step 0} {$step < $total_steps} {incr step} {
    # Training step
    set loss [train_one_step $model $optimizer]
    
    # Update learning rate (call after each step during warmup)
    if {$step < 1000} {
        torch::lr_scheduler_step $scheduler
    }
    
    # Log progress
    if {$step % 100 == 0} {
        set current_lr [torch::get_lr $scheduler]
        puts "Step $step: Loss = $loss, LR = $current_lr"
    }
}
```

## Error Handling

The command will raise an error for:
- Missing required parameters (`optimizer` and `numWarmupSteps`)
- Invalid optimizer handle (non-existent optimizer)
- Invalid parameter types (non-integer warmup steps, etc.)
- Negative warmup steps
- Unknown parameter names
- Missing parameter values in named syntax

### Error Examples
```tcl
# Error: Missing required parameters
catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt} error
# Error: "Required parameters missing or invalid"

# Error: Invalid optimizer
catch {torch::lr_scheduler_constant_with_warmup -optimizer "invalid" -numWarmupSteps 100} error
# Error: "Invalid optimizer name"

# Error: Negative warmup steps
catch {torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps -10} error
# Error: "Required parameters missing or invalid"
```

## Performance Notes

- **Memory usage**: Minimal overhead, stores only scheduler state
- **Computation**: Very lightweight, simple linear interpolation during warmup
- **Warmup overhead**: Only affects first `numWarmupSteps` steps
- **Scaling**: Performance independent of model size

## Best Practices

### 1. Warmup Duration
```tcl
# Rule of thumb: warmup for 1-10% of total training steps
set total_steps 100000
set warmup_steps [expr {$total_steps / 20}]  # 5% warmup
set scheduler [torch::lr_scheduler_constant_with_warmup $opt $warmup_steps]
```

### 2. Learning Rate Selection
```tcl
# Start with smaller LR when using longer warmup
set base_lr 0.001
set warmup_steps 5000
set optimizer [torch::optimizer_adam $params $base_lr]
set scheduler [torch::lr_scheduler_constant_with_warmup $opt $warmup_steps]
```

### 3. Combining with Other Schedulers
```tcl
# Use constant warmup followed by decay scheduler
# (This would require manual implementation or chaining)
set warmup_scheduler [torch::lr_scheduler_constant_with_warmup $opt 1000]
# ... run warmup phase
set decay_scheduler [torch::lr_scheduler_exponential $opt 0.95]
```

## Migration Guide

### From Legacy Syntax
```tcl
# Old positional syntax
set scheduler [torch::lr_scheduler_constant_with_warmup $optimizer 100 10]

# New named parameter syntax
set scheduler [torch::lr_scheduler_constant_with_warmup \
    -optimizer $optimizer \
    -numWarmupSteps 100 \
    -lastEpoch 10]
```

### Parameter Mapping
| Old Position | New Parameter | Notes |
|--------------|---------------|--------|
| 1 | -optimizer | Required |
| 2 | -numWarmupSteps | Required |
| 3 | -lastEpoch | Optional, default -1 |

### New Features in Named Syntax
```tcl
# Old: only positional parameters supported
torch::lr_scheduler_constant_with_warmup $opt 100

# New: both parameter naming styles supported
torch::lr_scheduler_constant_with_warmup -optimizer $opt -numWarmupSteps 100
torch::lr_scheduler_constant_with_warmup -optimizer $opt -num_warmup_steps 100

# New: camelCase alias available
torch::lrSchedulerConstantWithWarmup -optimizer $opt -numWarmupSteps 100

# New: flexible parameter order
torch::lr_scheduler_constant_with_warmup \
    -lastEpoch 5 \
    -optimizer $opt \
    -numWarmupSteps 100
```

## Comparison with Other Warmup Schedulers

### vs. Linear with Warmup
```tcl
# Constant with warmup: LR stays constant after warmup
set const_scheduler [torch::lr_scheduler_constant_with_warmup $opt 100]

# Linear with warmup: LR decays to 0 after warmup
set linear_scheduler [torch::lr_scheduler_linear_with_warmup $opt 100 1000]
```

### vs. Cosine with Warmup
```tcl
# Constant: Simple warmup then constant
set const_scheduler [torch::lr_scheduler_constant_with_warmup $opt 100]

# Cosine: Warmup then cosine decay (would need custom implementation)
```

## See Also

- `torch::lr_scheduler_linear_with_warmup` - Linear decay after warmup
- `torch::lr_scheduler_step` - Step decay scheduler
- `torch::lr_scheduler_exponential` - Exponential decay scheduler
- `torch::lr_scheduler_cosine_annealing` - Cosine annealing scheduler
- `torch::lr_scheduler_noam` - Noam scheduler (inverse square root with warmup)
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer 