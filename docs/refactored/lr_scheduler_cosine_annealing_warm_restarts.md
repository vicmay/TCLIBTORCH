# torch::lr_scheduler_cosine_annealing_warm_restarts

Creates a Cosine Annealing learning rate scheduler with warm restarts.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::lr_scheduler_cosine_annealing_warm_restarts optimizer T_0 ?T_mult? ?eta_min?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_cosine_annealing_warm_restarts -optimizer OPTIMIZER -t0 T_0 ?-tMult T_MULT? ?-etaMin ETA_MIN?
```

### CamelCase Alias
```tcl
torch::lrSchedulerCosineAnnealingWarmRestarts -optimizer OPTIMIZER -t0 T_0 ?-tMult T_MULT? ?-etaMin ETA_MIN?
```

## Parameters

| Parameter | Aliases | Type | Required | Default | Description |
|-----------|---------|------|----------|---------|-------------|
| optimizer | N/A | string | Yes | N/A | Handle to the optimizer |
| T_0 | -t0, -T_0, -T0 | integer | Yes | N/A | Number of iterations for the first restart |
| T_mult | -tMult, -T_mult, -TMult | integer | No | 1 | Factor by which T_i increases after each restart |
| eta_min | -etaMin, -eta_min | double | No | 0.0 | Minimum learning rate |

## Description

The Cosine Annealing with Warm Restarts scheduler implements the learning rate schedule from the paper "SGDR: Stochastic Gradient Descent with Warm Restarts". It uses cosine annealing to decrease the learning rate, with periodic "warm restarts" that reset the learning rate to its maximum value.

### Warm Restart Behavior
- **First cycle**: T_0 iterations
- **Second cycle**: T_0 * T_mult iterations  
- **Third cycle**: T_0 * T_mult² iterations
- **And so on...**

The learning rate follows a cosine curve within each cycle, starting at the maximum and decreasing to eta_min, then restarting.

## Return Value

Returns a handle to the created learning rate scheduler that can be used with `torch::lr_scheduler_step_update`.

## Examples

### Example 1: Basic Usage with Positional Syntax
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_sgd $tensor 0.1]

# Create cosine annealing warm restarts scheduler with T_0=10
set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10]

# Step the scheduler
for {set i 0} {$i < 25} {incr i} {
    torch::lr_scheduler_step_update $scheduler
    puts "Step $i: LR = [torch::get_lr $optimizer]"
}
```

### Example 2: Named Parameters with T_mult
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_adam $tensor 0.01]

# Create scheduler with T_0=5, T_mult=2 (cycles: 5, 10, 20, ...)
set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts \
    -optimizer $optimizer \
    -t0 5 \
    -tMult 2 \
    -etaMin 0.001]

# Step through multiple cycles
for {set i 0} {$i < 40} {incr i} {
    torch::lr_scheduler_step_update $scheduler
    if {$i % 5 == 0} {
        puts "Step $i: LR = [torch::get_lr $optimizer]"
    }
}
```

### Example 3: CamelCase Alias
```tcl
# Using camelCase alias
set optimizer [torch::optimizer_sgd $tensor 0.05]
set scheduler [torch::lrSchedulerCosineAnnealingWarmRestarts \
    -optimizer $optimizer \
    -t0 8 \
    -tMult 3]
```

### Example 4: Training Loop with Warm Restarts
```tcl
# Setup
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} {4}]
set optimizer [torch::optimizer_sgd $tensor 0.1]
set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 2 0.01]

# Training loop
set epochs 30
for {set epoch 0} {$epoch < $epochs} {incr epoch} {
    # Training step would go here
    # ...
    
    # Update learning rate
    torch::lr_scheduler_step_update $scheduler
    
    set current_lr [torch::get_lr $optimizer]
    puts "Epoch $epoch: Learning Rate = $current_lr"
    
    # Check for warm restart (learning rate increases)
    if {$epoch > 0 && $current_lr > $prev_lr} {
        puts "  -> Warm restart detected!"
    }
    set prev_lr $current_lr
}
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts $optimizer 10 2 0.001]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_cosine_annealing_warm_restarts \
    -optimizer $optimizer \
    -t0 10 \
    -tMult 2 \
    -etaMin 0.001]
```

**Or using CamelCase:**
```tcl
set scheduler [torch::lrSchedulerCosineAnnealingWarmRestarts \
    -optimizer $optimizer \
    -t0 10 \
    -tMult 2 \
    -etaMin 0.001]
```

## Parameter Details

### T_0 (First Restart Period)
- Controls the number of iterations in the first restart cycle
- Must be a positive integer
- Typical values: 10-100 depending on your training setup

### T_mult (Restart Multiplier)
- Factor by which the restart period increases after each restart
- Must be ≥ 1
- T_mult = 1: All cycles have the same length
- T_mult > 1: Each cycle gets progressively longer

### eta_min (Minimum Learning Rate)
- The minimum learning rate value during each cycle
- Should be much smaller than the initial learning rate
- Prevents the learning rate from going to zero

## Warm Restart Schedule Examples

### T_mult = 1 (Fixed Cycle Length)
```
T_0 = 10, T_mult = 1
Cycles: [10] [10] [10] [10] ...
```

### T_mult = 2 (Doubling Cycles)
```
T_0 = 10, T_mult = 2  
Cycles: [10] [20] [40] [80] ...
```

### T_mult = 1.5 (Gradual Increase)
```
T_0 = 10, T_mult = 1.5
Cycles: [10] [15] [22] [33] ...
```

## Error Handling

The command will return an error in the following cases:

- **Invalid optimizer**: The optimizer handle doesn't exist
- **Missing T_0**: The T_0 parameter is required
- **Invalid T_0**: T_0 must be a positive integer
- **Invalid T_mult**: T_mult must be ≥ 1
- **Invalid eta_min**: eta_min must be a non-negative number
- **Missing parameter values**: Named parameters must have values

## Mathematical Background

The learning rate at step t within cycle i follows:

```
η_t = η_min + (η_max - η_min) * (1 + cos(π * T_cur / T_i)) / 2
```

Where:
- η_max is the maximum learning rate (initial LR)
- η_min is the minimum learning rate
- T_cur is the current step within the cycle
- T_i is the length of cycle i

## See Also

- `torch::lr_scheduler_cosine` - Simple cosine annealing without restarts
- `torch::lr_scheduler_cosine_annealing` - Standard cosine annealing
- `torch::lr_scheduler_step_update` - Update the learning rate
- `torch::get_lr` - Get current learning rate
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## Notes

- The scheduler automatically handles cycle transitions and warm restarts
- Learning rate updates occur when `torch::lr_scheduler_step_update` is called
- The first warm restart occurs after T_0 steps
- Subsequent restarts occur at increasing intervals based on T_mult
- The scheduler works with any PyTorch optimizer created through this extension 