# torch::lr_scheduler_cyclic

Creates a Cyclic learning rate scheduler that cycles the learning rate between minimum and maximum values.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::lr_scheduler_cyclic optimizer base_lr max_lr ?step_size? ?mode?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::lr_scheduler_cyclic -optimizer OPTIMIZER -baseLr BASE_LR -maxLr MAX_LR ?-stepSize STEP_SIZE? ?-mode MODE?
```

### CamelCase Alias
```tcl
torch::lrSchedulerCyclic -optimizer OPTIMIZER -baseLr BASE_LR -maxLr MAX_LR ?-stepSize STEP_SIZE? ?-mode MODE?
```

## Parameters

| Parameter | Aliases | Type | Required | Default | Description |
|-----------|---------|------|----------|---------|-------------|
| optimizer | N/A | string | Yes | N/A | Handle to the optimizer |
| base_lr | -baseLr, -base_lr | double | Yes | N/A | Minimum learning rate in the cycle |
| max_lr | -maxLr, -max_lr | double | Yes | N/A | Maximum learning rate in the cycle |
| step_size | -stepSize, -step_size | integer | No | 2000 | Half the cycle length (steps for LR to go from base to max) |
| mode | -mode | string | No | "triangular" | Cycling mode: "triangular", "triangular2", or "exp_range" |

## Description

The Cyclic Learning Rate (CLR) scheduler implements the learning rate schedule from the paper "Cyclical Learning Rates for Training Neural Networks". It cycles the learning rate between a minimum (`base_lr`) and maximum (`max_lr`) value, allowing for faster convergence and better performance in many scenarios.

### Cycling Modes

1. **triangular**: Basic triangular cycle (linear increase/decrease)
2. **triangular2**: Triangular cycle where the amplitude decreases by half after each cycle
3. **exp_range**: Exponential range cycle where the amplitude scales exponentially

### Cycle Structure

- **Cycle length**: 2 × step_size
- **First half**: Learning rate increases from base_lr to max_lr
- **Second half**: Learning rate decreases from max_lr to base_lr

## Return Value

Returns a handle to the created learning rate scheduler that can be used with other scheduler operations.

## Examples

### Example 1: Basic Usage with Positional Syntax
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_sgd $tensor 0.01]

# Create cyclic scheduler: LR cycles between 0.001 and 0.1
set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1]

# The learning rate will cycle every 4000 steps (2 × 2000)
puts "Created cyclic scheduler: $scheduler"
```

### Example 2: Named Parameters with Custom Settings
```tcl
# Create optimizer
set tensor [torch::tensor_create {1.0 2.0 3.0} {3}]
set optimizer [torch::optimizer_adam $tensor 0.01]

# Create cyclic scheduler with custom parameters
set scheduler [torch::lr_scheduler_cyclic \
    -optimizer $optimizer \
    -baseLr 0.0001 \
    -maxLr 0.01 \
    -stepSize 1000 \
    -mode triangular2]

puts "Cyclic scheduler with triangular2 mode: $scheduler"
```

### Example 3: CamelCase Alias
```tcl
# Using camelCase alias
set optimizer [torch::optimizer_sgd $tensor 0.05]
set scheduler [torch::lrSchedulerCyclic \
    -optimizer $optimizer \
    -baseLr 0.005 \
    -maxLr 0.05 \
    -stepSize 500]
```

### Example 4: Different Cycling Modes
```tcl
# Setup
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} {4}]

# Triangular mode (default)
set opt1 [torch::optimizer_sgd $tensor 0.01]
set sched1 [torch::lr_scheduler_cyclic $opt1 0.001 0.1 1000 triangular]

# Triangular2 mode (amplitude decreases each cycle)
set opt2 [torch::optimizer_sgd $tensor 0.01]
set sched2 [torch::lr_scheduler_cyclic $opt2 0.001 0.1 1000 triangular2]

# Exponential range mode
set opt3 [torch::optimizer_sgd $tensor 0.01]
set sched3 [torch::lr_scheduler_cyclic $opt3 0.001 0.1 1000 exp_range]

puts "Triangular: $sched1"
puts "Triangular2: $sched2"
puts "Exp Range: $sched3"
```

### Example 5: Training Loop Integration
```tcl
# Setup
set tensor [torch::tensor_create {1.0 2.0 3.0 4.0} {4}]
set optimizer [torch::optimizer_sgd $tensor 0.01]
set scheduler [torch::lr_scheduler_cyclic $optimizer 0.0001 0.01 500]

# Training loop
set steps 2000
for {set step 0} {$step < $steps} {incr step} {
    # Training step would go here
    # ...
    
    # Update learning rate (if using step-based scheduler)
    if {$step % 10 == 0} {
        puts "Step $step: Current LR = [torch::get_lr $optimizer]"
    }
}
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_cyclic $optimizer 0.001 0.1 1500 triangular2]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_cyclic \
    -optimizer $optimizer \
    -baseLr 0.001 \
    -maxLr 0.1 \
    -stepSize 1500 \
    -mode triangular2]
```

**Or using CamelCase:**
```tcl
set scheduler [torch::lrSchedulerCyclic \
    -optimizer $optimizer \
    -baseLr 0.001 \
    -maxLr 0.1 \
    -stepSize 1500 \
    -mode triangular2]
```

## Parameter Details

### base_lr (Minimum Learning Rate)
- The minimum learning rate value in the cycle
- Must be positive and less than max_lr
- Typical values: 1e-5 to 1e-2

### max_lr (Maximum Learning Rate)
- The maximum learning rate value in the cycle
- Must be greater than base_lr
- Should be chosen carefully to avoid training instability
- Typical values: 1e-3 to 1e-1

### step_size (Half Cycle Length)
- Number of steps for learning rate to go from base to max (or max to base)
- Total cycle length = 2 × step_size
- Should be set based on your training schedule
- Typical values: 1000-5000 steps

### mode (Cycling Pattern)
- **triangular**: Simple linear increase/decrease
- **triangular2**: Amplitude halves after each cycle
- **exp_range**: Exponential scaling of amplitude

## Cycling Mode Details

### Triangular Mode
```
LR
^
|    /\      /\      /\
|   /  \    /  \    /  \
|  /    \  /    \  /    \
| /      \/      \/      \
+------------------------> Steps
base_lr                  
```

### Triangular2 Mode
```
LR
^
|    /\
|   /  \    /\
|  /    \  /  \  /\
| /      \/    \/  \
+-------------------> Steps
base_lr
```

### Exp_range Mode
Similar to triangular but with exponential amplitude scaling.

## Learning Rate Finding

The cyclic scheduler is particularly useful for finding optimal learning rates:

```tcl
# Start with a wide range to explore
set scheduler [torch::lr_scheduler_cyclic $optimizer 1e-6 1e-1 1000]

# Monitor loss during training to find the optimal range
# Then use a narrower range for actual training
```

## Error Handling

The command will return an error in the following cases:

- **Invalid optimizer**: The optimizer handle doesn't exist
- **Missing required parameters**: base_lr and max_lr are required
- **Invalid learning rates**: base_lr must be positive, max_lr must be greater than base_lr
- **Invalid step_size**: Must be a positive integer
- **Invalid mode**: Must be one of "triangular", "triangular2", or "exp_range"
- **Missing parameter values**: Named parameters must have values

## Mathematical Background

### Triangular Mode
```
cycle = floor(1 + step / (2 * step_size))
x = abs(step / step_size - 2 * cycle + 1)
lr = base_lr + (max_lr - base_lr) * max(0, 1 - x)
```

### Triangular2 Mode
```
lr = base_lr + (max_lr - base_lr) * max(0, 1 - x) / (2^(cycle - 1))
```

### Exp_range Mode
```
lr = base_lr + (max_lr - base_lr) * max(0, 1 - x) * gamma^step
```

## Benefits of Cyclic Learning Rates

1. **Faster convergence**: Can escape local minima
2. **Better generalization**: Regularization effect
3. **Learning rate range finding**: Helps identify optimal LR ranges
4. **Reduced hyperparameter tuning**: Less need for manual LR scheduling

## Best Practices

1. **Start wide**: Begin with a wide LR range to explore
2. **Monitor training**: Watch loss curves to adjust ranges
3. **Consider your optimizer**: Different optimizers may need different ranges
4. **Adjust step_size**: Should relate to your dataset size and batch size
5. **Use with care**: Very high learning rates can destabilize training

## See Also

- `torch::lr_scheduler_step` - Step-based LR scheduling
- `torch::lr_scheduler_exponential` - Exponential LR decay
- `torch::lr_scheduler_cosine` - Cosine annealing
- `torch::lr_scheduler_one_cycle` - One cycle policy
- `torch::get_lr` - Get current learning rate
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## Notes

- The scheduler stores parameters but doesn't automatically update learning rates
- Use with appropriate step/epoch-based update mechanisms
- Cycle length should be chosen based on your training schedule
- The scheduler works with any PyTorch optimizer created through this extension
- Consider starting with triangular mode before experimenting with other modes 