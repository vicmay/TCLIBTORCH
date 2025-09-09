# torch::lr_scheduler_one_cycle

Creates a one cycle learning rate scheduler that implements the learning rate policy described in "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates".

## Syntax

### Positional Syntax
```tcl
torch::lr_scheduler_one_cycle optimizer max_lr total_steps [pct_start] [anneal_strategy] [div_factor]
```

### camelCase Alias
```tcl
torch::lrSchedulerOneCycle ...
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| optimizer | string | Yes | - | Handle to the optimizer |
| max_lr | float | Yes | - | Maximum learning rate in the cycle |
| total_steps | integer | Yes | - | Total number of training steps |
| pct_start | float | No | 0.3 | Percentage of cycle spent increasing learning rate |
| anneal_strategy | string | No | "cos" | Annealing strategy: "cos" or "linear" |
| div_factor | float | No | 25.0 | Factor to divide max_lr to get initial learning rate |

## Returns
Returns a handle to the created one cycle learning rate scheduler.

## Description

The one cycle scheduler implements the learning rate policy that:
1. **Warmup Phase**: Increases learning rate linearly from `max_lr/div_factor` to `max_lr` over `pct_start * total_steps` steps
2. **Annealing Phase**: Decreases learning rate from `max_lr` to `max_lr/div_factor` using the specified annealing strategy over the remaining steps

The initial learning rate is calculated as: `initial_lr = max_lr / div_factor`

This scheduler is designed to achieve super-convergence by using large learning rates and cycling through different learning rate values.

## Examples

### Basic Usage
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.1]

# Create one cycle scheduler with default settings
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.1 1000]

# Step the scheduler
torch::lr_scheduler_step_update $scheduler
```

### Custom Warmup Percentage
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_adam $weights 0.001]

# 20% of cycle for warmup (instead of default 30%)
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.01 2000 0.2]

# Step multiple times
for {set i 0} {$i < 100} {incr i} {
    torch::lr_scheduler_step_update $scheduler
}
```

### Linear Annealing Strategy
```tcl
# Use linear annealing instead of cosine
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.05 1500 0.3 "linear"]

# Step the scheduler
torch::lr_scheduler_step_update $scheduler
```

### Custom Division Factor
```tcl
# Use smaller division factor (higher initial learning rate)
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.1 1000 0.3 "cos" 10.0]
# Initial LR will be 0.1/10.0 = 0.01

# Use larger division factor (lower initial learning rate)  
set scheduler2 [torch::lr_scheduler_one_cycle $optimizer 0.1 1000 0.3 "cos" 50.0]
# Initial LR will be 0.1/50.0 = 0.002
```

### Complete Training Loop Example
```tcl
# Setup
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.001]
set total_epochs 100
set steps_per_epoch 50
set total_steps [expr $total_epochs * $steps_per_epoch]

# Create one cycle scheduler for entire training
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.1 $total_steps 0.25 "cos" 20.0]

# Training loop
for {set epoch 0} {$epoch < $total_epochs} {incr epoch} {
    for {set step 0} {$step < $steps_per_epoch} {incr step} {
        # Training step here...
        
        # Update learning rate
        torch::lr_scheduler_step_update $scheduler
    }
}
```

### Using camelCase Alias
```tcl
set scheduler [torch::lrSchedulerOneCycle $optimizer 0.05 2000 0.25 "linear" 30.0]
```

## Annealing Strategies

### Cosine Annealing ("cos")
- Default strategy
- Smooth cosine curve for learning rate decrease
- Gradual decay that starts fast and slows down

### Linear Annealing ("linear")
- Linear decrease in learning rate
- Constant rate of decay
- Sometimes preferred for certain training scenarios

## Common Use Cases

### Super-Convergence Training
```tcl
# High learning rate with short training
set scheduler [torch::lr_scheduler_one_cycle $optimizer 1.0 500 0.1 "cos" 25.0]
```

### Fine-tuning
```tcl
# Lower learning rates for fine-tuning
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.001 2000 0.5 "linear" 10.0]
```

### Image Classification
```tcl
# Typical settings for image classification
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.1 10000 0.3 "cos" 25.0]
```

### Natural Language Processing
```tcl
# Lower learning rates often used in NLP
set scheduler [torch::lr_scheduler_one_cycle $optimizer 0.01 5000 0.25 "linear" 20.0]
```

## Parameter Guidelines

### max_lr Selection
- Start with 10x higher than typical learning rates
- Use learning rate range tests to find optimal values
- Common range: 0.01 to 1.0

### total_steps Calculation
```tcl
set total_steps [expr $num_epochs * $steps_per_epoch]
```

### pct_start Guidelines
- **0.1-0.2**: Short warmup, longer annealing
- **0.3**: Balanced (default)
- **0.4-0.5**: Longer warmup, shorter annealing

### div_factor Guidelines
- **10-20**: Higher initial learning rate
- **25**: Balanced (default) 
- **50-100**: Lower initial learning rate

## Mathematical Formula

The learning rate at step `t` is calculated as:

**Warmup Phase** (t ≤ pct_start × total_steps):
```
lr(t) = initial_lr + (max_lr - initial_lr) × (t / warmup_steps)
```

**Annealing Phase** (t > pct_start × total_steps):

For cosine annealing:
```
lr(t) = initial_lr + (max_lr - initial_lr) × (1 + cos(π × progress)) / 2
```

For linear annealing:
```
lr(t) = max_lr - (max_lr - initial_lr) × progress
```

Where:
- `initial_lr = max_lr / div_factor`
- `warmup_steps = pct_start × total_steps`
- `progress = (t - warmup_steps) / (total_steps - warmup_steps)`

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing required parameters
torch::lr_scheduler_one_cycle $optimizer
# Error: wrong # args: should be "torch::lr_scheduler_one_cycle optimizer max_lr total_steps ?pct_start? ?anneal_strategy? ?div_factor?"

# Invalid optimizer
torch::lr_scheduler_one_cycle "invalid_handle" 0.1 1000
# Error: Invalid optimizer handle

# Invalid numeric values
torch::lr_scheduler_one_cycle $optimizer "invalid" 1000
# Error: expected floating-point but got "invalid"
```

## Performance Considerations

- Scheduler computation is O(1) per step
- No significant performance impact during training
- Memory usage is minimal
- Suitable for large-scale training

## Comparison with Other Schedulers

| Scheduler | Use Case | Learning Rate Pattern |
|-----------|----------|----------------------|
| OneCycle | Super-convergence, fast training | Triangle wave with peak |
| StepLR | Traditional training | Step decreases |
| CosineAnnealing | Smooth decay | Cosine curve |
| ExponentialLR | Exponential decay | Exponential curve |

## See Also

- `torch::lr_scheduler_cosine_annealing` - Cosine annealing scheduler
- `torch::lr_scheduler_step` - Step-based scheduler
- `torch::lr_scheduler_exponential` - Exponential decay scheduler
- `torch::lr_scheduler_step_update` - Update scheduler step
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## References

- Smith, L.N. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." arXiv:1708.07120
- Smith, L.N. "Cyclical Learning Rates for Training Neural Networks." arXiv:1506.01186 