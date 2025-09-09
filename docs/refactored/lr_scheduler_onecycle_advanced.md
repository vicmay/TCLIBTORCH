# torch::lr_scheduler_onecycle_advanced

Creates an advanced one cycle learning rate scheduler with extended parameters for fine-grained control over the learning rate policy.

## Syntax

### Positional Syntax
```tcl
torch::lr_scheduler_onecycle_advanced optimizer max_lr total_steps [pct_start] [anneal_strategy] [div_factor] [final_div_factor]
```

### camelCase Alias
```tcl
torch::lrSchedulerOnecycleAdvanced ...
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| optimizer | string | Yes | - | Handle to the optimizer |
| max_lr | float | Yes | - | Maximum learning rate in the cycle |
| total_steps | integer | Yes | - | Total number of training steps |
| pct_start | float | No | 0.3 | Percentage of cycle spent increasing learning rate |
| anneal_strategy | string | No | "cos" | Annealing strategy (not fully implemented in current version) |
| div_factor | float | No | 25.0 | Factor to divide max_lr to get initial learning rate |
| final_div_factor | float | No | 10000.0 | Factor for final learning rate computation |

## Returns
Returns a handle to the created advanced one cycle learning rate scheduler.

## Description

The advanced one cycle scheduler extends the basic one cycle policy with additional parameters for more sophisticated learning rate control:

1. **Extended Warmup Control**: Fine-tuned control over the warmup phase
2. **Advanced Annealing**: Additional parameters for the annealing phase
3. **Final Learning Rate Control**: Separate control over the final learning rate via `final_div_factor`

The scheduler follows a similar pattern to the basic one cycle but with enhanced flexibility:
- **Warmup Phase**: Increases learning rate from `max_lr/div_factor` to `max_lr`
- **Annealing Phase**: Decreases learning rate with advanced controls
- **Final Phase**: Uses `final_div_factor` for final learning rate determination

## Examples

### Basic Advanced Usage
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.1]

# Create advanced one cycle scheduler with default settings
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 1000]

# Step the scheduler
torch::lr_scheduler_step_update $scheduler
```

### Custom Warmup and Annealing
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_adam $weights 0.001]

# Extended warmup period (40% of cycle) with custom annealing
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.01 2000 0.4 "linear"]

# Step multiple times
for {set i 0} {$i < 100} {incr i} {
    torch::lr_scheduler_step_update $scheduler
}
```

### Advanced Parameter Control
```tcl
# Full parameter specification
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.05 1500 0.25 "cos" 20.0 5000.0]
# Initial LR: 0.05/20.0 = 0.0025
# Final LR controlled by final_div_factor
```

### High-Performance Training Setup
```tcl
# Setup for aggressive training with high learning rates
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.001]
set total_epochs 50
set steps_per_epoch 200
set total_steps [expr $total_epochs * $steps_per_epoch]

# Aggressive schedule with short warmup, high max LR
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 1.0 $total_steps 0.1 "cos" 50.0 10000.0]

# Training loop
for {set epoch 0} {$epoch < $total_epochs} {incr epoch} {
    for {set step 0} {$step < $steps_per_epoch} {incr step} {
        # Training step here...
        
        # Update learning rate
        torch::lr_scheduler_step_update $scheduler
    }
}
```

### Fine-tuning Configuration
```tcl
# Conservative settings for fine-tuning
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.001 5000 0.5 "linear" 10.0 1000.0]
# Longer warmup (50%), lower learning rates, gentler final decay
```

### Using camelCase Alias
```tcl
set scheduler [torch::lrSchedulerOnecycleAdvanced $optimizer 0.05 2000 0.3 "cos" 15.0 2000.0]
```

## Advanced Features

### Extended Division Factor Control
The advanced scheduler provides two division factors:
- **div_factor**: Controls initial learning rate (`initial_lr = max_lr / div_factor`)
- **final_div_factor**: Controls final learning rate behavior

### Enhanced Phase Control
- **Flexible pct_start**: Can range from 0.01 to 0.99 for very short or very long warmup phases
- **Advanced annealing**: Additional parameters for sophisticated decay patterns

### Multiple Annealing Strategies
While the current implementation primarily uses the specified strategy, the framework supports:
- **"cos"**: Cosine annealing (smooth, curved decay)
- **"linear"**: Linear annealing (constant rate decay)

## Parameter Guidelines

### max_lr Selection for Advanced Training
- **Research/Experimentation**: 0.1 - 1.0
- **Production Training**: 0.01 - 0.1  
- **Fine-tuning**: 0.001 - 0.01
- **Transfer Learning**: 0.0001 - 0.001

### pct_start Advanced Usage
- **0.05-0.15**: Very short warmup for pre-trained models
- **0.2-0.3**: Standard warmup for training from scratch
- **0.4-0.6**: Extended warmup for unstable training
- **0.7-0.9**: Very long warmup for extremely sensitive models

### div_factor Guidelines
- **5-10**: Very high initial learning rate (aggressive training)
- **15-25**: Balanced approach (recommended)
- **30-50**: Conservative initial learning rate
- **100+**: Very low initial learning rate (fine-tuning)

### final_div_factor Guidelines
- **100-1000**: Gentle final decay
- **1000-10000**: Standard final decay (recommended)
- **10000+**: Aggressive final decay for convergence

## Mathematical Framework

The advanced scheduler computes learning rate as:

**Phase 1 - Warmup** (step ≤ pct_start × total_steps):
```
lr(step) = initial_lr + (max_lr - initial_lr) × (step / warmup_steps)
```

**Phase 2 - Annealing** (step > pct_start × total_steps):
```
lr(step) = max_lr × annealing_function(progress) × final_factor
```

Where:
- `initial_lr = max_lr / div_factor`
- `warmup_steps = pct_start × total_steps`
- `progress = (step - warmup_steps) / (total_steps - warmup_steps)`
- `final_factor` involves `final_div_factor` computations

## Advanced Use Cases

### Research and Experimentation
```tcl
# High learning rates for research
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 2.0 1000 0.1 "cos" 100.0 50000.0]
```

### Large-Scale Production Training
```tcl
# Balanced approach for production
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 50000 0.25 "cos" 25.0 10000.0]
```

### Multi-Stage Training
```tcl
# Stage 1: Aggressive training
set scheduler1 [torch::lr_scheduler_onecycle_advanced $optimizer 0.5 5000 0.2 "cos" 20.0 1000.0]

# Stage 2: Refinement (create new scheduler)
set scheduler2 [torch::lr_scheduler_onecycle_advanced $optimizer 0.01 10000 0.5 "linear" 10.0 10000.0]
```

### Domain-Specific Configurations

#### Computer Vision
```tcl
# Typical CV settings
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.1 20000 0.3 "cos" 25.0 10000.0]
```

#### Natural Language Processing
```tcl
# Conservative NLP settings
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.01 15000 0.4 "linear" 20.0 5000.0]
```

#### Speech Recognition
```tcl
# Speech-specific tuning
set scheduler [torch::lr_scheduler_onecycle_advanced $optimizer 0.05 30000 0.25 "cos" 30.0 15000.0]
```

## Comparison with Basic One Cycle

| Feature | Basic OneCycle | Advanced OneCycle |
|---------|----------------|-------------------|
| Parameters | 6 max | 7 max |
| Final LR Control | Limited | Full control via final_div_factor |
| Flexibility | Standard | Enhanced |
| Use Case | General training | Research, fine-tuning, production |

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing required parameters
torch::lr_scheduler_onecycle_advanced $optimizer
# Error: wrong # args: should be "torch::lr_scheduler_onecycle_advanced optimizer max_lr total_steps ?pct_start? ?anneal_strategy? ?div_factor? ?final_div_factor?"

# Invalid optimizer
torch::lr_scheduler_onecycle_advanced "invalid_handle" 0.1 1000
# Error: Invalid optimizer name

# Invalid numeric values
torch::lr_scheduler_onecycle_advanced $optimizer "invalid" 1000
# Error: expected floating-point but got "invalid"
```

## Performance Considerations

- **Computational Overhead**: Minimal additional overhead compared to basic one cycle
- **Memory Usage**: Slightly higher due to additional parameters
- **Training Speed**: No significant impact on training performance
- **Convergence**: May achieve better convergence with proper tuning

## Best Practices

1. **Start Conservative**: Begin with standard parameters and tune incrementally
2. **Monitor Training**: Watch for signs of instability with aggressive settings
3. **Log Learning Rates**: Track learning rate changes during training
4. **Experiment Systematically**: Change one parameter at a time for tuning
5. **Use Validation**: Monitor validation metrics to avoid overfitting

## See Also

- `torch::lr_scheduler_one_cycle` - Basic one cycle scheduler
- `torch::lr_scheduler_cosine_annealing` - Cosine annealing scheduler
- `torch::lr_scheduler_step` - Step-based scheduler
- `torch::lr_scheduler_step_update` - Update scheduler step
- `torch::optimizer_sgd` - SGD optimizer
- `torch::optimizer_adam` - Adam optimizer

## References

- Smith, L.N. "Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates." arXiv:1708.07120
- Smith, L.N. "Cyclical Learning Rates for Training Neural Networks." arXiv:1506.01186
- Howard, J. and Gugger, S. "Deep Learning for Coders with Fastai and PyTorch." O'Reilly Media, 2020. 