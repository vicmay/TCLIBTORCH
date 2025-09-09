# torch::lr_scheduler_noam

Creates a Noam learning rate scheduler that implements the learning rate schedule described in "Attention Is All You Need" paper.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::lr_scheduler_noam optimizer modelSize [warmupSteps]
```

### Named Parameter Syntax (Preferred)
```tcl
torch::lr_scheduler_noam -optimizer optimizer -modelSize modelSize [-warmupSteps warmupSteps]
```

### camelCase Alias
```tcl
torch::lrSchedulerNoam ...
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| optimizer | string | Yes | - | Handle to the optimizer |
| modelSize | integer | Yes | - | Model dimension size (d_model) |
| warmupSteps | integer | No | 4000 | Number of warmup steps |

### Parameter Aliases
- `-modelSize`, `-model_size`
- `-warmupSteps`, `-warmup_steps`

## Returns
Returns a handle to the created Noam learning rate scheduler.

## Description

The Noam scheduler implements the learning rate schedule from the Transformer paper "Attention Is All You Need". The learning rate is calculated as:

```
lr = model_size^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
```

This scheduler:
- Increases linearly for the first `warmupSteps` training steps
- Decreases proportionally to the inverse square root of the step number after warmup
- Is commonly used for training Transformer models

## Examples

### Basic Usage
```tcl
# Create optimizer
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_adam $weights 0.001]

# Create Noam scheduler with default warmup
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512]

# Step the scheduler
torch::lr_scheduler_step_update $scheduler
```

### Custom Warmup Steps
```tcl
# Create optimizer  
set weights [torch::tensor_create {1.0 2.0 3.0} float32]
set optimizer [torch::optimizer_sgd $weights 0.1]

# Create Noam scheduler with custom warmup
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 256 -warmupSteps 8000]

# Step multiple times
for {set i 0} {$i < 10} {incr i} {
    torch::lr_scheduler_step_update $scheduler
}
```

### Different Model Sizes
```tcl
# Large model (higher initial learning rate)
set scheduler1 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 1024]

# Small model (lower initial learning rate)  
set scheduler2 [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 128]
```

### Using camelCase Alias
```tcl
set scheduler [torch::lrSchedulerNoam -optimizer $optimizer -modelSize 512 -warmupSteps 4000]
```

## Mathematical Formula

The Noam scheduler learning rate at step `t` is:

```
lr(t) = d_model^(-0.5) * min(t^(-0.5), t * warmup_steps^(-1.5))
```

Where:
- `d_model` is the model dimension (modelSize parameter)
- `t` is the current step number
- `warmup_steps` is the number of warmup steps

## Common Use Cases

### Transformer Training
```tcl
# Typical settings for Transformer models
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 4000]
```

### BERT-style Models
```tcl
# Larger model dimension
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 768 -warmupSteps 10000]
```

### GPT-style Models
```tcl
# Very large model
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 2048 -warmupSteps 16000]
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set scheduler [torch::lr_scheduler_noam $optimizer 512]
set scheduler [torch::lr_scheduler_noam $optimizer 512 8000]
```

**New (Named Parameters):**
```tcl
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512]
set scheduler [torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps 8000]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Safety**: Reduces errors from parameter misplacement
- **Maintainability**: Easier to modify and understand

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing required parameter
torch::lr_scheduler_noam -optimizer $optimizer
# Error: Required parameter 'modelSize' is missing

# Invalid model size
torch::lr_scheduler_noam -optimizer $optimizer -modelSize 0
# Error: modelSize must be positive

# Invalid warmup steps
torch::lr_scheduler_noam -optimizer $optimizer -modelSize 512 -warmupSteps -100
# Error: warmupSteps must be positive
```

## Performance Considerations

- The scheduler computation is efficient with O(1) complexity per step
- Model size affects the initial learning rate scale
- Warmup steps determine the schedule shape but don't affect runtime performance
- Suitable for long training runs typical in large language model training

## See Also

- `torch::lr_scheduler_cosine_annealing` - Cosine annealing scheduler
- `torch::lr_scheduler_exponential` - Exponential decay scheduler
- `torch::lr_scheduler_step` - Step-based scheduler
- `torch::lr_scheduler_step_update` - Update scheduler step
- `torch::optimizer_adam` - Adam optimizer
- `torch::optimizer_sgd` - SGD optimizer

## References

- Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
- The learning rate schedule described in Section 5.3 of the Transformer paper. 