# torch::get_lr

Get the current learning rate from an optimizer.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::get_lr optimizer
```

### Named Parameter Syntax (Recommended)
```tcl
torch::get_lr -optimizer optimizer
```

### CamelCase Alias
```tcl
torch::getLr optimizer
torch::getLr -optimizer optimizer
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-optimizer` | string | Yes | Handle of the optimizer to get the learning rate from |

## Return Value

Returns the current learning rate as a floating-point number.

## Description

The `torch::get_lr` command retrieves the current learning rate from a specified optimizer. This is useful for:

- Monitoring the learning rate during training
- Debugging training issues
- Implementing custom learning rate schedules
- Logging training progress

The command works with all optimizer types supported by the LibTorch TCL Extension, including:
- SGD
- Adam
- AdamW
- RMSprop
- Adagrad
- And others

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create parameters and optimizer
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerSgd -parameters $params -lr 0.001]

# Get the current learning rate
set lr [torch::get_lr $optimizer]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.001
```

### Using Named Parameters
```tcl
# Create parameters and optimizer
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerAdam -parameters $params -lr 0.01]

# Get the current learning rate using named parameters
set lr [torch::get_lr -optimizer $optimizer]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.01
```

### Using CamelCase Alias
```tcl
# Create parameters and optimizer
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerAdamW -parameters $params -lr 0.005]

# Get the current learning rate using camelCase alias
set lr [torch::getLr -optimizer $optimizer]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.005
```

### Monitoring Learning Rate Changes
```tcl
# Create parameters and optimizer
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerSgd -parameters $params -lr 0.1]

# Create a step scheduler
set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 10 -gamma 0.1]

# Check initial learning rate
set initial_lr [torch::get_lr $optimizer]
puts "Initial learning rate: $initial_lr"
;# Output: Initial learning rate: 0.1

# Step the scheduler
torch::lrSchedulerStepUpdate $scheduler

# Check updated learning rate
set updated_lr [torch::get_lr $optimizer]
puts "Updated learning rate: $updated_lr"
;# Output: Updated learning rate: 0.01
```

### Different Optimizer Types
```tcl
# SGD optimizer
set params1 [torch::tensorCreate -data {1.0 2.0} -dtype float32 -requiresGrad true]
set sgd_opt [torch::optimizerSgd -parameters $params1 -lr 0.001]
set sgd_lr [torch::get_lr $sgd_opt]
puts "SGD learning rate: $sgd_lr"

# Adam optimizer
set params2 [torch::tensorCreate -data {3.0 4.0} -dtype float32 -requiresGrad true]
set adam_opt [torch::optimizerAdam -parameters $params2 -lr 0.01]
set adam_lr [torch::get_lr $adam_opt]
puts "Adam learning rate: $adam_lr"

# RMSprop optimizer
set params3 [torch::tensorCreate -data {5.0 6.0} -dtype float32 -requiresGrad true]
set rmsprop_opt [torch::optimizerRmsprop -parameters $params3 -lr 0.005]
set rmsprop_lr [torch::get_lr $rmsprop_opt]
puts "RMSprop learning rate: $rmsprop_lr"
```

## Error Handling

The command will raise an error if:

- The optimizer handle is invalid or doesn't exist
- The learning rate cannot be retrieved from the optimizer
- Required parameters are missing
- Unknown parameters are provided

### Error Examples
```tcl
# Invalid optimizer handle
catch {torch::get_lr "invalid_optimizer"} msg
puts "Error: $msg"
;# Output: Error: Invalid optimizer name or could not get learning rate

# Missing required parameter
catch {torch::get_lr -optimizer} msg
puts "Error: $msg"
;# Output: Error: Missing value for parameter -optimizer

# Unknown parameter
catch {torch::get_lr -unknownParam value} msg
puts "Error: $msg"
;# Output: Error: Unknown parameter: -unknownParam. Valid parameters are: -optimizer
```

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set lr [torch::get_lr $optimizer]

# New named parameter syntax
set lr [torch::get_lr -optimizer $optimizer]

# Both syntaxes work and produce identical results
```

### Using CamelCase Alias
```tcl
# Traditional snake_case
set lr [torch::get_lr $optimizer]

# Modern camelCase
set lr [torch::getLr $optimizer]

# Both work identically
```

## Notes

- The command returns the current learning rate as set in the optimizer
- If the learning rate has been modified by schedulers, the returned value reflects the current (modified) learning rate
- The command is read-only and does not modify the optimizer state
- All optimizer types are supported
- The learning rate is returned as a double-precision floating-point number

## See Also

- [torch::optimizerSgd](optimizer_sgd.md) - SGD optimizer
- [torch::optimizerAdam](optimizer_adam.md) - Adam optimizer
- [torch::optimizerAdamW](optimizer_adamw.md) - AdamW optimizer
- [torch::lrSchedulerStep](lr_scheduler_step.md) - Step learning rate scheduler
- [torch::lrSchedulerExponential](lr_scheduler_exponential.md) - Exponential learning rate scheduler 