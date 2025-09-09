# torch::get_lr_advanced

Get the current learning rate from a learning rate scheduler.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::get_lr_advanced scheduler
```

### Named Parameter Syntax (Recommended)
```tcl
torch::get_lr_advanced -scheduler scheduler
```

### CamelCase Alias
```tcl
torch::getLrAdvanced scheduler
torch::getLrAdvanced -scheduler scheduler
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-scheduler` | string | Yes | Handle of the learning rate scheduler to get the current learning rate from |

## Return Value

Returns the current learning rate as a floating-point number.

## Description

The `torch::get_lr_advanced` command retrieves the current learning rate from a specified learning rate scheduler. This is useful for:

- Monitoring learning rate changes during training
- Debugging learning rate scheduling issues
- Implementing custom training loops with learning rate tracking
- Logging training progress with scheduler information

The command works with all learning rate scheduler types supported by the LibTorch TCL Extension, including:
- Step schedulers
- Exponential schedulers
- Cosine schedulers
- Multi-step schedulers
- Plateau schedulers
- And others

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create parameters, optimizer, and scheduler
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerSgd -parameters $params -lr 0.001]
set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 10 -gamma 0.1]

# Get the current learning rate from scheduler
set lr [torch::get_lr_advanced $scheduler]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.001
```

### Using Named Parameters
```tcl
# Create parameters, optimizer, and scheduler
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerAdam -parameters $params -lr 0.01]
set scheduler [torch::lrSchedulerExponential -optimizer $optimizer -gamma 0.9]

# Get the current learning rate using named parameters
set lr [torch::get_lr_advanced -scheduler $scheduler]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.01
```

### Using CamelCase Alias
```tcl
# Create parameters, optimizer, and scheduler
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerAdamW -parameters $params -lr 0.005]
set scheduler [torch::lrSchedulerCosine -optimizer $optimizer -tMax 100]

# Get the current learning rate using camelCase alias
set lr [torch::getLrAdvanced -scheduler $scheduler]
puts "Current learning rate: $lr"
;# Output: Current learning rate: 0.005
```

### Monitoring Learning Rate Changes with Schedulers
```tcl
# Create parameters, optimizer, and scheduler
set params [torch::tensorCreate -data {1.0 2.0 3.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerSgd -parameters $params -lr 0.1]
set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 5 -gamma 0.5]

# Check initial learning rate
set initial_lr [torch::get_lr_advanced $scheduler]
puts "Initial learning rate: $initial_lr"
;# Output: Initial learning rate: 0.1

# Step the scheduler multiple times
for {set i 0} {$i < 10} {incr i} {
    torch::lrSchedulerStepUpdate $scheduler
    set current_lr [torch::get_lr_advanced $scheduler]
    puts "Step $i: Learning rate = $current_lr"
}
```

### Different Scheduler Types
```tcl
# Step scheduler
set params1 [torch::tensorCreate -data {1.0 2.0} -dtype float32 -requiresGrad true]
set opt1 [torch::optimizerSgd -parameters $params1 -lr 0.001]
set step_scheduler [torch::lrSchedulerStep -optimizer $opt1 -stepSize 10 -gamma 0.1]
set step_lr [torch::get_lr_advanced $step_scheduler]
puts "Step scheduler LR: $step_lr"

# Exponential scheduler
set params2 [torch::tensorCreate -data {3.0 4.0} -dtype float32 -requiresGrad true]
set opt2 [torch::optimizerAdam -parameters $params2 -lr 0.01]
set exp_scheduler [torch::lrSchedulerExponential -optimizer $opt2 -gamma 0.9]
set exp_lr [torch::get_lr_advanced $exp_scheduler]
puts "Exponential scheduler LR: $exp_lr"

# Cosine scheduler
set params3 [torch::tensorCreate -data {5.0 6.0} -dtype float32 -requiresGrad true]
set opt3 [torch::optimizerAdam -parameters $params3 -lr 0.005]
set cos_scheduler [torch::lrSchedulerCosine -optimizer $opt3 -tMax 100]
set cos_lr [torch::get_lr_advanced $cos_scheduler]
puts "Cosine scheduler LR: $cos_lr"

# Multi-step scheduler
set params4 [torch::tensorCreate -data {7.0 8.0} -dtype float32 -requiresGrad true]
set opt4 [torch::optimizerSgd -parameters $params4 -lr 0.02]
set multi_scheduler [torch::lrSchedulerMultiStep -optimizer $opt4 -milestones {30 60 90} -gamma 0.1]
set multi_lr [torch::get_lr_advanced $multi_scheduler]
puts "Multi-step scheduler LR: $multi_lr"
```

### Training Loop with Learning Rate Monitoring
```tcl
# Setup training components
set params [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32 -requiresGrad true]
set optimizer [torch::optimizerSgd -parameters $params -lr 0.1]
set scheduler [torch::lrSchedulerStep -optimizer $optimizer -stepSize 3 -gamma 0.5]

# Training loop
for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Get current learning rate
    set current_lr [torch::get_lr_advanced $scheduler]
    puts "Epoch $epoch: Learning rate = $current_lr"
    
    # Simulate training step
    # ... training logic here ...
    
    # Step the scheduler
    torch::lrSchedulerStepUpdate $scheduler
}
```

## Error Handling

The command will raise an error if:

- The scheduler handle is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided

### Error Examples
```tcl
# Invalid scheduler handle
catch {torch::get_lr_advanced "invalid_scheduler"} msg
puts "Error: $msg"
;# Output: Error: Invalid scheduler handle

# Missing required parameter
catch {torch::get_lr_advanced -scheduler} msg
puts "Error: $msg"
;# Output: Error: Missing value for parameter -scheduler

# Unknown parameter
catch {torch::get_lr_advanced -unknownParam value} msg
puts "Error: $msg"
;# Output: Error: Unknown parameter: -unknownParam. Valid parameters are: -scheduler
```

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set lr [torch::get_lr_advanced $scheduler]

# New named parameter syntax
set lr [torch::get_lr_advanced -scheduler $scheduler]

# Both syntaxes work and produce identical results
```

### Using CamelCase Alias
```tcl
# Traditional snake_case
set lr [torch::get_lr_advanced $scheduler]

# Modern camelCase
set lr [torch::getLrAdvanced $scheduler]

# Both work identically
```

## Notes

- The command returns the current learning rate from the scheduler state
- The returned value reflects any modifications made by the scheduler
- The command is read-only and does not modify the scheduler state
- All scheduler types are supported
- The learning rate is returned as a double-precision floating-point number
- This is different from `torch::get_lr` which gets the learning rate directly from an optimizer

## See Also

- [torch::get_lr](get_lr.md) - Get learning rate directly from optimizer
- [torch::lrSchedulerStep](lr_scheduler_step.md) - Step learning rate scheduler
- [torch::lrSchedulerExponential](lr_scheduler_exponential.md) - Exponential learning rate scheduler
- [torch::lrSchedulerCosine](lr_scheduler_cosine.md) - Cosine learning rate scheduler
- [torch::lrSchedulerMultiStep](lr_scheduler_multi_step.md) - Multi-step learning rate scheduler
- [torch::lrSchedulerPlateau](lr_scheduler_plateau.md) - Plateau learning rate scheduler 