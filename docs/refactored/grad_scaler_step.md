# torch::grad_scaler_step

Step an optimizer using gradient scaler for mixed precision training.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::grad_scaler_step -scaler gradient_scaler -optimizer optimizer_handle
torch::gradScalerStep -scaler gradient_scaler -optimizer optimizer_handle
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::grad_scaler_step gradient_scaler optimizer_handle
torch::gradScalerStep gradient_scaler optimizer_handle
```

## Parameters

### Named Parameters
- `-scaler` (or `-gradScaler`) - Gradient scaler handle (required)
- `-optimizer` (or `-optim`) - Optimizer handle (required)

### Positional Parameters
1. `gradient_scaler` - Gradient scaler handle
2. `optimizer_handle` - Optimizer handle

## Return Value
Returns "scaler step completed" on success.

## Description

This command steps an optimizer using gradient scaler for automatic mixed precision (AMP) training. It unscales gradients, checks for infinite/NaN values, and steps the optimizer only if no infinite gradients are found. This is a crucial step in the AMP training workflow.

The command automatically:
1. Unscales gradients from the optimizer's parameters
2. Checks for infinite or NaN gradients 
3. Steps the optimizer only if gradients are finite
4. Skips optimizer step if infinite gradients are detected

## Examples

### Basic Usage with Named Parameters
```tcl
;# Create gradient scaler and optimizer
set scaler [torch::grad_scaler_new]
set tensor [torch::randn -shape {10 10} -dtype float32]
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]

;# Step optimizer with scaler
set result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
puts $result  ;# "scaler step completed"
```

### Basic Usage with Positional Parameters (Legacy)
```tcl
;# Create gradient scaler and optimizer
set scaler [torch::grad_scaler_new]
set tensor [torch::randn -shape {10 10} -dtype float32]
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]

;# Step optimizer with scaler (positional)
set result [torch::grad_scaler_step $scaler $optimizer]
puts $result  ;# "scaler step completed"
```

### Using camelCase Alias
```tcl
;# Create gradient scaler and optimizer
set scaler [torch::grad_scaler_new]
set tensor [torch::randn -shape {5 5} -dtype float32]
set optimizer [torch::optimizer_adam -parameters [list $tensor] -lr 0.001]

;# Step with camelCase alias
set result [torch::gradScalerStep -scaler $scaler -optimizer $optimizer]
puts $result  ;# "scaler step completed"
```

### Alternative Parameter Names
```tcl
;# Using alternative parameter names
set scaler [torch::grad_scaler_new]
set tensor [torch::randn -shape {3 3} -dtype float32]  
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]

;# Using -gradScaler and -optim aliases
set result [torch::grad_scaler_step -gradScaler $scaler -optim $optimizer]
puts $result  ;# "scaler step completed"
```

### Complete AMP Training Workflow
```tcl
;# Initialize scaler and model parameters
set scaler [torch::grad_scaler_new -initScale 1024.0]
set weights [torch::randn -shape {10 5} -dtype float32]
set bias [torch::zeros -shape {5} -dtype float32]

;# Create optimizer with model parameters
set optimizer [torch::optimizer_adam -parameters [list $weights $bias] -lr 0.001]

;# Forward pass (dummy loss)
set loss [torch::randn -shape {1} -dtype float32]

;# Scale loss for backward pass
set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss]

;# Backward pass would happen here in real training
;# ... backward pass computation ...

;# Step optimizer with gradient scaler
set step_result [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
puts "Step result: $step_result"

;# Update scaler for next iteration
set update_result [torch::grad_scaler_update -scaler $scaler]
puts "Update result: $update_result"
```

### Working with Multiple Optimizers
```tcl
;# Create scaler and multiple optimizers
set scaler [torch::grad_scaler_new]
set tensor1 [torch::randn -shape {10 10} -dtype float32]
set tensor2 [torch::randn -shape {5 5} -dtype float32]

set optimizer1 [torch::optimizer_sgd -parameters [list $tensor1] -lr 0.01]
set optimizer2 [torch::optimizer_adam -parameters [list $tensor2] -lr 0.001]

;# Step both optimizers
set result1 [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer1]
set result2 [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer2]

puts "SGD step: $result1"
puts "Adam step: $result2"
```

### Parameter Order Flexibility
```tcl
;# Parameters can be provided in any order
set scaler [torch::grad_scaler_new]
set tensor [torch::randn -shape {4 4} -dtype float32]
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]

;# Both are equivalent
set result1 [torch::grad_scaler_step -scaler $scaler -optimizer $optimizer]
set result2 [torch::grad_scaler_step -optimizer $optimizer -scaler $scaler]

puts "Result 1: $result1"
puts "Result 2: $result2"
```

## Error Handling

### Invalid Scaler Handle
```tcl
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
catch {torch::grad_scaler_step -scaler invalid_scaler -optimizer $optimizer} error
puts "Error: $error"  ;# "Gradient scaler not found"
```

### Invalid Optimizer Handle
```tcl
set scaler [torch::grad_scaler_new]
catch {torch::grad_scaler_step -scaler $scaler -optimizer invalid_optimizer} error
puts "Error: $error"  ;# "Optimizer not found"
```

### Missing Required Parameters
```tcl
set scaler [torch::grad_scaler_new]
catch {torch::grad_scaler_step -scaler $scaler} error
puts "Error: $error"  ;# "Required parameters missing: scaler and optimizer handles required"
```

### Unknown Parameters
```tcl
set scaler [torch::grad_scaler_new]
set optimizer [torch::optimizer_sgd -parameters [list $tensor] -lr 0.01]
catch {torch::grad_scaler_step -invalidParam value -scaler $scaler -optimizer $optimizer} error
puts "Error: $error"  ;# "Unknown parameter: -invalidParam"
```

## Migration Guide

### From Positional to Named Parameters
```tcl
;# Old positional syntax
torch::grad_scaler_step $scaler $optimizer

;# New named parameter syntax  
torch::grad_scaler_step -scaler $scaler -optimizer $optimizer

;# Or using camelCase alias
torch::gradScalerStep -scaler $scaler -optimizer $optimizer
```

Both syntaxes are supported and produce identical results.

## Technical Notes

- This command is thread-safe and can be used in multi-threaded environments
- The scaler automatically handles gradient unscaling and infinity checking
- Only finite gradients will trigger an optimizer step
- Compatible with all PyTorch optimizers supported by the system
- Part of the automatic mixed precision training workflow

## See Also

- [torch::grad_scaler_new](grad_scaler_new.md) - Create gradient scaler
- [torch::grad_scaler_scale](grad_scaler_scale.md) - Scale tensors/gradients  
- [torch::grad_scaler_update](grad_scaler_update.md) - Update scaler state
- [torch::grad_scaler_get_scale](grad_scaler_get_scale.md) - Get current scale value
- [torch::optimizer_sgd](optimizer_sgd.md) - SGD optimizer
- [torch::optimizer_adam](optimizer_adam.md) - Adam optimizer

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added named parameter support and camelCase alias (`torch::gradScalerStep`)
- **v2.0**: Added dual syntax parser with backward compatibility
- **v2.0**: Enhanced error handling and parameter validation 