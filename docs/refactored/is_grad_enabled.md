# torch::is_grad_enabled

Check if gradient computation is currently enabled for autograd operations.

## Syntax

```tcl
torch::is_grad_enabled
torch::isGradEnabled
```

## Description

The `torch::is_grad_enabled` command checks the current state of PyTorch's automatic differentiation system. It returns a boolean value indicating whether gradient computation is currently enabled or disabled.

This command allows you to query the global gradient computation state that is controlled by commands like `torch::enable_grad`, `torch::no_grad`, and `torch::set_grad_enabled`.

## Parameters

This command takes no parameters.

## Return Value

Returns:
- `1` if gradient computation is currently enabled
- `0` if gradient computation is currently disabled

## Examples

### Basic Usage

```tcl
# Check current gradient state
set grad_enabled [torch::is_grad_enabled]
puts "Gradients enabled: $grad_enabled"
```

### Using CamelCase Syntax

```tcl
# Check gradient state using camelCase
set grad_enabled [torch::isGradEnabled]
puts "Gradients enabled: $grad_enabled"
```

### Checking State After Operations

```tcl
# Enable gradients and check
torch::enable_grad
set enabled_state [torch::is_grad_enabled]
puts "After enable_grad: $enabled_state"  ;# Output: 1

# Disable gradients and check
torch::no_grad
set disabled_state [torch::is_grad_enabled]
puts "After no_grad: $disabled_state"  ;# Output: 0
```

### State Verification in Conditional Logic

```tcl
# Enable gradients
torch::enable_grad

# Conditional execution based on gradient state
if {[torch::is_grad_enabled]} {
    puts "Gradients are enabled - can perform training"
    # Create tensors that require gradients
    set x [torch::tensorCreate -data {2.0} -shape {1} -dtype float32 -requiresGrad true]
    set y [torch::tensorCreate -data {3.0} -shape {1} -dtype float32 -requiresGrad true]
    
    # Operations will be tracked for gradients
    set z [torch::tensorMul $x $y]
    torch::tensorBackward $z
} else {
    puts "Gradients are disabled - inference mode"
}
```

### Using with set_grad_enabled

```tcl
# Set gradient state with boolean parameter
torch::set_grad_enabled true
puts "After set_grad_enabled true: [torch::is_grad_enabled]"  ;# Output: 1

torch::set_grad_enabled false
puts "After set_grad_enabled false: [torch::is_grad_enabled]"  ;# Output: 0
```

## Usage Patterns

### Training vs Inference Mode Detection

```tcl
proc model_forward {model input} {
    if {[torch::is_grad_enabled]} {
        puts "Running in training mode"
        # Full forward pass with gradient tracking
        return [training_forward $model $input]
    } else {
        puts "Running in inference mode"
        # Optimized forward pass without gradients
        return [inference_forward $model $input]
    }
}
```

### State Restoration

```tcl
# Save current gradient state
set original_grad_state [torch::is_grad_enabled]

# Temporarily disable gradients
torch::no_grad
# ... do some operations without gradients ...

# Restore original state
torch::set_grad_enabled $original_grad_state
puts "Restored to original state: [torch::is_grad_enabled]"
```

### Debug and Logging

```tcl
proc debug_gradient_state {} {
    set state [torch::is_grad_enabled]
    puts "Current gradient computation state: [expr {$state ? "ENABLED" : "DISABLED"}]"
    return $state
}

# Use in debug context
debug_gradient_state
torch::enable_grad
debug_gradient_state
torch::no_grad
debug_gradient_state
```

### Consistency Checks

```tcl
# Verify both syntaxes return the same result
torch::enable_grad
set snake_result [torch::is_grad_enabled]
set camel_result [torch::isGradEnabled]

if {$snake_result == $camel_result} {
    puts "Consistent results: $snake_result"
} else {
    puts "ERROR: Inconsistent results!"
}
```

## Error Handling

The command will return an error if called with any arguments:

```tcl
# These will cause errors
catch {torch::is_grad_enabled extra_arg} error1
catch {torch::isGradEnabled some_param} error2

puts $error1  ;# "wrong # args: should be \"torch::is_grad_enabled\""
puts $error2  ;# "wrong # args: should be \"torch::isGradEnabled\""
```

## Advanced Usage

### Context Manager Pattern

```tcl
proc with_grad_disabled {script} {
    # Save current state
    set original_state [torch::is_grad_enabled]
    
    # Disable gradients
    torch::no_grad
    
    # Execute script
    set result [catch {uplevel $script} error]
    
    # Restore original state
    torch::set_grad_enabled $original_state
    
    # Re-throw error if occurred
    if {$result} {
        error $error
    }
}

# Usage
with_grad_disabled {
    puts "Grad state inside: [torch::is_grad_enabled]"  ;# Output: 0
    # Do inference operations
}
puts "Grad state outside: [torch::is_grad_enabled]"  ;# Output: original state
```

### Performance Monitoring

```tcl
proc measure_with_gradients {operation} {
    # Test with gradients enabled
    torch::enable_grad
    set start_time [clock milliseconds]
    eval $operation
    set time_with_grad [expr {[clock milliseconds] - $start_time}]
    
    # Test with gradients disabled
    torch::no_grad
    set start_time [clock milliseconds]
    eval $operation
    set time_without_grad [expr {[clock milliseconds] - $start_time}]
    
    puts "With gradients: ${time_with_grad}ms"
    puts "Without gradients: ${time_without_grad}ms"
    puts "Overhead: [expr {$time_with_grad - $time_without_grad}]ms"
}
```

## Notes

- **Global State**: This command queries the global gradient computation state
- **Thread Safety**: The gradient state is shared across threads
- **Performance**: This is a lightweight query operation with minimal overhead
- **Consistency**: Both snake_case and camelCase syntaxes return identical results
- **No Side Effects**: This command only queries state and doesn't modify anything

## Related Commands

- `torch::enable_grad` - Enable gradient computation
- `torch::no_grad` - Disable gradient computation  
- `torch::set_grad_enabled` - Set gradient state with boolean parameter
- `torch::tensorRequiresGrad` - Check if specific tensor requires gradients
- `torch::tensorBackward` - Compute gradients via backpropagation

## Compatibility

### Backward Compatibility
- `torch::is_grad_enabled` - Original snake_case syntax (fully supported)

### Modern Syntax  
- `torch::isGradEnabled` - New camelCase syntax (recommended for new code)

Both syntaxes are functionally identical and can be used interchangeably.

## Technical Details

### Implementation

The command internally calls:
```cpp
torch::autograd::GradMode::is_enabled()
```

This queries PyTorch's global autograd mode state.

### Return Value Details

The command returns:
- Integer `1` when gradients are enabled (equivalent to boolean true)
- Integer `0` when gradients are disabled (equivalent to boolean false)

### State Persistence

The gradient state:
- Persists across function calls within the same process
- Is shared between all threads in the application
- Defaults to enabled when the PyTorch extension loads
- Can be changed by any of the gradient control commands

## Use Cases

### Model Training
- Check gradient state before starting training loops
- Verify gradients are enabled for parameter updates
- Debug gradient-related issues

### Model Inference
- Confirm gradients are disabled for inference performance
- Implement automatic optimization based on gradient state
- Validate inference pipelines

### Testing and Debugging
- Assert expected gradient states in unit tests
- Debug gradient tracking issues
- Verify gradient context managers work correctly

### Performance Optimization
- Conditionally optimize operations based on gradient requirements
- Monitor gradient overhead in performance-critical code
- Implement adaptive execution strategies

## Examples in Context

### Training Loop with State Verification

```tcl
proc training_epoch {model data} {
    # Ensure gradients are enabled for training
    if {![torch::is_grad_enabled]} {
        torch::enable_grad
        puts "Enabled gradients for training"
    }
    
    foreach batch $data {
        # Forward pass
        set output [model_forward $model $batch]
        
        # Verify we can compute gradients
        if {[torch::is_grad_enabled]} {
            torch::tensorBackward $output
            # Update parameters
        } else {
            error "Gradients unexpectedly disabled during training"
        }
    }
}
```

### Inference with Automatic Optimization

```tcl
proc model_inference {model input} {
    set grad_was_enabled [torch::is_grad_enabled]
    
    # Disable gradients for inference efficiency
    torch::no_grad
    
    # Run inference
    set result [model_forward $model $input]
    
    # Restore original gradient state
    torch::set_grad_enabled $grad_was_enabled
    
    return $result
}
``` 