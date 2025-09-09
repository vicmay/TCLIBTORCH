# torch::enable_grad

Enable gradient computation for autograd operations.

## Syntax

```tcl
torch::enable_grad
torch::enableGrad
```

## Description

The `torch::enable_grad` command enables gradient computation for PyTorch's automatic differentiation system. When gradient computation is enabled, PyTorch will track operations on tensors that have `requires_grad=True` and build a computational graph for backpropagation.

This command is the opposite of `torch::no_grad` and sets the global gradient computation state to enabled. It affects all subsequent tensor operations until the state is changed again.

## Parameters

This command takes no parameters.

## Return Value

Returns the string `"ok"` upon successful execution.

## Examples

### Basic Usage

```tcl
# Enable gradient computation
torch::enable_grad

# Check if gradients are enabled
set grad_enabled [torch::is_grad_enabled]
puts "Gradients enabled: $grad_enabled"  ;# Output: 1
```

### Using CamelCase Syntax

```tcl
# Enable gradient computation using camelCase
torch::enableGrad

# Check the state
set grad_enabled [torch::is_grad_enabled]
puts "Gradients enabled: $grad_enabled"  ;# Output: 1
```

### Enabling After Disabling

```tcl
# Disable gradients first
torch::no_grad
puts "Gradients enabled: [torch::is_grad_enabled]"  ;# Output: 0

# Re-enable gradients
torch::enable_grad
puts "Gradients enabled: [torch::is_grad_enabled]"  ;# Output: 1
```

### Impact on Tensor Operations

```tcl
# Enable gradients
torch::enable_grad

# Create tensors with gradient tracking
set x [torch::tensorCreate -data {2.0} -shape {1} -dtype float32 -requires_grad true]
set y [torch::tensorCreate -data {3.0} -shape {1} -dtype float32 -requires_grad true]

# Perform operation - gradient graph will be built
set z [torch::tensorMul $x $y]

# Check if result requires gradients
puts "Result requires grad: [torch::tensorRequiresGrad $z]"  ;# Output: 1
```

### Mixed Usage with no_grad

```tcl
# Start with gradients enabled
torch::enable_grad
puts "Initial state: [torch::is_grad_enabled]"  ;# Output: 1

# Temporarily disable
torch::no_grad
puts "After no_grad: [torch::is_grad_enabled]"  ;# Output: 0

# Re-enable using camelCase
torch::enableGrad
puts "After enableGrad: [torch::is_grad_enabled]"  ;# Output: 1
```

## Usage Patterns

### Global Gradient Control

```tcl
# Enable gradients globally for training
torch::enable_grad

# Create model parameters
set weight [torch::tensorCreate -data {0.5 0.3} -shape {2} -dtype float32 -requires_grad true]
set bias [torch::tensorCreate -data {0.1} -shape {1} -dtype float32 -requires_grad true]

# Forward pass - gradients will be tracked
set input [torch::tensorCreate -data {1.0 2.0} -shape {2} -dtype float32]
set linear_out [torch::tensorAdd [torch::tensorDot $input $weight] $bias]

# Can now compute gradients
torch::tensorBackward $linear_out
```

### Idempotent Operations

```tcl
# Multiple calls are safe and idempotent
torch::enable_grad
torch::enable_grad
torch::enableGrad
torch::enable_grad

# State remains enabled
puts "Final state: [torch::is_grad_enabled]"  ;# Output: 1
```

## Error Handling

The command will return an error if called with any arguments:

```tcl
# These will cause errors
catch {torch::enable_grad extra_arg} error1
catch {torch::enableGrad some_param} error2

puts $error1  ;# "wrong # args: should be \"torch::enable_grad\""
puts $error2  ;# "wrong # args: should be \"torch::enableGrad\""
```

## Notes

- **Global State**: This command affects the global gradient computation state for all subsequent operations
- **Thread Safety**: The gradient state is shared across threads
- **Performance**: Enabling gradients has computational overhead for tracking operations
- **Memory**: Gradient tracking uses additional memory to store the computational graph
- **Idempotent**: Multiple calls to enable gradients are safe and don't change behavior

## Related Commands

- `torch::no_grad` - Disable gradient computation
- `torch::set_grad_enabled` - Set gradient state with boolean parameter
- `torch::is_grad_enabled` - Check current gradient computation state
- `torch::tensorBackward` - Compute gradients via backpropagation
- `torch::tensorRequiresGrad` - Check if tensor requires gradients

## Compatibility

### Backward Compatibility
- `torch::enable_grad` - Original snake_case syntax (fully supported)

### Modern Syntax  
- `torch::enableGrad` - New camelCase syntax (recommended for new code)

Both syntaxes are functionally identical and can be used interchangeably.

## Technical Details

### Implementation

The command internally calls:
```cpp
torch::autograd::GradMode::set_enabled(true);
```

### Gradient Computation Context

When gradients are enabled:
1. Operations on tensors with `requires_grad=True` are tracked
2. A computational graph is built automatically
3. Memory usage increases to store graph nodes
4. Backward pass can compute gradients through the graph

### Performance Considerations

- **Training**: Enable gradients during training phases
- **Inference**: Consider disabling gradients during inference for better performance
- **Mixed Mode**: Can enable/disable as needed for different code sections

## Examples in Context

### Training Loop

```tcl
proc train_step {model data target} {
    # Enable gradients for training
    torch::enable_grad
    
    # Forward pass
    set output [model_forward $model $data]
    set loss [torch::mse_loss $output $target]
    
    # Backward pass
    torch::tensorBackward $loss
    
    return $loss
}
```

### Evaluation Mode

```tcl
proc evaluate {model data target} {
    # Disable gradients for evaluation
    torch::no_grad
    
    # Forward pass only
    set output [model_forward $model $data]
    set loss [torch::mse_loss $output $target]
    
    # Re-enable for next training
    torch::enableGrad
    
    return $loss
}
``` 