# torch::optimizer_zero_grad

Zeros the gradients of all parameters managed by the optimizer. This is typically called before computing gradients for a new batch in the training loop.

## Syntax

### Positional Syntax (Original)
```tcl
torch::optimizer_zero_grad optimizer ?set_to_none?
```

### Named Parameter Syntax (New)
```tcl
torch::optimizer_zero_grad -optimizer handle ?-setToNone bool?
```

### CamelCase Alias
```tcl
torch::optimizerZeroGrad -optimizer handle ?-setToNone bool?
```

## Parameters

### Required Parameters
- **optimizer** (`string`): Handle to an optimizer (created by any `torch::optimizer_*` command)

### Optional Parameters
- **setToNone** (`boolean`): Whether to set gradients to None instead of zero (default: true)
  - `true`: Sets gradients to None (more memory efficient)
  - `false`: Sets gradients to zero tensors

### Alternative Parameter Names
- `-opt` instead of `-optimizer`
- `-set_to_none` instead of `-setToNone`

## Return Value
Returns "OK" if the gradient zeroing was successful.

## Examples

### Basic Usage (Positional)
```tcl
# Create optimizer
set params [list $tensor1 $tensor2]
set optimizer [torch::optimizer_sgd $params 0.01]

# Zero gradients (default behavior)
torch::optimizer_zero_grad $optimizer
```

### Explicit Set to None
```tcl
# Explicitly set gradients to None (memory efficient)
torch::optimizer_zero_grad $optimizer true
```

### Set to Zero Tensors
```tcl
# Set gradients to zero tensors instead of None
torch::optimizer_zero_grad $optimizer false
```

### Named Parameter Syntax
```tcl
# Zero gradients using named parameters
torch::optimizer_zero_grad -optimizer $optimizer
```

### Named Parameter with setToNone
```tcl
# Zero gradients with explicit setToNone setting
torch::optimizer_zero_grad -optimizer $optimizer -setToNone false
```

### Using Alternative Parameter Names
```tcl
# Using short form and alternative parameter name
torch::optimizer_zero_grad -opt $optimizer -set_to_none true
```

### Using CamelCase Alias
```tcl
# Zero gradients using camelCase
torch::optimizerZeroGrad -optimizer $optimizer -setToNone true
```

### Training Loop Example
```tcl
# Typical training loop
set optimizer [torch::optimizer_adam $params 0.001]

for {set epoch 0} {$epoch < 100} {incr epoch} {
    for {set batch 0} {$batch < $num_batches} {incr batch} {
        # Zero gradients from previous step
        torch::optimizer_zero_grad $optimizer
        
        # Forward pass
        set output [model_forward $batch_data]
        set loss [compute_loss $output $targets]
        
        # Backward pass
        torch::tensor_backward $loss
        
        # Update parameters
        torch::optimizer_step $optimizer
    }
}
```

### Memory Optimization Example
```tcl
# For memory-constrained scenarios, use set_to_none=false
torch::optimizer_zero_grad $optimizer false

# For maximum memory efficiency, use set_to_none=true (default)
torch::optimizer_zero_grad $optimizer true
```

## Algorithm Details

The `torch::optimizer_zero_grad` command calls the underlying PyTorch optimizer's `zero_grad()` method, which:

1. **Iterates through all parameters** managed by the optimizer
2. **Clears gradients** for each parameter using one of two methods:
   - **Set to None** (default): `param.grad = None` - more memory efficient
   - **Set to Zero**: `param.grad.zero_()` - preserves gradient tensor structure

### Set to None vs Zero Tensors

| Method | Memory Usage | Performance | Use Case |
|--------|--------------|-------------|----------|
| **Set to None** (default) | Lower | Faster | General training, memory-constrained |
| **Set to Zero** | Higher | Slower | When gradient tensor structure must be preserved |

## Use Cases

- **Training Loops**: Essential before each forward pass in training
- **Memory Management**: Clearing gradients to free memory
- **Multi-step Optimization**: Clearing gradients between optimization steps
- **Gradient Accumulation**: Selectively clearing gradients in accumulation scenarios

## Error Handling

The command will raise an error if:
- No optimizer handle is provided
- The optimizer handle is invalid or doesn't exist
- The `set_to_none` parameter is not a valid boolean value
- Named parameters are not provided in pairs

## Performance Considerations

### Memory Efficiency
- **set_to_none=true**: More memory efficient, gradients are deallocated
- **set_to_none=false**: Less memory efficient, gradient tensors are preserved but zeroed

### Performance Impact
- **set_to_none=true**: Faster, no tensor operations needed
- **set_to_none=false**: Slower, requires tensor zeroing operations

### Best Practices
- Use `set_to_none=true` (default) for most training scenarios
- Use `set_to_none=false` only when you need to preserve gradient tensor structure
- Always call before computing new gradients in training loops

## Typical Workflow

1. **Create optimizer** with model parameters
2. **Training loop**:
   - **Zero gradients** using `torch::optimizer_zero_grad`
   - **Forward pass** through the model
   - **Compute loss** 
   - **Backward pass** using `torch::tensor_backward`
   - **Step optimizer** using `torch::optimizer_step`

## Memory Usage Patterns

```tcl
# High-memory scenario (large models)
torch::optimizer_zero_grad $optimizer true   # Use set_to_none=true

# Gradient accumulation scenario
for {set i 0} {$i < $accumulation_steps} {incr i} {
    if {$i == 0} {
        torch::optimizer_zero_grad $optimizer  # Zero at start
    }
    # Forward and backward passes...
}
torch::optimizer_step $optimizer
```

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Cross-platform**: Works on CPU and CUDA devices
- **Thread Safe**: Can be used in multi-threaded environments (with proper synchronization)
- **LibTorch**: Uses LibTorch's native optimizer zero_grad functionality
- **All Optimizers**: Works with SGD, Adam, RMSprop, AdamW, and all other optimizer types

## See Also

- `torch::optimizer_step` - Step the optimizer
- `torch::optimizer_sgd` - Create SGD optimizer
- `torch::optimizer_adam` - Create Adam optimizer
- `torch::tensor_backward` - Compute gradients
- `torch::tensor_grad` - Access tensor gradients 