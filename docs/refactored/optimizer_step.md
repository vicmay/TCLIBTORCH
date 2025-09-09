# torch::optimizer_step

Steps the optimizer, performing a single optimization step to update the parameters based on their gradients.

## Syntax

### Positional Syntax (Original)
```tcl
torch::optimizer_step optimizer
```

### Named Parameter Syntax (New)
```tcl
torch::optimizer_step -optimizer handle
```

### CamelCase Alias
```tcl
torch::optimizerStep -optimizer handle
```

## Parameters

### Required Parameters
- **optimizer** (`string`): Handle to an optimizer (created by any `torch::optimizer_*` command)

### Alternative Parameter Names
- `-opt` instead of `-optimizer`

## Return Value
Returns "OK" if the optimization step was successful.

## Examples

### Basic Usage (Positional)
```tcl
# Create optimizer
set params [list $tensor1 $tensor2]
set optimizer [torch::optimizer_sgd $params 0.01]

# Step the optimizer
torch::optimizer_step $optimizer
```

### Named Parameter Syntax
```tcl
# Step the optimizer using named parameters
torch::optimizer_step -optimizer $optimizer
```

### Using Alternative Parameter Name
```tcl
# Step the optimizer using short form
torch::optimizer_step -opt $optimizer
```

### Using CamelCase Alias
```tcl
# Step the optimizer using camelCase
torch::optimizerStep -optimizer $optimizer
```

### Training Loop Example
```tcl
# Typical training loop
set optimizer [torch::optimizer_adam $params 0.001]

for {set epoch 0} {$epoch < 100} {incr epoch} {
    # Zero gradients
    torch::optimizer_zero_grad $optimizer
    
    # Forward pass and loss computation
    set loss [compute_loss]
    
    # Backward pass
    torch::tensor_backward $loss
    
    # Update parameters
    torch::optimizer_step $optimizer
}
```

### Multiple Optimizer Types
```tcl
# Works with any optimizer type
set sgd_opt [torch::optimizer_sgd $params 0.01]
set adam_opt [torch::optimizer_adam $params 0.001]
set rmsprop_opt [torch::optimizer_rmsprop $params 0.001]

torch::optimizer_step $sgd_opt
torch::optimizer_step $adam_opt
torch::optimizer_step $rmsprop_opt
```

## Algorithm Details

The `torch::optimizer_step` command calls the underlying PyTorch optimizer's `step()` method, which:

1. **Applies the optimization algorithm** specific to the optimizer type (SGD, Adam, RMSprop, etc.)
2. **Updates parameters** based on their gradients and the optimizer's internal state
3. **Updates internal state** (momentum buffers, running averages, etc.) as needed by the algorithm

### Optimizer-Specific Behavior

- **SGD**: Updates parameters using gradient descent with optional momentum
- **Adam**: Updates parameters using adaptive moment estimation
- **RMSprop**: Updates parameters using root mean square propagation
- **AdamW**: Updates parameters using Adam with decoupled weight decay
- **And more**: Works with all supported optimizer types

## Use Cases

- **Training Neural Networks**: Essential for parameter updates during training
- **Fine-tuning**: Updating pre-trained model parameters
- **Optimization Research**: Experimenting with different optimization algorithms
- **Transfer Learning**: Adapting models to new tasks

## Error Handling

The command will raise an error if:
- No optimizer handle is provided
- The optimizer handle is invalid or doesn't exist
- The optimizer has been destroyed or is no longer valid
- Named parameters are not provided in pairs

## Performance Considerations

- **Gradient Computation**: Ensure gradients are computed before calling step
- **Memory Usage**: Step operation may use additional memory for optimizer state
- **Device Consistency**: Optimizer and parameters should be on the same device

## Typical Workflow

1. **Create optimizer** with model parameters
2. **Zero gradients** using `torch::optimizer_zero_grad`
3. **Forward pass** through the model
4. **Compute loss** and call `torch::tensor_backward`
5. **Step optimizer** using `torch::optimizer_step`
6. **Repeat** for training loop

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Cross-platform**: Works on CPU and CUDA devices
- **Thread Safe**: Can be used in multi-threaded environments (with proper synchronization)
- **LibTorch**: Uses LibTorch's native optimizer step functionality

## See Also

- `torch::optimizer_zero_grad` - Zero optimizer gradients
- `torch::optimizer_sgd` - Create SGD optimizer
- `torch::optimizer_adam` - Create Adam optimizer
- `torch::tensor_backward` - Compute gradients 