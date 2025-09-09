# torch::no_grad / torch::noGrad

Disables gradient computation globally. This is useful for inference or when you want to prevent gradient computation for a specific section of code.

## Syntax

```tcl
torch::no_grad
torch::noGrad
```

This command takes no parameters.

## Description

The `torch::no_grad` command disables gradient computation globally by setting PyTorch's autograd mode to false. This means that no gradients will be computed or recorded for any operations performed after this command is called, even if the tensors involved have `requires_grad=true`.

This is particularly useful in two scenarios:
1. During model inference/evaluation where gradients are not needed
2. When you want to perform operations on tensors without accumulating gradients

The effect of `torch::no_grad` remains active until explicitly re-enabled using `torch::enable_grad`.

## Return Value

Returns "ok" on successful execution.

## Examples

### Basic Usage
```tcl
# Create a tensor that requires gradients
set x [torch::tensor_create -data {1.0 2.0 3.0} -requires_grad true]

# Disable gradient computation
torch::no_grad

# Operations performed here won't compute gradients
set y [torch::tensor_mul $x 2.0]
# y will have requires_grad=false

# Re-enable gradient computation when needed
torch::enable_grad
```

### Using with Model Evaluation
```tcl
# During training
torch::enable_grad
# ... training code ...

# During evaluation
torch::no_grad
# ... evaluation code ...
# No gradients will be computed, saving memory
```

## Error Handling

The command will raise an error if:
- Any arguments are provided (the command takes no arguments)

## See Also

- `torch::enable_grad` - Enable gradient computation
- `torch::set_grad_enabled` - Set gradient computation state
- `torch::is_grad_enabled` - Check if gradient computation is enabled
- `torch::tensor_requires_grad` - Check if a tensor requires gradients 