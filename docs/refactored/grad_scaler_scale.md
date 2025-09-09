# torch::grad_scaler_scale

Scales a tensor using a gradient scaler for automatic mixed precision (AMP) training.

## Syntax

### Modern Syntax (Named Parameters)
```tcl
torch::grad_scaler_scale -scaler gradient_scaler -tensor tensor_handle
torch::gradScalerScale -scaler gradient_scaler -tensor tensor_handle
```

### Legacy Syntax (Positional Parameters)
```tcl
torch::grad_scaler_scale gradient_scaler tensor_handle
torch::gradScalerScale gradient_scaler tensor_handle
```

## Parameters

### Named Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `-scaler` | string | Handle to the gradient scaler created by `torch::grad_scaler_new` |
| `-tensor` | string | Handle to the tensor to be scaled |

### Parameter Aliases

- `-scaler` / `-gradScaler` - Gradient scaler handle
- `-tensor` / `-input` - Tensor handle

## Return Value

Returns a new tensor handle containing the scaled tensor values (original tensor × scale factor).

## Description

This function multiplies a tensor by the current scale factor stored in the gradient scaler. It is a core component of automatic mixed precision training that:

1. **Prevents gradient underflow** by scaling gradients to a larger range
2. **Maintains numerical stability** during mixed precision computation
3. **Preserves gradient information** that would otherwise be lost to float16 precision limits
4. **Enables efficient training** with reduced memory usage

### Mathematical Operation

```
scaled_tensor = input_tensor × scaler.scale
```

Where `scaler.scale` is the current scale value from the gradient scaler.

## Examples

### Basic Usage

```tcl
# Create a gradient scaler and tensor
set scaler [torch::grad_scaler_new -initScale 1024.0]
set tensor [torch::tensor_create {0.001 0.002 0.003}]

# Scale the tensor (modern syntax)
set scaled [torch::grad_scaler_scale -scaler $scaler -tensor $tensor]

# Scale the tensor (legacy syntax)
set scaled [torch::grad_scaler_scale $scaler $tensor]
```

### Complete AMP Training Example

```tcl
# Setup
set model [torch::sequential]
set optimizer [torch::optimizer_adam ...]
set scaler [torch::grad_scaler_new -initScale 2048.0]

# Training step
for {set batch 0} {$batch < 100} {incr batch} {
    # Forward pass with autocast
    torch::autocast_enable
    set logits [torch::sequential_forward $model $input]
    set loss [torch::cross_entropy_loss $logits $target]
    torch::autocast_disable
    
    # Scale loss for backward pass
    set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss]
    
    # Backward pass
    torch::tensor_backward $scaled_loss
    
    # Unscale gradients and step optimizer
    torch::grad_scaler_step $scaler $optimizer
    torch::grad_scaler_update $scaler
    torch::optimizer_zero_grad $optimizer
}
```

### Using Different Syntax Styles

```tcl
set scaler [torch::grad_scaler_new -initScale 512.0]
set gradient [torch::tensor_create {0.0001 0.0002 0.0003}]

# Modern named parameter syntax
set scaled1 [torch::grad_scaler_scale -scaler $scaler -tensor $gradient]

# Legacy positional syntax
set scaled2 [torch::grad_scaler_scale $scaler $gradient]

# CamelCase alias with named parameters
set scaled3 [torch::gradScalerScale -scaler $scaler -tensor $gradient]

# Parameter aliases
set scaled4 [torch::grad_scaler_scale -gradScaler $scaler -input $gradient]
```

### Multiple Tensor Scaling

```tcl
set scaler [torch::grad_scaler_new -initScale 1024.0]

# Scale multiple tensors with the same scaler
set loss_tensor [torch::tensor_create {0.001}]
set grad_tensor1 [torch::tensor_create {0.0001 0.0002}]
set grad_tensor2 [torch::tensor_create {0.0003 0.0004}]

set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss_tensor]
set scaled_grad1 [torch::grad_scaler_scale -scaler $scaler -tensor $grad_tensor1]
set scaled_grad2 [torch::grad_scaler_scale -scaler $scaler -tensor $grad_tensor2]
```

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid scaler handle
catch {torch::grad_scaler_scale -scaler invalid_scaler -tensor $tensor} error
puts $error  ;# "Gradient scaler not found"

# Invalid tensor handle
catch {torch::grad_scaler_scale -scaler $scaler -tensor invalid_tensor} error
puts $error  ;# "Tensor not found"

# Missing required parameters
catch {torch::grad_scaler_scale -tensor $tensor} error
puts $error  ;# "Required parameters missing: scaler and tensor handles required"

# Unknown parameter
catch {torch::grad_scaler_scale -invalidParam value -scaler $scaler -tensor $tensor} error
puts $error  ;# "Unknown parameter: -invalidParam"
```

## Integration with AMP Workflow

This function is typically used as part of the automatic mixed precision training workflow:

### 1. Scale Loss for Backward Pass
```tcl
set loss [torch::mse_loss $predictions $targets]
set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss]
torch::tensor_backward $scaled_loss
```

### 2. Scale Individual Gradients (if needed)
```tcl
# Usually not needed as backward pass handles this automatically
set gradient [torch::tensor_grad $parameter]
set scaled_gradient [torch::grad_scaler_scale -scaler $scaler -tensor $gradient]
```

### 3. Integration with Gradient Accumulation
```tcl
set accumulated_loss [torch::zeros {1}]

for {set i 0} {$i < $accumulation_steps} {incr i} {
    set batch_loss [torch::mse_loss $pred $target]
    set scaled_batch_loss [torch::grad_scaler_scale -scaler $scaler -tensor $batch_loss]
    set accumulated_loss [torch::tensor_add $accumulated_loss $scaled_batch_loss]
}

# Average and backward
set avg_loss [torch::tensor_div $accumulated_loss [torch::tensor_create [list $accumulation_steps]]]
torch::tensor_backward $avg_loss
```

## Performance Considerations

### Scale Factor Selection
- **Appropriate scaling** prevents gradient underflow while avoiding overflow
- **Too small**: Gradients may underflow to zero (vanishing gradients)
- **Too large**: Gradients may overflow to infinity (exploding gradients)
- **Dynamic scaling** via `torch::grad_scaler_update` adapts automatically

### Memory Usage
- **Creates new tensor**: Does not modify the input tensor in-place
- **Memory efficient**: Uses same precision as input tensor
- **Temporary tensors**: Intermediate scaled tensors can be freed after use

### Numerical Stability
```tcl
# Good: Scale before accumulated operations
set scaled1 [torch::grad_scaler_scale -scaler $scaler -tensor $grad1]
set scaled2 [torch::grad_scaler_scale -scaler $scaler -tensor $grad2]
set sum [torch::tensor_add $scaled1 $scaled2]

# Less optimal: Scale after accumulated operations (may lose precision)
set sum [torch::tensor_add $grad1 $grad2]
set scaled_sum [torch::grad_scaler_scale -scaler $scaler -tensor $sum]
```

## Common Patterns

### 1. Loss Scaling
```tcl
set loss [torch::compute_loss $predictions $targets]
set scaled_loss [torch::grad_scaler_scale -scaler $scaler -tensor $loss]
torch::tensor_backward $scaled_loss
```

### 2. Gradient Scaling (Manual)
```tcl
# Usually handled automatically by backward pass, but can be done manually
set param_grad [torch::tensor_grad $parameter]
set scaled_grad [torch::grad_scaler_scale -scaler $scaler -tensor $param_grad]
```

### 3. Multi-Loss Scaling
```tcl
set main_loss [torch::mse_loss $pred $target]
set regularization_loss [torch::l2_regularization $model]
set total_loss [torch::tensor_add $main_loss $regularization_loss]
set scaled_total [torch::grad_scaler_scale -scaler $scaler -tensor $total_loss]
```

## Troubleshooting

### Common Issues

1. **"Gradient scaler not found"**
   - Ensure the scaler handle is valid and created with `torch::grad_scaler_new`
   - Check that the scaler hasn't been accidentally unset

2. **"Tensor not found"**
   - Verify the tensor handle is valid and the tensor exists
   - Ensure the tensor wasn't freed or went out of scope

3. **Numerical Issues**
   - Check scale factor with `torch::grad_scaler_get_scale`
   - Monitor for overflow/underflow in scaled tensors
   - Adjust initial scale or growth/backoff factors in scaler

### Best Practices

1. **Always scale losses** before calling `torch::tensor_backward`
2. **Use consistent scaler** throughout training loop
3. **Monitor scale factor** to ensure appropriate range
4. **Free intermediate tensors** to manage memory usage
5. **Test without scaling** first to ensure training works

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::grad_scaler_new](grad_scaler_new.md) - Create gradient scaler
- [torch::grad_scaler_step](grad_scaler_step.md) - Step optimizer with scaling
- [torch::grad_scaler_update](grad_scaler_update.md) - Update gradient scaler
- [torch::grad_scaler_get_scale](grad_scaler_get_scale.md) - Get current scale
- [torch::autocast_enable](autocast_enable.md) - Enable mixed precision
- [torch::tensor_backward](tensor_backward.md) - Compute gradients
- [Automatic Mixed Precision Training Guide](../guides/amp_training.md) 