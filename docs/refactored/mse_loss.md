# torch::mse_loss

Computes the mean squared error (MSE) loss between input and target tensors.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::mse_loss -input tensor -target tensor ?-reduction string?
torch::mseLoss -input tensor -target tensor ?-reduction string?  # camelCase alias
```

### Positional Parameters (Legacy)
```tcl
torch::mse_loss input target ?reduction?
```

## Parameters

### Required Parameters
- **`-input tensor`** / **`input`**: Input tensor containing predicted values
- **`-target tensor`** / **`target`**: Target tensor containing ground truth values

### Optional Parameters
- **`-reduction string`** / **`reduction`**: Specifies the reduction to apply to the output
  - `"mean"` (default): Returns the mean of all losses
  - `"sum"`: Returns the sum of all losses  
  - `"none"`: No reduction, returns losses for each element

## Return Value

Returns a tensor handle containing the MSE loss value(s).

## Mathematical Definition

The mean squared error loss is computed as:

- **Element-wise**: `loss(i) = (input[i] - target[i])²`
- **Mean reduction**: `MSE = mean(loss) = (1/N) * Σ(input[i] - target[i])²`
- **Sum reduction**: `MSE = sum(loss) = Σ(input[i] - target[i])²`
- **None reduction**: Returns unreduced loss tensor

## Examples

### Basic Usage
```tcl
# Create input and target tensors
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target [torch::tensor_create {1.1 1.9 3.2 3.8} float32]

# Named parameter syntax (recommended)
set loss [torch::mse_loss -input $input -target $target]
puts "MSE Loss: [torch::tensor_item $loss]"  ; # ~0.025

# CamelCase alias
set loss [torch::mseLoss -input $input -target $target]
puts "MSE Loss: [torch::tensor_item $loss]"  ; # ~0.025

# Positional syntax (legacy)
set loss [torch::mse_loss $input $target]
puts "MSE Loss: [torch::tensor_item $loss]"  ; # ~0.025
```

### Different Reduction Modes
```tcl
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target [torch::tensor_create {2.0 3.0 4.0 5.0} float32]

# Mean reduction (default)
set loss_mean [torch::mse_loss -input $input -target $target -reduction mean]
puts "Mean loss: [torch::tensor_item $loss_mean]"  ; # 1.0

# Sum reduction
set loss_sum [torch::mse_loss -input $input -target $target -reduction sum]
puts "Sum loss: [torch::tensor_item $loss_sum]"    ; # 4.0

# No reduction (element-wise losses)
set loss_none [torch::mse_loss -input $input -target $target -reduction none]
puts "Loss shape: [torch::tensor_shape $loss_none]"  ; # {4}
puts "Individual losses: [torch::tensor_to_list $loss_none]"  ; # {1.0 1.0 1.0 1.0}
```

### Multi-dimensional Tensors
```tcl
# Create 2D tensors
set input [torch::tensor_create {1.0 2.0 3.0 4.0} {2 2} float32]
set target [torch::tensor_create {1.1 1.9 3.1 3.9} {2 2} float32]

set loss [torch::mse_loss -input $input -target $target]
puts "2D MSE Loss: [torch::tensor_item $loss]"
```

### Perfect Prediction (Zero Loss)
```tcl
set input [torch::tensor_create {1.0 2.0 3.0} float32]
set target [torch::tensor_create {1.0 2.0 3.0} float32]

set loss [torch::mse_loss -input $input -target $target]
puts "Perfect prediction loss: [torch::tensor_item $loss]"  ; # 0.0
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing required parameter
catch {torch::mse_loss -input $input} result
puts $result  ; # "Required parameters -input and -target must be provided"

# Invalid parameter name
catch {torch::mse_loss -input $input -target $target -invalid value} result
puts $result  ; # "Unknown parameter: -invalid"

# Invalid tensor handle
catch {torch::mse_loss -input "invalid_tensor" -target $target} result
puts $result  ; # "Invalid input tensor name"
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional)**:
```tcl
set loss [torch::mse_loss $input $target]
set loss [torch::mse_loss $input $target "sum"]
```

**New (Named Parameters)**:
```tcl
set loss [torch::mse_loss -input $input -target $target]
set loss [torch::mse_loss -input $input -target $target -reduction sum]

# Or using camelCase alias
set loss [torch::mseLoss -input $input -target $target]
set loss [torch::mseLoss -input $input -target $target -reduction sum]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Flexible ordering**: Parameters can be specified in any order
- **Default handling**: Optional parameters can be omitted easily
- **Error prevention**: Reduces mistakes from parameter ordering
- **Future-proof**: Easy to add new optional parameters

## Use Cases

1. **Regression Tasks**: Primary loss function for continuous value prediction
2. **Neural Network Training**: Backpropagation through MSE gradients
3. **Model Evaluation**: Measuring prediction accuracy
4. **Loss Monitoring**: Tracking training and validation loss
5. **Hyperparameter Tuning**: Comparing model configurations

## Compatibility

- **Backward Compatibility**: ✅ All existing positional syntax code continues to work
- **Tensor Requirements**: Input and target tensors must have the same shape
- **Data Types**: Supports all floating-point tensor types (float16, float32, float64)
- **Devices**: Works with both CPU and CUDA tensors

## Performance Notes

- **Memory Efficient**: In-place operations where possible
- **GPU Accelerated**: Optimized for CUDA when available
- **Broadcasting**: Automatic shape alignment for compatible tensors
- **Gradient Ready**: Fully compatible with automatic differentiation

## Related Commands

- [`torch::l1_loss`](l1_loss.md) - Mean Absolute Error loss
- [`torch::smooth_l1_loss`](smooth_l1_loss.md) - Huber loss  
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Classification loss
- [`torch::bce_loss`](bce_loss.md) - Binary classification loss

## See Also

- [Tensor Creation](../tensor_creation.md)
- [Loss Functions Overview](../loss_functions.md)
- [Training Pipeline](../training.md) 