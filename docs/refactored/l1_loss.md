# torch::l1_loss

Computes the L1 loss (Mean Absolute Error) between input and target tensors.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::l1_loss -input tensor -target tensor ?-reduction string?
torch::l1Loss -input tensor -target tensor ?-reduction string?  # camelCase alias
```

### Positional Parameters (Legacy)
```tcl
torch::l1_loss input target ?reduction?
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
  - Legacy integer values: `0` (none), `1` (mean), `2` (sum) - for backward compatibility

## Return Value

Returns a tensor handle containing the L1 loss value(s).

## Mathematical Definition

The L1 loss (Mean Absolute Error) is computed as:

- **Element-wise**: `loss(i) = |input[i] - target[i]|`
- **Mean reduction**: `L1 = mean(loss) = (1/N) * Σ|input[i] - target[i]|`
- **Sum reduction**: `L1 = sum(loss) = Σ|input[i] - target[i]|`
- **None reduction**: Returns unreduced loss tensor

## Examples

### Basic Usage
```tcl
# Create input and target tensors
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target [torch::tensor_create {1.1 1.9 3.2 3.8} float32]

# Named parameter syntax (recommended)
set loss [torch::l1_loss -input $input -target $target]
puts "L1 Loss: [torch::tensor_item $loss]"  ; # ~0.15

# CamelCase alias
set loss [torch::l1Loss -input $input -target $target]
puts "L1 Loss: [torch::tensor_item $loss]"  ; # ~0.15

# Positional syntax (legacy)
set loss [torch::l1_loss $input $target]
puts "L1 Loss: [torch::tensor_item $loss]"  ; # ~0.15
```

### Different Reduction Modes
```tcl
set input [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
set target [torch::tensor_create {2.0 3.0 4.0 5.0} float32]

# Mean reduction (default)
set loss_mean [torch::l1_loss -input $input -target $target -reduction mean]
puts "Mean loss: [torch::tensor_item $loss_mean]"  ; # 1.0

# Sum reduction
set loss_sum [torch::l1_loss -input $input -target $target -reduction sum]
puts "Sum loss: [torch::tensor_item $loss_sum]"    ; # 4.0

# No reduction (element-wise losses)
set loss_none [torch::l1_loss -input $input -target $target -reduction none]
puts "Loss shape: [torch::tensor_shape $loss_none]"  ; # {4}
puts "Individual losses: [torch::tensor_to_list $loss_none]"  ; # {1.0 1.0 1.0 1.0}
```

### Multi-dimensional Tensors
```tcl
# Create 2D tensors
set input [torch::tensor_create {1.0 2.0 3.0 4.0} {2 2} float32]
set target [torch::tensor_create {1.1 1.9 3.1 3.9} {2 2} float32]

set loss [torch::l1_loss -input $input -target $target]
puts "2D L1 Loss: [torch::tensor_item $loss]"
```

### Perfect Prediction (Zero Loss)
```tcl
set input [torch::tensor_create {1.0 2.0 3.0} float32]
set target [torch::tensor_create {1.0 2.0 3.0} float32]

set loss [torch::l1_loss -input $input -target $target]
puts "Perfect prediction loss: [torch::tensor_item $loss]"  ; # 0.0
```

### Backward Compatibility with Integer Reduction
```tcl
set input [torch::tensor_create {1.0 2.0} float32]
set target [torch::tensor_create {2.0 4.0} float32]

# Old integer reduction syntax still works
set loss_sum [torch::l1_loss $input $target 2]  ; # 2 = sum
puts "Sum loss: [torch::tensor_item $loss_sum]"  ; # 3.0
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing required parameter
catch {torch::l1_loss -input $input} result
puts $result  ; # "Required parameters -input and -target must be provided"

# Invalid parameter name
catch {torch::l1_loss -input $input -target $target -invalid value} result
puts $result  ; # "Unknown parameter: -invalid"

# Invalid tensor handle
catch {torch::l1_loss -input "invalid_tensor" -target $target} result
puts $result  ; # "Invalid input tensor name"
```

## Migration Guide

### From Positional to Named Parameters

**Old (Positional)**:
```tcl
set loss [torch::l1_loss $input $target]
set loss [torch::l1_loss $input $target "sum"]
set loss [torch::l1_loss $input $target 2]  ; # Integer reduction
```

**New (Named Parameters)**:
```tcl
set loss [torch::l1_loss -input $input -target $target]
set loss [torch::l1_loss -input $input -target $target -reduction sum]
set loss [torch::l1_loss -input $input -target $target -reduction sum]

# Or using camelCase alias
set loss [torch::l1Loss -input $input -target $target]
set loss [torch::l1Loss -input $input -target $target -reduction sum]
```

### Benefits of Named Parameters
- **Self-documenting**: Parameter names make code more readable
- **Flexible ordering**: Parameters can be specified in any order
- **Default handling**: Optional parameters can be omitted easily
- **Error prevention**: Reduces mistakes from parameter ordering
- **Future-proof**: Easy to add new optional parameters

## L1 vs L2 (MSE) Loss Comparison

```tcl
set input [torch::tensor_create {1.0 3.0} float32]
set target [torch::tensor_create {2.0 4.0} float32]

set l1_loss [torch::l1_loss -input $input -target $target]
set mse_loss [torch::mse_loss -input $input -target $target]

puts "L1 Loss:  [torch::tensor_item $l1_loss]"   ; # 1.0 (mean of |1| + |1|)
puts "MSE Loss: [torch::tensor_item $mse_loss]"  ; # 1.0 (mean of 1² + 1²)
```

### Key Differences:
- **L1 Loss (MAE)**: More robust to outliers, promotes sparsity
- **L2 Loss (MSE)**: Penalizes large errors more heavily, smoother gradients
- **L1 Gradient**: Constant magnitude, creates sparse solutions
- **L2 Gradient**: Proportional to error, smoother optimization

## Use Cases

1. **Robust Regression**: When outliers are present in the data
2. **Sparse Model Training**: Encourages sparse weight solutions
3. **Computer Vision**: Object detection bounding box regression
4. **Time Series**: Forecasting when outlier robustness is important
5. **Feature Selection**: L1 regularization for automatic feature selection

## Compatibility

- **Backward Compatibility**: ✅ All existing positional syntax code continues to work
- **Integer Reduction**: ✅ Legacy integer reduction values (0, 1, 2) still supported
- **Tensor Requirements**: Input and target tensors must have the same shape
- **Data Types**: Supports all floating-point tensor types (float16, float32, float64)
- **Devices**: Works with both CPU and CUDA tensors

## Performance Notes

- **Memory Efficient**: In-place operations where possible
- **GPU Accelerated**: Optimized for CUDA when available
- **Broadcasting**: Automatic shape alignment for compatible tensors
- **Gradient Ready**: Fully compatible with automatic differentiation
- **Robust**: Better numerical stability than squared losses for outliers

## Mathematical Properties

- **Convex**: L1 loss is a convex function
- **Non-differentiable**: At zero difference (subgradient exists)
- **Symmetric**: L1(a,b) = L1(b,a)
- **Translation Invariant**: L1(a+c, b+c) = L1(a,b)
- **Scale Invariant**: L1(ka, kb) = |k| * L1(a,b)

## Related Commands

- [`torch::mse_loss`](mse_loss.md) - Mean Squared Error (L2) loss
- [`torch::smooth_l1_loss`](smooth_l1_loss.md) - Huber loss (combines L1 and L2)
- [`torch::huber_loss`](huber_loss.md) - Huber loss with configurable delta
- [`torch::cross_entropy_loss`](cross_entropy_loss.md) - Classification loss
- [`torch::bce_loss`](bce_loss.md) - Binary classification loss

## See Also

- [Tensor Creation](../tensor_creation.md)
- [Loss Functions Overview](../loss_functions.md)
- [Training Pipeline](../training.md)
- [Regularization Techniques](../regularization.md) 