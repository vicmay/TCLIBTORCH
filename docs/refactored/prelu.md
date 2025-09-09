# torch::prelu

Applies the Parametric Rectified Linear Unit (PReLU) activation function element-wise to the input tensor. PReLU is a generalization of the traditional ReLU that allows negative values to have a non-zero slope.

## Syntax

### Current (Positional - Backward Compatible)
```tcl
torch::prelu input_tensor weight_tensor
```

### New (Named Parameters)
```tcl
torch::prelu -input input_tensor -weight weight_tensor
```

### camelCase Alias
```tcl
torch::prelu -input input_tensor -weight weight_tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `-input` | string | Handle to the input tensor |
| `-weight` | string | Handle to the weight tensor (learnable parameter) |

## Description

The Parametric ReLU function is defined as:

```
PReLU(x) = max(0, x) + weight * min(0, x)
```

Where:
- For positive input values: `PReLU(x) = x`
- For negative input values: `PReLU(x) = weight * x`
- `weight` is a learnable parameter

This is equivalent to:
```
PReLU(x) = x                if x >= 0
PReLU(x) = weight * x       if x < 0
```

### Key Features

1. **Learnable Parameter**: Unlike ReLU, PReLU has a learnable weight parameter that controls the slope for negative values
2. **Prevents Dead Neurons**: Allows gradient flow through negative values, preventing the "dying ReLU" problem
3. **Channel-specific**: Can have different weights for different channels
4. **Backward Compatibility**: Reduces to ReLU when weight = 0, Leaky ReLU when weight is fixed

### Weight Tensor Requirements

The weight tensor must be broadcastable with the input tensor:
- **Scalar weight**: Single weight value applied to all channels
- **Channel-wise weights**: One weight per channel (common in CNN layers)
- **Element-wise weights**: Same shape as input (rarely used)

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create input tensor and weight
set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
set weight [torch::tensor_create 0.2 float32]

# Apply PReLU
set result [torch::prelu $input $weight]
# Result: {1.0 -0.2 2.0 -0.4}
```

### Named Parameter Syntax
```tcl
# Create tensors
set input [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32]
set weight [torch::tensor_create 0.1 float32]

# Apply PReLU with named parameters
set result [torch::prelu -input $input -weight $weight]
```

### Parameter Order Flexibility
```tcl
# Parameters can be in any order
set result [torch::prelu -weight $weight -input $input]
```

### Channel-wise PReLU (CNN Applications)
```tcl
# Create 3-channel input (batch_size=1, channels=3, height=2, width=2)
set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0 3.0 -3.0 4.0 -4.0 5.0 -5.0 6.0 -6.0} float32] {1 3 2 2}]

# Different weight for each channel
set weight [torch::tensor_create {0.1 0.2 0.3} float32]

# Apply channel-wise PReLU
set result [torch::prelu -input $input -weight $weight]
```

### 2D Image Processing
```tcl
# Process a 2D image with PReLU
set image [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32] {2 2}]
set weight [torch::tensor_create 0.25 float32]
set activated [torch::prelu $image $weight]
```

### Neural Network Layer Implementation
```tcl
# In a neural network forward pass
proc prelu_layer {input weight} {
    return [torch::prelu -input $input -weight $weight]
}

# Usage in network
set input [torch::linear $previous_layer $weights $bias]
set activated [prelu_layer $input $prelu_weights]
```

## Mathematical Examples

### Example 1: Basic Scalar Weight
```tcl
set input [torch::tensor_create {3.0 -2.0 1.0 -4.0} float32]
set weight [torch::tensor_create 0.2 float32]
set result [torch::prelu $input $weight]
# Input:  [ 3.0, -2.0,  1.0, -4.0]
# Output: [ 3.0, -0.4,  1.0, -0.8]
# Computation: [max(0,3), 0.2*min(0,-2), max(0,1), 0.2*min(0,-4)]
```

### Example 2: Zero Weight (ReLU Behavior)
```tcl
set input [torch::tensor_create {2.0 -3.0 1.0 -1.0} float32]
set weight [torch::tensor_create 0.0 float32]
set result [torch::prelu $input $weight]
# Input:  [ 2.0, -3.0,  1.0, -1.0]
# Output: [ 2.0,  0.0,  1.0,  0.0]
# Behaves exactly like ReLU
```

### Example 3: Channel-wise Weights
```tcl
# 2 channels, each with different weight
set input [torch::tensor_reshape [torch::tensor_create {1.0 -1.0 2.0 -2.0} float32] {1 2 2}]
set weight [torch::tensor_create {0.1 0.3} float32]
set result [torch::prelu $input $weight]
# Channel 0: [1.0, -0.1] (weight=0.1)
# Channel 1: [2.0, -0.6] (weight=0.3)
```

## Use Cases

### 1. Convolutional Neural Networks
```tcl
# CNN layer with PReLU activation
proc conv_prelu_layer {input conv_weights conv_bias prelu_weights} {
    set conv_out [torch::conv2d $input $conv_weights $conv_bias]
    return [torch::prelu $conv_out $prelu_weights]
}

# In CNN forward pass
set conv1 [conv_prelu_layer $input $conv1_weights $conv1_bias $prelu1_weights]
```

### 2. Deep Residual Networks
```tcl
# Residual block with PReLU
proc residual_block {input weights1 bias1 weights2 bias2 prelu_weights} {
    set conv1 [torch::conv2d $input $weights1 $bias1]
    set prelu1 [torch::prelu $conv1 $prelu_weights]
    set conv2 [torch::conv2d $prelu1 $weights2 $bias2]
    return [torch::tensor_add $conv2 $input]  ;# Skip connection
}
```

### 3. Feature Extraction Networks
```tcl
# Feature extractor with PReLU activations
proc extract_features {input} {
    set layer1 [torch::linear $input $w1 $b1]
    set act1 [torch::prelu $layer1 $prelu_w1]
    
    set layer2 [torch::linear $act1 $w2 $b2]
    set act2 [torch::prelu $layer2 $prelu_w2]
    
    return $act2
}
```

### 4. Image Enhancement
```tcl
# Image processing with adaptive activation
proc enhance_image {image enhancement_weights} {
    set processed [some_image_processing $image]
    return [torch::prelu $processed $enhancement_weights]
}
```

## Training Considerations

### Weight Initialization
```tcl
# Initialize PReLU weights (typically small positive values)
set prelu_weights [torch::tensor_create 0.25 float32]  ;# Single weight
set channel_weights [torch::rand {64}]  ;# One per channel
set scaled_weights [torch::tensor_mul $channel_weights 0.1]  ;# Scale to small values
```

### Gradient Flow
- PReLU allows gradient flow through negative values
- Helps prevent vanishing gradients in deep networks
- Weight parameter is learnable and updated during backpropagation

### Comparison with Other Activations
```tcl
# ReLU (no negative slope)
set relu_result [torch::relu $input]

# Leaky ReLU (fixed negative slope)
set leaky_result [torch::leaky_relu $input 0.1]

# PReLU (learnable negative slope)
set prelu_result [torch::prelu $input $learnable_weight]

# ELU (exponential for negative values)
set elu_result [torch::elu $input 1.0]
```

## Error Handling

### Invalid Tensor Handles
```tcl
# Invalid input tensor
catch {torch::prelu "invalid_tensor" $weight} error
puts "Error: $error"
# Output: Invalid input tensor name

# Invalid weight tensor
catch {torch::prelu $input "invalid_weight"} error
puts "Error: $error"
# Output: Invalid weight tensor name
```

### Missing Parameters
```tcl
# Missing weight in positional syntax
catch {torch::prelu $input} error
puts "Error: $error"
# Output: Usage: torch::prelu tensor weight

# Missing parameters in named syntax
catch {torch::prelu -input $input} error
puts "Error: $error"
# Output: Required parameters missing (input and weight tensors required)
```

### Dimension Mismatch
```tcl
# Weight dimensions incompatible with input
set input [torch::tensor_create {1.0 -1.0} float32]  ;# 1D, 2 elements
set weight [torch::tensor_create {0.1 0.2 0.3} float32]  ;# 3 weights for 2-element input
catch {torch::prelu $input $weight} error
puts "Error: $error"
# Output: Mismatch of parameter numbers and input channel size
```

### Unknown Parameters
```tcl
# Invalid parameter name
catch {torch::prelu -input $input -weight $weight -slope 0.1} error
puts "Error: $error"
# Output: Unknown parameter: -slope
```

## Performance Considerations

### Memory Usage
- PReLU requires storing the weight tensor
- For channel-wise weights: minimal additional memory
- For element-wise weights: memory usage doubles

### Computational Cost
- Slightly more expensive than ReLU due to multiplication
- Much cheaper than exponential activations (ELU, Swish)
- Efficient implementation in PyTorch

### Numerical Stability
- Generally stable for reasonable weight values
- Avoid extremely large weight values that might cause overflow
- Consider weight clipping during training if necessary

## Integration with Optimizers

### Weight Updates
```tcl
# PReLU weights are typically included in optimizer parameters
set all_params [list $conv_weights $conv_bias $prelu_weights]
set optimizer [torch::optimizer_adam $all_params 0.001]

# During training
torch::optimizer_zero_grad $optimizer
# ... forward pass, loss computation, backward pass ...
torch::optimizer_step $optimizer
```

### Learning Rate Scheduling
```tcl
# Different learning rates for different parameter types
set conv_params [list $conv_weights $conv_bias]
set prelu_params [list $prelu_weights]

set conv_optimizer [torch::optimizer_adam $conv_params 0.001]
set prelu_optimizer [torch::optimizer_adam $prelu_params 0.0001]  ;# Lower LR for PReLU
```

## Return Value

Returns a string handle to the result tensor containing the PReLU-activated values. The output tensor has the same shape as the input tensor.

## Comparison with Related Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| `torch::relu` | `max(0, x)` | Basic activation, dead neurons possible |
| `torch::leaky_relu` | `max(0, x) + α*min(0, x)` | Fixed negative slope |
| `torch::prelu` | `max(0, x) + w*min(0, x)` | Learnable negative slope |
| `torch::elu` | `x if x≥0 else α*(e^x-1)` | Smooth negative values |
| `torch::selu` | Scaled ELU | Self-normalizing networks |

## Mathematical Properties

### Derivatives
- For x > 0: `∂PReLU/∂x = 1`
- For x < 0: `∂PReLU/∂x = weight`
- For x = 0: `∂PReLU/∂x = (1 + weight)/2` (subgradient)

### Monotonicity
- PReLU is monotonically increasing for any weight value
- Preserves ordering of input values

### Zero-centered
- Output is not zero-centered (like ReLU)
- Positive bias in activations may require batch normalization

## Best Practices

1. **Initialize weights small**: Start with values around 0.01-0.25
2. **Use channel-wise weights**: One weight per feature map in CNNs
3. **Monitor weight values**: Ensure they don't grow too large during training
4. **Combine with batch normalization**: Helps with training stability
5. **Consider placement**: Works well after convolutional and linear layers

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old positional syntax
set result [torch::prelu $input $weight]

# New named parameter syntax (equivalent)
set result [torch::prelu -input $input -weight $weight]

# Modern readable form
set result [torch::prelu -input $input -weight $weight]
```

### Advantages of Named Parameters
- **Clear parameter roles**: Explicitly shows input and weight tensors
- **Order independence**: Parameters can be in any sequence
- **Self-documenting**: Code intent is immediately clear
- **Future-proof**: Easy to add new parameters if needed

## See Also

- `torch::relu` - Basic ReLU activation
- `torch::leaky_relu` - ReLU with fixed negative slope
- `torch::elu` - Exponential Linear Unit
- `torch::selu` - Scaled Exponential Linear Unit
- `torch::gelu` - Gaussian Error Linear Unit
- `torch::swish` - Swish activation function
- `torch::linear` - Linear layer (commonly used before PReLU)
- `torch::conv2d` - 2D convolution (commonly used before PReLU) 