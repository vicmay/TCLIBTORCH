# torch::conv2d_set_weights

## Overview

The `torch::conv2d_set_weights` command allows you to set the weights (and optionally bias) of a Conv2d layer after it has been created. This is useful for custom weight initialization, loading pre-trained weights, or implementing custom training procedures.

## Syntax

### Legacy Syntax (Backward Compatible)
```tcl
torch::conv2d_set_weights layer_handle weight_tensor ?bias_tensor?
```

### Modern Syntax (Named Parameters)
```tcl
torch::conv2d_set_weights -layer layer_handle -weight weight_tensor ?-bias bias_tensor?
```

### CamelCase Alias
```tcl
torch::conv2dSetWeights -layer layer_handle -weight weight_tensor ?-bias bias_tensor?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `layer` / `layer_handle` | string | Yes | Handle of the Conv2d layer |
| `weight` / `weight_tensor` | string | Yes | Handle of the weight tensor |
| `bias` / `bias_tensor` | string | No | Handle of the bias tensor (if layer has bias) |

## Parameter Details

### Layer Handle
- Must be a valid Conv2d layer created with `torch::conv2d`
- The command will verify that the layer is actually a Conv2d layer

### Weight Tensor
- Must be a 4D tensor with shape `[out_channels, in_channels, kernel_height, kernel_width]`
- Must match the dimensions of the Conv2d layer
- Will be copied into the layer's weight parameter

### Bias Tensor (Optional)
- Only used if the Conv2d layer was created with bias enabled
- Must be a 1D tensor with shape `[out_channels]`
- Will be copied into the layer's bias parameter if present

## Return Value

Returns "OK" on successful completion.

## Examples

### Basic Weight Setting

```tcl
# Create a Conv2d layer (3 input channels, 16 output channels, 3x3 kernel)
set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]

# Create custom weight tensor (16 x 3 x 3 x 3)
set custom_weights [torch::randn -shape {16 3 3 3}]

# Set weights using legacy syntax
torch::conv2d_set_weights $conv2d $custom_weights

# Or using modern named parameter syntax
torch::conv2d_set_weights -layer $conv2d -weight $custom_weights

# Or using camelCase alias
torch::conv2dSetWeights -layer $conv2d -weight $custom_weights
```

### Setting Both Weights and Bias

```tcl
# Create a Conv2d layer with bias
set conv2d [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -bias true]

# Create custom weight and bias tensors
set custom_weights [torch::randn -shape {16 3 3 3}]
set custom_bias [torch::zeros -shape {16}]

# Set both weights and bias using legacy syntax
torch::conv2d_set_weights $conv2d $custom_weights $custom_bias

# Or using modern named parameter syntax
torch::conv2d_set_weights -layer $conv2d -weight $custom_weights -bias $custom_bias

# Or using camelCase alias
torch::conv2dSetWeights -layer $conv2d -weight $custom_weights -bias $custom_bias
```

### Custom Initialization Example

```tcl
# Create a Conv2d layer
set conv2d [torch::conv2d -inChannels 1 -outChannels 32 -kernelSize 5]

# Initialize weights with Xavier/Glorot initialization
set fan_in 25    ;# 1 * 5 * 5
set fan_out 800  ;# 32 * 5 * 5
set limit [expr {sqrt(6.0 / ($fan_in + $fan_out))}]

# Create uniformly distributed weights in [-limit, limit]
set weights [torch::uniform -shape {32 1 5 5} -low [expr {-$limit}] -high $limit]

# Set the weights
torch::conv2dSetWeights -layer $conv2d -weight $weights

puts "Custom initialized Conv2d layer ready for training"
```

### Edge Detection Filter Example

```tcl
# Create a single-channel Conv2d for edge detection
set edge_detector [torch::conv2d -inChannels 1 -outChannels 1 -kernelSize 3 -bias false]

# Create Sobel edge detection kernel
set sobel_x [torch::tensor_create {
    {-1 0 1}
    {-2 0 2}
    {-1 0 1}
} float32]

# Reshape to proper 4D format [1, 1, 3, 3]
set sobel_4d [torch::unsqueeze [torch::unsqueeze $sobel_x 0] 0]

# Set the Sobel filter as weights
torch::conv2dSetWeights -layer $edge_detector -weight $sobel_4d

# Now the layer will perform Sobel edge detection
set input_image [torch::randn -shape {1 1 28 28}]
set edges [torch::layer_forward $edge_detector $input_image]

puts "Edge detection filter applied successfully"
```

## Error Handling

### Common Errors

```tcl
# Invalid layer handle
catch {torch::conv2dSetWeights -layer "invalid_handle" -weight $weights} error
puts "Error: $error"  ;# Output: Invalid layer name

# Invalid weight tensor handle  
catch {torch::conv2dSetWeights -layer $conv2d -weight "invalid_handle"} error
puts "Error: $error"  ;# Output: Invalid weight tensor name

# Missing required parameters
catch {torch::conv2dSetWeights -layer $conv2d} error
puts "Error: $error"  ;# Output: Required parameters: layer and weight

# Unknown parameter
catch {torch::conv2dSetWeights -layer $conv2d -weight $weights -unknown_param value} error
puts "Error: $error"  ;# Output: Unknown parameter: -unknown_param
```

### Dimension Validation

```tcl
# Weight tensor with wrong dimensions will cause runtime error
set wrong_weights [torch::randn -shape {10 3 3 3}]  ;# Wrong output channels

catch {torch::conv2dSetWeights -layer $conv2d -weight $wrong_weights} error
puts "Error: $error"  ;# Will show dimension mismatch error
```

## Performance Notes

- Weight copying is performed efficiently using LibTorch's tensor copy operations
- The operation is performed in-place on the layer's existing weight tensors
- Memory allocation is minimal as existing weight tensors are reused
- CUDA tensors are handled properly if the layer is on GPU

## Integration with Training

```tcl
# Example: Custom weight initialization before training
proc initialize_conv_layer {layer input_channels output_channels kernel_size} {
    # He initialization for ReLU networks
    set fan_in [expr {$input_channels * $kernel_size * $kernel_size}]
    set std [expr {sqrt(2.0 / $fan_in)}]
    
    set weights [torch::randn -shape [list $output_channels $input_channels $kernel_size $kernel_size]]
    set weights [torch::mul $weights $std]
    
    torch::conv2dSetWeights -layer $layer -weight $weights
    
    return $layer
}

# Usage
set conv1 [torch::conv2d -inChannels 3 -outChannels 64 -kernelSize 3]
initialize_conv_layer $conv1 3 64 3
```

## Compatibility

- **Backward Compatibility**: ✅ Full support for legacy positional syntax
- **Named Parameters**: ✅ Modern `-parameter value` syntax supported  
- **CamelCase**: ✅ `torch::conv2dSetWeights` alias available
- **Parameter Order**: ✅ Named parameters can be specified in any order

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# BEFORE (Legacy - still works)
torch::conv2d_set_weights $layer $weights $bias

# AFTER (Modern - recommended for new code)
torch::conv2dSetWeights -layer $layer -weight $weights -bias $bias
```

### Benefits of Modern Syntax
- **Explicit parameter names**: Clearer what each argument represents
- **Flexible parameter order**: Parameters can be specified in any order
- **Optional parameters**: Easier to specify only needed parameters
- **Better error messages**: Parameter validation provides clearer feedback
- **IDE support**: Better autocompletion and documentation

## See Also

- [`torch::conv2d`](conv2d.md) - Create Conv2d layers
- [`torch::layer_forward`](layer_forward.md) - Forward pass through layers
- [`torch::randn`](randn.md) - Generate random tensors for weight initialization
- [`torch::zeros`](zeros.md) - Create zero tensors for bias initialization
- [`torch::ones`](ones.md) - Create tensors filled with ones

## Technical Notes

- Uses LibTorch's `Tensor::copy_()` method for efficient weight copying
- Supports all tensor data types supported by LibTorch
- Thread-safe operation
- GPU memory management handled automatically
- Compatible with autograd for gradient computation 