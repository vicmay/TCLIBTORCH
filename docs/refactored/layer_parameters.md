# torch::layer_parameters

Get list of trainable parameters from a neural network layer or module.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::layer_parameters -layer layer_name
torch::layerParameters -layer layer_name
```

### Positional Parameters (Legacy)
```tcl
torch::layer_parameters layer_name
torch::layerParameters layer_name
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-layer` | string | Yes | Name of the layer/module to extract parameters from |

## Returns

Returns a TCL list of tensor handles representing the trainable parameters of the layer.

## Description

The `torch::layer_parameters` command extracts all trainable parameters from a neural network layer or module. This is commonly used for:

- Optimizer parameter registration
- Parameter analysis and debugging
- Custom training loop implementations
- Parameter initialization and manipulation

Each parameter is returned as a tensor handle that can be used with other tensor operations.

## Examples

### Basic Usage
```tcl
# Create a linear layer
set layer [torch::linear -inFeatures 4 -outFeatures 2]

# Get parameters using named syntax (recommended)
set params [torch::layer_parameters -layer $layer]

# Get parameters using camelCase alias
set params [torch::layerParameters -layer $layer]

# Get parameters using legacy positional syntax
set params [torch::layer_parameters $layer]

# Display parameter count
puts "Layer has [llength $params] parameters"
```

### Parameter Analysis
```tcl
# Create a convolutional layer
set conv_layer [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]

# Get parameters
set params [torch::layer_parameters -layer $conv_layer]

# Analyze each parameter
foreach param $params {
    set shape [torch::tensor_shape $param]
    set dtype [torch::tensor_dtype $param]
    puts "Parameter: $param, Shape: $shape, Type: $dtype"
}
```

### Optimizer Registration
```tcl
# Create a neural network with multiple layers
set linear1 [torch::linear -inFeatures 784 -outFeatures 128]
set linear2 [torch::linear -inFeatures 128 -outFeatures 10]

# Collect all parameters
set all_params {}
lappend all_params {*}[torch::layer_parameters -layer $linear1]
lappend all_params {*}[torch::layer_parameters -layer $linear2]

# Create optimizer with all parameters
set optimizer [torch::optimizerAdam -parameters $all_params -lr 0.001]
```

### Parameter Manipulation
```tcl
# Create a layer
set layer [torch::linear -inFeatures 10 -outFeatures 5]

# Get parameters
set params [torch::layer_parameters -layer $layer]

# Move parameters to GPU
foreach param $params {
    set gpu_param [torch::tensor_to $param -device cuda]
    # Note: This creates new tensors, doesn't modify the original layer
}
```

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old syntax (still supported)
set params [torch::layer_parameters $layer]

# New syntax (recommended)
set params [torch::layer_parameters -layer $layer]

# Or using camelCase
set params [torch::layerParameters -layer $layer]
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Maintainability**: Code is self-documenting
- **Consistency**: Matches modern TCL conventions
- **Extensibility**: Easy to add new parameters in the future

## Error Handling

The command will throw an error in the following cases:

```tcl
# Invalid layer name
catch {torch::layer_parameters -layer "nonexistent"} error
puts "Error: $error"

# Missing required parameter
catch {torch::layer_parameters} error
puts "Error: $error"

# Unknown parameter
catch {torch::layer_parameters -unknown_param $layer} error
puts "Error: $error"
```

## Technical Details

- **Parameter Types**: Returns handles to actual parameter tensors
- **Memory**: Parameter tensors share memory with the original layer
- **Order**: Parameter order is consistent with PyTorch's `module.parameters()`
- **Types**: Typically includes weights and biases (if enabled)

## Common Layer Parameter Counts

| Layer Type | Typical Parameters |
|------------|-------------------|
| Linear | 2 (weight, bias) |
| Conv2d | 2 (weight, bias) |
| BatchNorm | 2 (weight, bias) |
| LSTM | 8+ (multiple weight matrices) |
| Sequential | Sum of all sublayers |

## Integration with Training Workflow

```tcl
# Complete training setup example
set model [torch::sequential]
set layer1 [torch::linear -inFeatures 784 -outFeatures 128]
set layer2 [torch::linear -inFeatures 128 -outFeatures 10]

# Add layers to model
torch::sequential_add $model $layer1
torch::sequential_add $model $layer2

# Get all parameters for optimizer
set all_params {}
lappend all_params {*}[torch::layer_parameters -layer $layer1]
lappend all_params {*}[torch::layer_parameters -layer $layer2]

# Create optimizer
set optimizer [torch::optimizerAdam -parameters $all_params -lr 0.001]

# Now ready for training loop
```

## See Also

- [torch::parameters_to](parameters_to.md) - Move parameters to device
- [torch::optimizerAdam](optimizer_adam.md) - Adam optimizer
- [torch::linear](linear.md) - Linear layer
- [torch::conv2d](conv2d.md) - 2D convolution layer

## Notes

- Parameter tensors are live references to the layer's internal parameters
- Modifying parameter tensors will affect the layer's behavior
- Use with caution in multi-threaded environments
- Parameter handles remain valid as long as the layer exists 