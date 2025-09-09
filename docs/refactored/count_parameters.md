# torch::count_parameters

Count the total number of parameters in a neural network model.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::count_parameters -model model_handle
torch::countParameters -model model_handle  # camelCase alias
```

### Positional Parameters (Legacy)
```tcl
torch::count_parameters model_handle
torch::countParameters model_handle  # camelCase alias
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-model` | string | Yes | Handle/name of the model to count parameters for |

## Returns

Returns an integer representing the total number of trainable parameters in the model.

## Description

The `torch::count_parameters` command counts all trainable parameters in a neural network model. This includes weights and biases from all layers in the model. The count represents the total number of individual scalar parameters that will be updated during training.

## Examples

### Basic Usage
```tcl
# Create a simple linear model
set model [torch::linear -inFeatures 10 -outFeatures 5 -bias true]

# Count parameters using named syntax (recommended)
set param_count [torch::count_parameters -model $model]
puts "Model has $param_count parameters"  # Output: Model has 55 parameters

# Count parameters using positional syntax (legacy)
set param_count [torch::count_parameters $model]
puts "Model has $param_count parameters"  # Output: Model has 55 parameters

# Using camelCase alias
set param_count [torch::countParameters -model $model]
puts "Model has $param_count parameters"  # Output: Model has 55 parameters
```

### Sequential Model
```tcl
# Create a multi-layer model
set layer1 [torch::linear -inFeatures 784 -outFeatures 128 -bias true]
set layer2 [torch::linear -inFeatures 128 -outFeatures 64 -bias true]
set layer3 [torch::linear -inFeatures 64 -outFeatures 10 -bias true]
set model [torch::sequential [list $layer1 $layer2 $layer3]]

# Count total parameters
set param_count [torch::count_parameters -model $model]
puts "Total parameters: $param_count"
# Output: Total parameters: 109386
# Calculation:
#   Layer 1: (784 * 128) + 128 = 100480
#   Layer 2: (128 * 64) + 64 = 8256  
#   Layer 3: (64 * 10) + 10 = 650
#   Total: 100480 + 8256 + 650 = 109386
```

### Model with Different Bias Settings
```tcl
# Model with bias
set model_with_bias [torch::linear -inFeatures 100 -outFeatures 50 -bias true]
set count_with_bias [torch::count_parameters -model $model_with_bias]
puts "With bias: $count_with_bias parameters"  # Output: With bias: 5050 parameters

# Model without bias
set model_no_bias [torch::linear -inFeatures 100 -outFeatures 50 -bias false]
set count_no_bias [torch::count_parameters -model $model_no_bias]
puts "Without bias: $count_no_bias parameters"  # Output: Without bias: 5000 parameters
```

### Convolutional Model
```tcl
# Create convolutional layers
set conv1 [torch::conv2d -inChannels 3 -outChannels 32 -kernelSize 3 -bias true]
set conv2 [torch::conv2d -inChannels 32 -outChannels 64 -kernelSize 3 -bias true]
set fc [torch::linear -inFeatures 1600 -outFeatures 10 -bias true]
set model [torch::sequential [list $conv1 $conv2 $fc]]

# Count parameters
set param_count [torch::count_parameters -model $model]
puts "CNN parameters: $param_count"
```

## Parameter Calculation

### Linear Layer
For a linear layer with input size `in_features` and output size `out_features`:
- **With bias**: `(in_features × out_features) + out_features`
- **Without bias**: `in_features × out_features`

### Convolutional Layer
For a 2D convolution with `in_channels`, `out_channels`, and kernel size `[H, W]`:
- **With bias**: `(in_channels × out_channels × H × W) + out_channels`
- **Without bias**: `in_channels × out_channels × H × W`

### Sequential Model
Sum of parameters from all constituent layers.

## Error Handling

### Invalid Model
```tcl
# Non-existent model
catch {torch::count_parameters -model "invalid_model"} error
puts $error  # Output: Model not found
```

### Missing Parameters
```tcl
# Missing required model parameter
catch {torch::count_parameters} error
puts $error  # Output: Required parameter missing: -model model_name

# Invalid parameter name
catch {torch::count_parameters -invalid_param model} error
puts $error  # Output: Unknown parameter: -invalid_param
```

## Migration Guide

### From Positional to Named Parameters

**Before (Positional)**:
```tcl
set count [torch::count_parameters $model]
```

**After (Named Parameters)**:
```tcl
set count [torch::count_parameters -model $model]
```

**Or using camelCase**:
```tcl
set count [torch::countParameters -model $model]
```

## Notes

- The command counts only trainable parameters (excludes buffers and non-trainable parameters)
- Empty models (sequential models with no layers) return 0 parameters
- The count includes all parameters regardless of their current `requires_grad` setting
- Performance is O(n) where n is the number of parameters in the model

## Related Commands

- [`torch::model_summary`](model_summary.md) - Display detailed model architecture
- [`torch::linear`](linear.md) - Create linear layers
- [`torch::sequential`](sequential.md) - Create sequential models
- [`torch::conv2d`](conv2d.md) - Create convolutional layers

## See Also

- [Model Architecture Documentation](../models.md)
- [Training Workflow Guide](../training.md)
- [Parameter Management](../parameters.md) 