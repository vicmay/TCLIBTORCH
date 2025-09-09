# torch::layer_cpu

Move a neural network layer to CPU device.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::layer_cpu layer_name
```

### New Syntax (Named Parameters)
```tcl
torch::layer_cpu -layer layer_name
torch::layer_cpu -input layer_name  ;# alternative parameter name
```

### camelCase Alias
```tcl
torch::layerCpu layer_name
torch::layerCpu -layer layer_name
```

## Parameters

### Named Parameters
- **`-layer`** (string, required): Name/handle of the layer to move to CPU
- **`-input`** (string, required): Alternative parameter name for the layer (same as `-layer`)

### Positional Parameters  
- **`layer_name`** (string, required): Name/handle of the layer to move to CPU

## Description

The `torch::layer_cpu` command moves a neural network layer/module to the CPU device. This is useful for:

- Moving layers between devices (CUDA ↔ CPU)
- Reducing GPU memory usage
- Performing CPU-only computations
- Debugging and development workflows

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

## Return Value

Returns the layer name/handle that was moved to CPU, enabling command chaining.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a layer
set layer [torch::linear 128 64]

# Move to CPU
torch::layer_cpu $layer
```

### Named Parameter Syntax
```tcl
# Create a layer  
set layer [torch::linear 256 128]

# Move to CPU using named parameters
torch::layer_cpu -layer $layer

# Alternative parameter name
torch::layer_cpu -input $layer
```

### camelCase Alias
```tcl
# Create a layer
set layer [torch::linear 512 256]

# Move to CPU using camelCase alias
torch::layerCpu $layer
torch::layerCpu -layer $layer
```

### Command Chaining
```tcl
# Create layer and immediately move to CPU
set layer [torch::linear 784 256]
set result [torch::layer_cpu $layer]
puts "Layer $result moved to CPU"
```

### Device Management Workflow
```tcl
# Create a layer
set layer [torch::linear 1024 512]

# Check current device
set current_device [torch::layer_device $layer]
puts "Current device: $current_device"

# Move to CPU
torch::layer_cpu $layer

# Verify device change
set new_device [torch::layer_device $layer]
puts "New device: $new_device"
```

### Multi-Layer Movement
```tcl
# Create multiple layers
set layer1 [torch::linear 256 128]
set layer2 [torch::conv2d -inChannels 3 -outChannels 32 -kernelSize 3]
set layer3 [torch::batchnorm2d -numFeatures 32]

# Move all to CPU
torch::layer_cpu $layer1
torch::layer_cpu $layer2  
torch::layer_cpu $layer3

# Or using named parameters
torch::layer_cpu -layer $layer1
torch::layer_cpu -layer $layer2
torch::layer_cpu -layer $layer3
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Invalid layer name
catch {torch::layer_cpu "nonexistent_layer"} error
puts $error  ;# "Invalid layer name"

# Missing parameter value
catch {torch::layer_cpu -layer} error
puts $error  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::layer_cpu -unknown_param value} error
puts $error  ;# "Unknown parameter: -unknown_param"

# Empty layer name
catch {torch::layer_cpu -layer ""} error
puts $error  ;# "Required parameter missing: layer"
```

## Device Management Commands

Related commands for device management:

- **`torch::layer_cuda`** - Move layer to CUDA device
- **`torch::layer_device`** - Get current device of layer
- **`torch::layer_to`** - Move layer to specific device

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::layer_cpu $layer

# New named parameter syntax  
torch::layer_cpu -layer $layer
```

### From snake_case to camelCase

```tcl
# Old snake_case command
torch::layer_cpu $layer

# New camelCase command
torch::layerCpu $layer
```

## Performance Notes

- Moving layers between devices involves memory allocation and data copying
- CPU operations are generally slower than GPU operations for large layers
- Use device movement strategically based on computational requirements
- Consider memory constraints when moving large layers

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Forward Compatible**: New named parameter syntax is preferred for new code
- **Alias Support**: camelCase aliases provide modern API style
- **Error Handling**: Comprehensive validation with clear error messages

## Version History

- **v1.0**: Original positional syntax implementation
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::layer_cuda](layer_cuda.md) - Move layer to CUDA device
- [torch::layer_device](layer_device.md) - Get current device of layer  
- [torch::layer_to](layer_to.md) - Move layer to specific device
- [Device Management Guide](../guides/device_management.md)
- [Neural Network Layers](../guides/neural_layers.md) 