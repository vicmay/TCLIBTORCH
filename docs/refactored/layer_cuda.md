# torch::layer_cuda

Move a neural network layer to CUDA device.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::layer_cuda layer_name
```

### New Syntax (Named Parameters)
```tcl
torch::layer_cuda -layer layer_name
torch::layer_cuda -input layer_name  ;# alternative parameter name
```

### camelCase Alias
```tcl
torch::layerCuda layer_name
torch::layerCuda -layer layer_name
```

## Parameters

### Named Parameters
- **`-layer`** (string, required): Name/handle of the layer to move to CUDA
- **`-input`** (string, required): Alternative parameter name for the layer (same as `-layer`)

### Positional Parameters  
- **`layer_name`** (string, required): Name/handle of the layer to move to CUDA

## Description

The `torch::layer_cuda` command moves a neural network layer/module to a CUDA device. This is useful for:

- GPU acceleration of neural network computations
- Moving layers from CPU to GPU for training
- Leveraging CUDA parallel processing capabilities
- Optimizing performance for large models

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

**Prerequisites**: CUDA must be available and properly configured on the system.

## Return Value

Returns the layer name/handle that was moved to CUDA, enabling command chaining.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a layer
set layer [torch::linear 128 64]

# Move to CUDA
torch::layer_cuda $layer
```

### Named Parameter Syntax
```tcl
# Create a layer  
set layer [torch::linear 256 128]

# Move to CUDA using named parameters
torch::layer_cuda -layer $layer

# Alternative parameter name
torch::layer_cuda -input $layer
```

### camelCase Alias
```tcl
# Create a layer
set layer [torch::linear 512 256]

# Move to CUDA using camelCase alias
torch::layerCuda $layer
torch::layerCuda -layer $layer
```

### Command Chaining
```tcl
# Create layer and immediately move to CUDA
set layer [torch::linear 784 256]
set result [torch::layer_cuda $layer]
puts "Layer $result moved to CUDA"
```

### GPU Training Workflow
```tcl
# Create a neural network
set layer1 [torch::linear 784 512]
set layer2 [torch::linear 512 256]
set layer3 [torch::linear 256 10]

# Move all layers to CUDA for GPU training
torch::layer_cuda $layer1
torch::layer_cuda $layer2
torch::layer_cuda $layer3

# Verify GPU placement
puts "Layer 1 device: [torch::layer_device $layer1]"
puts "Layer 2 device: [torch::layer_device $layer2]"
puts "Layer 3 device: [torch::layer_device $layer3]"
```

### Performance Optimization
```tcl
# Large model optimization
set large_layer [torch::linear 4096 4096]

# Check current device
set current_device [torch::layer_device $large_layer]
puts "Current device: $current_device"

# Move to CUDA for acceleration
torch::layer_cuda $large_layer

# Verify CUDA placement
set new_device [torch::layer_device $large_layer]
puts "New device: $new_device"
```

### Conditional CUDA Usage
```tcl
# Check CUDA availability before moving
if {[torch::cuda_is_available]} {
    set layer [torch::linear 1024 512]
    torch::layer_cuda $layer
    puts "Layer moved to CUDA"
} else {
    puts "CUDA not available, keeping on CPU"
}
```

### Mixed Device Training
```tcl
# Create multiple layers
set encoder [torch::linear 784 256]
set decoder [torch::linear 256 784]

# Move encoder to CUDA, keep decoder on CPU
torch::layer_cuda $encoder
torch::layer_cpu $decoder

puts "Encoder device: [torch::layer_device $encoder]"
puts "Decoder device: [torch::layer_device $decoder]"
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Invalid layer name
catch {torch::layer_cuda "nonexistent_layer"} error
puts $error  ;# "Invalid layer name"

# CUDA not available
catch {torch::layer_cuda $layer} error
puts $error  ;# "CUDA is not available" (if CUDA not installed)

# Missing parameter value
catch {torch::layer_cuda -layer} error
puts $error  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::layer_cuda -unknown_param value} error
puts $error  ;# "Unknown parameter: -unknown_param"

# Empty layer name
catch {torch::layer_cuda -layer ""} error
puts $error  ;# "Required parameter missing: layer"
```

## CUDA Requirements

### System Requirements
- NVIDIA GPU with CUDA compute capability 3.5 or higher
- CUDA toolkit installed and properly configured
- Compatible NVIDIA driver

### Checking CUDA Availability
```tcl
# Check if CUDA is available
if {[torch::cuda_is_available]} {
    puts "CUDA is available"
    puts "CUDA device count: [torch::cuda_device_count]"
} else {
    puts "CUDA is not available"
}
```

### Memory Considerations
- Moving layers to CUDA allocates GPU memory
- Large layers may require significant GPU memory
- Monitor GPU memory usage to avoid out-of-memory errors

## Device Management Commands

Related commands for device management:

- **`torch::layer_cpu`** - Move layer to CPU device
- **`torch::layer_device`** - Get current device of layer
- **`torch::layer_to`** - Move layer to specific device
- **`torch::cuda_is_available`** - Check CUDA availability
- **`torch::cuda_device_count`** - Get number of CUDA devices

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::layer_cuda $layer

# New named parameter syntax  
torch::layer_cuda -layer $layer
```

### From snake_case to camelCase

```tcl
# Old snake_case command
torch::layer_cuda $layer

# New camelCase command
torch::layerCuda $layer
```

## Performance Notes

- **GPU Acceleration**: CUDA operations are typically much faster for large tensors
- **Memory Transfer**: Moving data between CPU and GPU involves memory transfer overhead
- **Batch Processing**: GPU performance improves significantly with larger batch sizes
- **Model Size**: Larger models benefit more from GPU acceleration
- **Memory Management**: Monitor GPU memory usage to prevent out-of-memory errors

## Best Practices

### Device Strategy
```tcl
# Check CUDA availability first
if {[torch::cuda_is_available]} {
    # Move compute-intensive layers to CUDA
    torch::layer_cuda $conv_layers
    torch::layer_cuda $attention_layers
    
    # Keep lightweight layers on CPU if needed
    torch::layer_cpu $embedding_layer
}
```

### Memory Management
```tcl
# For very large models, move layers individually
torch::layer_cuda $layer1
# ... perform computations
torch::layer_cpu $layer1  ;# Free GPU memory

torch::layer_cuda $layer2
# ... continue processing
```

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Forward Compatible**: New named parameter syntax is preferred for new code
- **Alias Support**: camelCase aliases provide modern API style
- **Error Handling**: Comprehensive validation with clear error messages
- **Platform Support**: Works on all platforms with CUDA support

## Version History

- **v1.0**: Original positional syntax implementation
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::layer_cpu](layer_cpu.md) - Move layer to CPU device
- [torch::layer_device](layer_device.md) - Get current device of layer  
- [torch::layer_to](layer_to.md) - Move layer to specific device
- [CUDA Setup Guide](../guides/cuda_setup.md)
- [Device Management Guide](../guides/device_management.md)
- [Performance Optimization](../guides/performance.md) 