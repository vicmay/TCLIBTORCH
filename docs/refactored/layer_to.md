# torch::layer_to

Move a neural network layer or module to a specific computing device (CPU or CUDA).

## Syntax

### Named Parameters (Recommended)
```tcl
torch::layer_to -layer layer_name -device device_string
torch::layerTo -layer layer_name -device device_string
```

### Positional Parameters (Legacy)
```tcl
torch::layer_to layer_name device_string
torch::layerTo layer_name device_string
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-layer` | string | Yes | Name of the layer/module to move |
| `-device` | string | Yes | Target device ("cpu", "cuda", "cuda:0", etc.) |

## Returns

Returns the layer name (for chaining operations).

## Description

The `torch::layer_to` command moves a neural network layer or module to a specific computing device. This is essential for:

- **GPU Training**: Moving layers to CUDA devices for accelerated computation
- **Memory Management**: Moving layers between CPU and GPU to manage memory usage
- **Inference Optimization**: Placing layers on optimal devices for inference
- **Multi-GPU Training**: Distributing layers across multiple GPU devices

When a layer is moved to a device, all its parameters (weights, biases) are also moved to that device.

## Device Strings

| Device String | Description |
|---------------|-------------|
| `cpu` | CPU device |
| `cuda` | Default CUDA device (usually cuda:0) |
| `cuda:0` | CUDA device 0 |
| `cuda:1` | CUDA device 1 |
| `cuda:N` | CUDA device N |

## Examples

### Basic Usage
```tcl
# Create a linear layer
set layer [torch::linear -inFeatures 784 -outFeatures 128]

# Move to CPU using named syntax (recommended)
torch::layer_to -layer $layer -device cpu

# Move to CUDA using camelCase alias
torch::layerTo -layer $layer -device cuda

# Move to specific CUDA device using legacy positional syntax
torch::layer_to $layer cuda:0
```

### Training Setup
```tcl
# Create model layers
set conv1 [torch::conv2d -inChannels 3 -outChannels 32 -kernelSize 3]
set conv2 [torch::conv2d -inChannels 32 -outChannels 64 -kernelSize 3]
set fc1 [torch::linear -inFeatures 9216 -outFeatures 128]
set fc2 [torch::linear -inFeatures 128 -outFeatures 10]

# Move all layers to GPU for training
torch::layer_to -layer $conv1 -device cuda
torch::layer_to -layer $conv2 -device cuda  
torch::layer_to -layer $fc1 -device cuda
torch::layer_to -layer $fc2 -device cuda

puts "All layers moved to GPU for training"
```

### Memory Management
```tcl
# Create a large model
set large_model [torch::conv2d -inChannels 3 -outChannels 512 -kernelSize 7]

# Move to GPU for training
torch::layer_to -layer $large_model -device cuda

# ... training code ...

# Move back to CPU to free GPU memory
torch::layer_to -layer $large_model -device cpu
puts "Model moved back to CPU"
```

### Chaining Operations
```tcl
# Create a layer
set layer [torch::linear -inFeatures 100 -outFeatures 50]

# Chain operations: move to GPU, then verify location
set moved_layer [torch::layer_to -layer $layer -device cuda]
set device [torch::layer_device $moved_layer]
puts "Layer $moved_layer is now on device: $device"
```

### Sequential Models
```tcl
# Create individual layers
set linear1 [torch::linear -inFeatures 784 -outFeatures 256]
set linear2 [torch::linear -inFeatures 256 -outFeatures 128]
set linear3 [torch::linear -inFeatures 128 -outFeatures 10]

# Create sequential model
set model [torch::sequential [list $linear1 $linear2 $linear3]]

# Move entire sequential model to GPU
torch::layer_to -layer $model -device cuda

# Verify device placement
set device [torch::layer_device $model]
puts "Sequential model is on device: $device"
```

### Multi-GPU Setup
```tcl
# Create multiple layers for distributed training
set encoder [torch::conv2d -inChannels 3 -outChannels 64 -kernelSize 3]
set decoder [torch::conv2d -inChannels 64 -outChannels 3 -kernelSize 3]

# Distribute across multiple GPUs
torch::layer_to -layer $encoder -device cuda:0
torch::layer_to -layer $decoder -device cuda:1

puts "Encoder on GPU 0, Decoder on GPU 1"
```

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old syntax (still supported)
torch::layer_to $layer cuda

# New syntax (recommended)
torch::layer_to -layer $layer -device cuda

# Or using camelCase
torch::layerTo -layer $layer -device cuda
```

### Benefits of Named Parameters
- **Clarity**: Parameter purpose is explicit
- **Maintainability**: Code is self-documenting
- **Consistency**: Matches modern TCL conventions
- **Flexibility**: Parameter order doesn't matter
- **Error Prevention**: Reduces parameter mix-ups

## Error Handling

The command will throw an error in the following cases:

```tcl
# Invalid layer name
catch {torch::layer_to -layer "nonexistent" -device cpu} error
puts "Error: $error"

# Invalid device string
catch {torch::layer_to -layer $layer -device "invalid_device"} error
puts "Error: $error"

# Missing required parameters
catch {torch::layer_to -layer $layer} error
puts "Error: $error"

# Unknown parameter
catch {torch::layer_to -unknown_param $layer -device cpu} error
puts "Error: $error"
```

## Device Compatibility

### CPU Device
- Always available
- Slower computation but unlimited memory
- Compatible with all layer types

### CUDA Device
- Requires CUDA-capable GPU
- Faster computation but limited memory
- Check availability with `torch::cuda_is_available`

```tcl
# Check CUDA availability before moving to GPU
if {[torch::cuda_is_available]} {
    torch::layer_to -layer $layer -device cuda
    puts "Layer moved to CUDA"
} else {
    torch::layer_to -layer $layer -device cpu
    puts "CUDA not available, using CPU"
}
```

## Performance Considerations

### Memory Usage
- Moving layers to GPU consumes GPU memory
- Large models may exceed GPU memory limits
- Consider batch processing for large models

### Transfer Overhead
- Moving data between CPU and GPU has overhead
- Minimize device transfers during training
- Keep related operations on the same device

### Best Practices
```tcl
# Good: Move model once, keep on GPU
torch::layer_to -layer $model -device cuda
# ... perform multiple operations ...

# Bad: Frequent device transfers
torch::layer_to -layer $model -device cpu
torch::layer_to -layer $model -device cuda
torch::layer_to -layer $model -device cpu
```

## Integration with Training

### Complete Training Setup
```tcl
# Create model
set model [torch::linear -inFeatures 784 -outFeatures 10]

# Move to GPU
torch::layer_to -layer $model -device cuda

# Get parameters for optimizer
set params [torch::layer_parameters -layer $model]

# Move parameters to same device
torch::parameters_to $params cuda

# Create optimizer
set optimizer [torch::optimizerAdam -parameters $params -lr 0.001]

# Now ready for GPU training
puts "Training setup complete on GPU"
```

### Inference Optimization
```tcl
# Load trained model
set model [torch::load_model "trained_model.pt"]

# Move to optimal device for inference
if {[torch::cuda_is_available]} {
    torch::layer_to -layer $model -device cuda
    puts "Using GPU for inference"
} else {
    torch::layer_to -layer $model -device cpu
    puts "Using CPU for inference"
}

# Set to evaluation mode
torch::model_eval $model
```

## See Also

- [torch::layer_device](layer_device.md) - Get current device of layer
- [torch::layer_cuda](layer_cuda.md) - Move layer to CUDA
- [torch::layer_cpu](layer_cpu.md) - Move layer to CPU
- [torch::parameters_to](parameters_to.md) - Move parameters to device
- [torch::cuda_is_available](cuda_is_available.md) - Check CUDA availability

## Notes

- Device movement affects all layer parameters
- Operations on tensors must be on the same device
- Moving large models may take time
- GPU memory is limited compared to CPU memory
- Device transfers are synchronous operations 