# torch::layer_forward

Perform a forward pass through a neural network layer or module.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::layer_forward layer_handle input_tensor
```

### New Syntax (Named Parameters)
```tcl
torch::layer_forward -layer layer_handle -input input_tensor
```

### camelCase Alias
```tcl
torch::layerForward layer_handle input_tensor
torch::layerForward -layer layer_handle -input input_tensor
```

## Parameters

### Named Parameters
- **`-layer`** (string, required): Handle/name of the layer or module
- **`-input`** (string, required): Handle/name of the input tensor

### Positional Parameters  
- **`layer_handle`** (string, required): Handle/name of the layer or module
- **`input_tensor`** (string, required): Handle/name of the input tensor

## Description

The `torch::layer_forward` command applies a neural network layer or module to an input tensor, performing a forward pass through the network. This is the fundamental operation for neural network inference and training. 

The command supports various layer types including:
- **Linear layers** (fully connected)
- **Convolutional layers** (Conv2d)
- **Pooling layers** (MaxPool2d, AvgPool2d)
- **Normalization layers** (BatchNorm2d)
- **Regularization layers** (Dropout)
- **Container layers** (Sequential)

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

## Return Value

Returns a tensor handle containing the output of the forward pass. The output tensor shape depends on the layer type and input tensor dimensions.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a linear layer
set layer [torch::linear 784 128]

# Create input tensor (batch_size=32, features=784)
set input [torch::randn -shape {32 784}]

# Forward pass
set output [torch::layer_forward $layer $input]
puts "Output shape: [torch::tensor_shape $output]"  ;# Output: 32 128
```

### Named Parameter Syntax
```tcl
# Create a convolutional layer
set conv [torch::conv2d 3 16 3 1 1]  ;# 3->16 channels, 3x3 kernel

# Create input tensor (batch=1, channels=3, height=32, width=32)
set input [torch::randn -shape {1 3 32 32}]

# Forward pass using named parameters
set output [torch::layer_forward -layer $conv -input $input]

# Alternative parameter order
set output [torch::layer_forward -input $input -layer $conv]
```

### camelCase Alias
```tcl
# Create a layer
set layer [torch::linear 256 128]
set input [torch::randn -shape {16 256}]

# Forward pass using camelCase alias
set output [torch::layerForward $layer $input]
set output [torch::layerForward -layer $layer -input $input]
```

### Sequential Model Forward Pass
```tcl
# Create individual layers
set conv1 [torch::conv2d 3 16 3 1 1]
set pool1 [torch::maxpool2d 2]
set conv2 [torch::conv2d 16 32 3 1 1]
set pool2 [torch::maxpool2d 2]

# Create sequential model
set model [torch::sequential [list $conv1 $pool1 $conv2 $pool2]]

# Input tensor
set input [torch::randn -shape {1 3 32 32}]

# Forward pass through entire model
set output [torch::layer_forward $model $input]
puts "Final output shape: [torch::tensor_shape $output]"
```

### Multi-Layer Network
```tcl
# Create a multi-layer perceptron
set fc1 [torch::linear 784 512]
set fc2 [torch::linear 512 256]
set fc3 [torch::linear 256 10]

# Input: flattened MNIST image
set input [torch::randn -shape {64 784}]  ;# Batch of 64 images

# Forward pass through each layer
set hidden1 [torch::layer_forward $fc1 $input]
set hidden2 [torch::layer_forward $fc2 $hidden1]
set output [torch::layer_forward $fc3 $hidden2]

puts "Hidden1 shape: [torch::tensor_shape $hidden1]"  ;# 64 512
puts "Hidden2 shape: [torch::tensor_shape $hidden2]"  ;# 64 256
puts "Output shape: [torch::tensor_shape $output]"    ;# 64 10
```

### CNN for Image Classification
```tcl
# Create CNN layers
set conv1 [torch::conv2d 3 32 3 1 1]
set conv2 [torch::conv2d 32 64 3 1 1]
set pool [torch::maxpool2d 2]
set fc [torch::linear 12544 10]  ;# Calculated based on feature map size

# Input: RGB image batch
set input [torch::randn -shape {8 3 32 32}]  ;# Batch of 8 images

# Forward pass
set conv1_out [torch::layer_forward $conv1 $input]      ;# 8x32x32x32
set pool1_out [torch::layer_forward $pool $conv1_out]   ;# 8x32x16x16
set conv2_out [torch::layer_forward $conv2 $pool1_out]  ;# 8x64x16x16
set pool2_out [torch::layer_forward $pool $conv2_out]   ;# 8x64x8x8

# Flatten for fully connected layer
set flattened [torch::tensor_reshape $pool2_out {8 4096}]
set output [torch::layer_forward $fc $flattened]        ;# 8x10

puts "Classification output: [torch::tensor_shape $output]"
```

### Batch Processing with Different Sizes
```tcl
# Create layer
set layer [torch::linear 100 50]

# Process different batch sizes
set batch1 [torch::randn -shape {1 100}]   ;# Single sample
set batch32 [torch::randn -shape {32 100}] ;# Standard batch
set batch128 [torch::randn -shape {128 100}] ;# Large batch

set out1 [torch::layer_forward $layer $batch1]
set out32 [torch::layer_forward $layer $batch32]
set out128 [torch::layer_forward $layer $batch128]

puts "Batch 1 output: [torch::tensor_shape $out1]"    ;# 1 50
puts "Batch 32 output: [torch::tensor_shape $out32]"  ;# 32 50
puts "Batch 128 output: [torch::tensor_shape $out128]" ;# 128 50
```

### Feature Extraction Pipeline
```tcl
# Create feature extraction network
set encoder [torch::sequential [list \
    [torch::conv2d 3 64 7 2 3] \
    [torch::maxpool2d 3 2 1] \
    [torch::conv2d 64 128 3 1 1] \
    [torch::conv2d 128 256 3 1 1] \
    [torch::avgpool2d 7] \
]]

# Input images
set images [torch::randn -shape {16 3 224 224}]  ;# ImageNet-like input

# Extract features
set features [torch::layer_forward $encoder $images]
puts "Extracted features shape: [torch::tensor_shape $features]"  ;# 16x256x1x1
```

### Transfer Learning Example
```tcl
# Pre-trained feature extractor (frozen)
set feature_extractor [torch::sequential [list \
    [torch::conv2d 3 64 3 1 1] \
    [torch::conv2d 64 128 3 1 1] \
    [torch::avgpool2d 8] \
]]

# Custom classifier head (trainable)
set classifier [torch::linear 128 10]

# Forward pass
set input [torch::randn -shape {32 3 32 32}]
set features [torch::layer_forward $feature_extractor $input]
set flattened [torch::tensor_reshape $features {32 128}]
set predictions [torch::layer_forward $classifier $flattened]

puts "Predictions shape: [torch::tensor_shape $predictions]"  ;# 32 10
```

### Debugging Layer Outputs
```tcl
# Function to debug layer outputs
proc debug_forward_pass {layers input} {
    set current_input $input
    puts "Input shape: [torch::tensor_shape $current_input]"
    
    for {set i 0} {$i < [llength $layers]} {incr i} {
        set layer [lindex $layers $i]
        set output [torch::layer_forward $layer $current_input]
        puts "Layer $i output shape: [torch::tensor_shape $output]"
        set current_input $output
    }
    
    return $current_input
}

# Usage
set layers [list \
    [torch::linear 784 512] \
    [torch::linear 512 256] \
    [torch::linear 256 10] \
]
set input [torch::randn -shape {32 784}]
set final_output [debug_forward_pass $layers $input]
```

## Error Handling

The command performs comprehensive error checking:

```tcl
# Invalid layer handle
catch {torch::layer_forward "nonexistent_layer" $input} error
puts $error  ;# "Invalid layer name"

# Invalid input tensor handle
catch {torch::layer_forward $layer "nonexistent_tensor"} error
puts $error  ;# "Invalid input tensor name"

# Missing parameter value
catch {torch::layer_forward -layer} error
puts $error  ;# "Missing value for parameter"

# Unknown parameter
catch {torch::layer_forward -unknown_param value -layer $layer} error
puts $error  ;# "Unknown parameter: -unknown_param"

# Missing required parameters
catch {torch::layer_forward -layer ""} error
puts $error  ;# "Required parameters missing: layer and input"

# Unsupported module type
catch {torch::layer_forward $unsupported_layer $input} error
puts $error  ;# "Unsupported module type for forward pass"
```

## Supported Layer Types

### Linear Layers
```tcl
set linear [torch::linear 512 256]
set input [torch::randn -shape {32 512}]
set output [torch::layer_forward $linear $input]  ;# Shape: 32x256
```

### Convolutional Layers
```tcl
set conv [torch::conv2d 3 16 3 1 1]
set input [torch::randn -shape {1 3 32 32}]
set output [torch::layer_forward $conv $input]  ;# Shape: 1x16x32x32
```

### Pooling Layers
```tcl
set maxpool [torch::maxpool2d 2]
set avgpool [torch::avgpool2d 2]
set input [torch::randn -shape {1 16 32 32}]
set max_out [torch::layer_forward $maxpool $input]  ;# Shape: 1x16x16x16
set avg_out [torch::layer_forward $avgpool $input]  ;# Shape: 1x16x16x16
```

### Normalization Layers
```tcl
set batchnorm [torch::batchnorm2d 16]
set input [torch::randn -shape {32 16 32 32}]
set output [torch::layer_forward $batchnorm $input]  ;# Shape: 32x16x32x32
```

### Regularization Layers
```tcl
set dropout [torch::dropout 0.5]
set input [torch::randn -shape {32 512}]
set output [torch::layer_forward $dropout $input]  ;# Shape: 32x512 (with dropout applied)
```

### Sequential Containers
```tcl
set model [torch::sequential [list $conv1 $pool1 $conv2 $pool2]]
set input [torch::randn -shape {1 3 32 32}]
set output [torch::layer_forward $model $input]  ;# Passes through all layers
```

## Performance Considerations

### Memory Management
- Forward passes create new output tensors
- Large batch sizes increase memory usage
- Consider processing in smaller batches for memory-constrained environments

### Computational Efficiency
```tcl
# Efficient batch processing
set large_input [torch::randn -shape {1000 784}]

# Better: Process in batches
set batch_size 100
for {set i 0} {$i < 1000} {incr i $batch_size} {
    set end_idx [expr {min($i + $batch_size - 1, 999)}]
    set batch [torch::tensor_slice $large_input $i $end_idx]
    set batch_output [torch::layer_forward $layer $batch]
    # Process batch_output...
}
```

### GPU Acceleration
```tcl
# Move layers to GPU for acceleration
if {[torch::cuda_is_available]} {
    torch::layer_cuda $layer
    torch::layer_cuda $conv
    
    # Ensure input tensors are also on GPU
    set gpu_input [torch::tensor_cuda $input]
    set output [torch::layer_forward $layer $gpu_input]
}
```

## Input Shape Requirements

Different layer types have specific input shape requirements:

### Linear Layers
- **Input**: `[batch_size, in_features]`
- **Output**: `[batch_size, out_features]`

### Conv2d Layers
- **Input**: `[batch_size, in_channels, height, width]`
- **Output**: `[batch_size, out_channels, out_height, out_width]`

### Pooling Layers
- **Input**: `[batch_size, channels, height, width]`
- **Output**: `[batch_size, channels, pooled_height, pooled_width]`

### Shape Validation
```tcl
# Validate input shapes before forward pass
proc validate_linear_input {layer input} {
    set shape [torch::tensor_shape $input]
    set dims [llength [split $shape]]
    
    if {$dims != 2} {
        error "Linear layer requires 2D input (batch_size, features), got shape: $shape"
    }
}

# Usage
validate_linear_input $linear $input
set output [torch::layer_forward $linear $input]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set output [torch::layer_forward $layer $input]

# New named parameter syntax
set output [torch::layer_forward -layer $layer -input $input]
```

### From snake_case to camelCase

```tcl
# Old snake_case command
set output [torch::layer_forward $layer $input]

# New camelCase command
set output [torch::layerForward $layer $input]
```

## Integration with Training

### Forward Pass in Training Loop
```tcl
# Training loop example
set model [torch::sequential [list $conv $fc]]
set criterion [torch::loss_mse]
set optimizer [torch::optimizer_adam [torch::layer_parameters $model]]

for {set epoch 0} {$epoch < 10} {incr epoch} {
    # Forward pass
    set predictions [torch::layer_forward $model $train_input]
    
    # Compute loss
    set loss [torch::loss_forward $criterion $predictions $train_targets]
    
    # Backward pass and optimization
    torch::optimizer_zero_grad $optimizer
    torch::tensor_backward $loss
    torch::optimizer_step $optimizer
}
```

### Evaluation Mode
```tcl
# For evaluation, ensure layers are in eval mode
# Note: Training/eval mode switching depends on implementation
set predictions [torch::layer_forward $model $test_input]
```

## Best Practices

### Model Architecture Design
```tcl
# Good: Modular design with clear layer separation
set encoder [torch::sequential [list \
    [torch::conv2d 3 64 3 1 1] \
    [torch::batchnorm2d 64] \
    [torch::maxpool2d 2] \
]]

set decoder [torch::sequential [list \
    [torch::linear 1024 512] \
    [torch::linear 512 10] \
]]

# Forward pass through components
set features [torch::layer_forward $encoder $input]
set flattened [torch::tensor_reshape $features {-1 1024}]
set output [torch::layer_forward $decoder $flattened]
```

### Error Handling in Pipelines
```tcl
proc safe_forward_pass {layer input} {
    if {[catch {torch::layer_forward $layer $input} result]} {
        puts "Forward pass failed: $result"
        return ""
    }
    return $result
}

# Usage
set output [safe_forward_pass $layer $input]
if {$output ne ""} {
    # Continue processing
}
```

### Memory Monitoring
```tcl
# Monitor GPU memory usage during forward passes
if {[torch::cuda_is_available]} {
    set memory_before [torch::cuda_memory_allocated]
    set output [torch::layer_forward $large_model $input]
    set memory_after [torch::cuda_memory_allocated]
    
    puts "Memory used: [expr {$memory_after - $memory_before}] bytes"
}
```

## Compatibility

- **Backward Compatible**: All existing positional syntax continues to work
- **Forward Compatible**: New named parameter syntax is preferred for new code
- **Alias Support**: camelCase aliases provide modern API style
- **Error Handling**: Comprehensive validation with clear error messages
- **Cross-platform**: Works on all supported PyTorch platforms

## Version History

- **v1.0**: Original positional syntax implementation
- **v2.0**: Added dual syntax support with named parameters and camelCase aliases

## See Also

- [torch::linear](linear.md) - Linear layer creation
- [torch::conv2d](conv2d.md) - Convolutional layer creation
- [torch::sequential](sequential.md) - Sequential container
- [torch::layer_parameters](layer_parameters.md) - Get layer parameters
- [Neural Network Guide](../guides/neural_networks.md)
- [Training Workflow Guide](../guides/training.md)
- [Performance Optimization](../guides/performance.md) 