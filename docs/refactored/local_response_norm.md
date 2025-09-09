# torch::local_response_norm

Apply local response normalization to input tensors.

## Syntax

### Current Syntax (Backward Compatible)
```tcl
torch::local_response_norm tensor size alpha beta k
```

### New Syntax (Named Parameters)
```tcl
torch::local_response_norm -input tensor -size size ?-alpha alpha? ?-beta beta? ?-k k?
```

### camelCase Alias
```tcl
torch::localResponseNorm tensor size alpha beta k
torch::localResponseNorm -input tensor -size size ?-alpha alpha? ?-beta beta? ?-k k?
```

## Parameters

### Named Parameters
- **`-input`** (tensor, required): Input tensor to apply local response normalization to
- **`-size`** (int, required): Size of the normalization window (neighborhood size)
- **`-alpha`** (double, optional): Alpha parameter for scaling (default: 1e-4)
- **`-beta`** (double, optional): Beta parameter for power (default: 0.75)
- **`-k`** (double, optional): K parameter for bias (default: 1.0)

### Positional Parameters  
- **`tensor`** (tensor, required): Input tensor to apply local response normalization to
- **`size`** (int, required): Size of the normalization window
- **`alpha`** (double, required): Alpha parameter for scaling
- **`beta`** (double, required): Beta parameter for power
- **`k`** (double, required): K parameter for bias

## Description

The `torch::local_response_norm` command applies local response normalization (LRN) to input tensors. Local response normalization is a technique used in convolutional neural networks to improve generalization by normalizing activations within local neighborhoods across feature maps.

The normalization is computed as:
```
output[i] = input[i] / (k + alpha * sum(input[j]^2 for j in neighborhood))^beta
```

Where:
- `i` is the current position
- `j` ranges over the local neighborhood of size `size`
- `k` is the bias parameter
- `alpha` is the scaling parameter
- `beta` is the exponential parameter

Key characteristics:
- **Local normalization**: Normalizes activations within a local neighborhood
- **Feature map competition**: Encourages competition between adjacent feature maps
- **Contrast enhancement**: Enhances contrast and reduces redundancy
- **Historical significance**: Originally popularized by AlexNet (2012)
- **Modern alternatives**: Largely replaced by batch normalization in contemporary architectures

The command supports both the original positional syntax for backward compatibility and the new named parameter syntax for improved readability and flexibility.

## Return Value

Returns a new tensor with the same shape as the input tensor, containing the local response normalized values.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create input tensor (typical CNN feature map)
set input [torch::randn -shape {1 64 32 32}]

# Apply local response normalization with default-like parameters
set output [torch::local_response_norm $input 5 0.0001 0.75 1.0]

# Apply with different size
set output [torch::local_response_norm $input 3 0.0001 0.75 1.0]
```

### Named Parameter Syntax
```tcl
# Create input tensor
set input [torch::randn -shape {1 64 32 32}]

# Apply local response normalization with named parameters
set output [torch::local_response_norm -input $input -size 5]

# Apply with custom parameters
set output [torch::local_response_norm -input $input -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]

# Alternative parameter order
set output [torch::local_response_norm -alpha 0.0001 -input $input -beta 0.75 -size 5 -k 1.0]
```

### camelCase Alias
```tcl
# Create input tensor
set input [torch::randn -shape {1 64 32 32}]

# Apply using camelCase alias
set output [torch::localResponseNorm $input 5 0.0001 0.75 1.0]
set output [torch::localResponseNorm -input $input -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
```

### AlexNet-style Usage
```tcl
# AlexNet-style local response normalization
# Typically applied after ReLU activation in convolutional layers

# Create conv layer output (simulated)
set conv_output [torch::randn -shape {8 96 55 55}]  ;# Batch=8, Channels=96, Height=55, Width=55

# Apply local response normalization (AlexNet parameters)
set lrn_output [torch::local_response_norm -input $conv_output -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
```

### Different Window Sizes
```tcl
# Create input tensor
set input [torch::randn -shape {4 128 16 16}]

# Small window (more local)
set output_small [torch::local_response_norm -input $input -size 3]

# Medium window (balanced)
set output_medium [torch::local_response_norm -input $input -size 5]

# Large window (more global)
set output_large [torch::local_response_norm -input $input -size 7]
```

### Different Alpha Values (Scaling)
```tcl
# Create input tensor
set input [torch::randn -shape {2 64 32 32}]

# Conservative scaling
set output_conservative [torch::local_response_norm -input $input -size 5 -alpha 0.0001]

# Moderate scaling
set output_moderate [torch::local_response_norm -input $input -size 5 -alpha 0.001]

# Aggressive scaling
set output_aggressive [torch::local_response_norm -input $input -size 5 -alpha 0.01]
```

### Different Beta Values (Power)
```tcl
# Create input tensor
set input [torch::randn -shape {2 64 32 32}]

# Lower power (less aggressive normalization)
set output_low [torch::local_response_norm -input $input -size 5 -beta 0.5]

# Standard power (typical value)
set output_standard [torch::local_response_norm -input $input -size 5 -beta 0.75]

# Higher power (more aggressive normalization)
set output_high [torch::local_response_norm -input $input -size 5 -beta 1.0]
```

### Different K Values (Bias)
```tcl
# Create input tensor
set input [torch::randn -shape {2 64 32 32}]

# Lower bias (more normalization effect)
set output_low_bias [torch::local_response_norm -input $input -size 5 -k 0.5]

# Standard bias (typical value)
set output_standard_bias [torch::local_response_norm -input $input -size 5 -k 1.0]

# Higher bias (less normalization effect)
set output_high_bias [torch::local_response_norm -input $input -size 5 -k 2.0]
```

### Integration with CNN Architectures
```tcl
# Simplified AlexNet-style architecture
proc alexnet_conv_block {input filters kernel_size stride padding} {
    # Convolution layer
    set conv_weight [torch::randn -shape [list $filters [expr {[lindex [torch::tensor_shape $input] 1]}] $kernel_size $kernel_size]]
    set conv_output [torch::conv2d -input $input -weight $conv_weight -stride $stride -padding $padding]
    
    # ReLU activation
    set relu_output [torch::relu -input $conv_output]
    
    # Local response normalization
    set lrn_output [torch::local_response_norm -input $relu_output -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
    
    # Max pooling
    set pool_output [torch::max_pool2d -input $lrn_output -kernelSize 3 -stride 2]
    
    return $pool_output
}

# Example usage in CNN
set input [torch::randn -shape {8 3 224 224}]  ;# ImageNet-like input
set conv1 [alexnet_conv_block $input 96 11 4 2]
set conv2 [alexnet_conv_block $conv1 256 5 1 2]
```

### Comparison with Other Normalization Methods
```tcl
# Create input tensor
set input [torch::randn -shape {8 64 32 32}]

# Local response normalization (original)
set lrn_output [torch::local_response_norm -input $input -size 5]

# For comparison (if available):
# Batch normalization (more common in modern architectures)
# set bn_output [torch::batch_norm2d -input $input -numFeatures 64]

# Layer normalization (for sequence models)
# set ln_output [torch::layer_norm -input $input -normalizedShape {64 32 32}]
```

### Research and Experimental Settings
```tcl
# Create input tensor
set input [torch::randn -shape {4 256 8 8}]

# Experimental parameter combinations
set exp1 [torch::local_response_norm -input $input -size 7 -alpha 0.0005 -beta 0.8 -k 1.5]
set exp2 [torch::local_response_norm -input $input -size 3 -alpha 0.0001 -beta 0.6 -k 0.8]
set exp3 [torch::local_response_norm -input $input -size 9 -alpha 0.001 -beta 0.9 -k 2.0]
```

### Batch Processing
```tcl
# Process different batch sizes
set batch_1 [torch::randn -shape {1 128 16 16}]
set batch_8 [torch::randn -shape {8 128 16 16}]
set batch_32 [torch::randn -shape {32 128 16 16}]

# Apply same normalization to all batches
set output_1 [torch::local_response_norm -input $batch_1 -size 5]
set output_8 [torch::local_response_norm -input $batch_8 -size 5]
set output_32 [torch::local_response_norm -input $batch_32 -size 5]
```

### Edge Cases and Validation
```tcl
# Create input tensor
set input [torch::randn -shape {2 32 16 16}]

# Minimum window size
set min_output [torch::local_response_norm -input $input -size 1]

# Large window size
set large_output [torch::local_response_norm -input $input -size 11]

# Very small alpha (minimal scaling)
set small_alpha [torch::local_response_norm -input $input -size 5 -alpha 1e-8]

# Large alpha (strong scaling)
set large_alpha [torch::local_response_norm -input $input -size 5 -alpha 1.0]
```

### Historical Context Usage
```tcl
# Original AlexNet paper parameters
set alexnet_input [torch::randn -shape {8 96 55 55}]
set alexnet_lrn [torch::local_response_norm -input $alexnet_input -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]

# GoogleNet (some layers used LRN)
set googlenet_input [torch::randn -shape {8 192 28 28}]
set googlenet_lrn [torch::local_response_norm -input $googlenet_input -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]
```

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old syntax (still supported)
set output [torch::local_response_norm $input 5 0.0001 0.75 1.0]

# New syntax (recommended)
set output [torch::local_response_norm -input $input -size 5 -alpha 0.0001 -beta 0.75 -k 1.0]

# Minimal named syntax (uses defaults)
set output [torch::local_response_norm -input $input -size 5]
```

### Benefits of Named Parameters
1. **Readability**: Parameter names make the code self-documenting
2. **Flexibility**: Parameters can be specified in any order
3. **Defaults**: Optional parameters can be omitted
4. **Maintainability**: Easier to modify and understand code
5. **Error prevention**: Less likely to mix up parameter positions

### Backward Compatibility
The original positional syntax is fully supported and will continue to work:
```tcl
# This will always work
set output [torch::local_response_norm $input 5 0.0001 0.75 1.0]
```

## Technical Notes

### Mathematical Formula
The local response normalization operation is defined as:
```
output[i] = input[i] / (k + alpha * sum(input[j]^2 for j in neighborhood))^beta
```

### Performance Considerations
- **Memory usage**: Creates temporary tensors for computation
- **Computational cost**: O(n * size) where n is the number of elements
- **GPU acceleration**: Fully supported on CUDA devices
- **Batch processing**: Efficient for multiple samples

### When to Use LRN
- **Historical architectures**: Reproducing AlexNet, early GoogleNet
- **Research**: Comparing with modern normalization methods
- **Specific applications**: When local contrast enhancement is needed
- **Ablation studies**: Understanding the effect of different normalizations

### Modern Alternatives
- **Batch Normalization**: More common in current architectures
- **Layer Normalization**: Better for sequence models
- **Group Normalization**: Better for small batch sizes
- **Instance Normalization**: Better for style transfer

## See Also

- `torch::batch_norm2d` - Batch normalization for 2D data
- `torch::layer_norm` - Layer normalization
- `torch::group_norm` - Group normalization
- `torch::instance_norm2d` - Instance normalization for 2D data
- `torch::cross_map_lrn2d` - Cross-map local response normalization 