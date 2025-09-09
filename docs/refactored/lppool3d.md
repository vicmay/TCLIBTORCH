# torch::lppool3d Command Reference

## Overview
The `torch::lppool3d` command performs 3D Lp pooling operations on input tensors. This is useful for 3D data such as volumetric images, video data, or 3D feature maps in neural networks.

## Syntax

### Original Syntax (Backward Compatible)
```tcl
torch::lppool3d input norm_type kernel_size ?stride? ?ceil_mode?
```

### Modern Named Parameter Syntax
```tcl
torch::lppool3d -input input -normType norm_type -kernelSize kernel_size ?-stride stride? ?-ceilMode ceil_mode?
```

### CamelCase Alias
```tcl
torch::lpPool3d [same parameters as above]
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input`/`-tensor` | string | Yes | - | Name of input tensor (5D: N×C×D×H×W or 4D: C×D×H×W) |
| `-normType`/`-norm_type` | double | Yes | - | Lp norm type (p > 0, e.g., 1.0 for L1, 2.0 for L2) |
| `-kernelSize`/`-kernel_size` | int or list | Yes | - | Size of pooling kernel (int or {d h w}) |
| `-stride` | int or list | No | kernelSize | Stride for pooling operation (int or {d h w}) |
| `-ceilMode`/`-ceil_mode` | bool | No | false | Use ceiling instead of floor for output size |

## Mathematical Background

### Lp Pooling Formula
For a pooling region R, the Lp pooling operation computes:

```
output = (Σ |x_i|^p)^(1/p)
```

Where:
- p is the norm type (`normType`)
- x_i are the elements in the pooling region
- The sum is taken over all elements in the kernel window

### Common Norm Types
- **L1 norm (p=1)**: Sum of absolute values
- **L2 norm (p=2)**: Euclidean norm (root mean square)
- **L∞ norm (p→∞)**: Maximum value (approaches max pooling)

### Output Size Calculation
```
output_size = floor((input_size + 2*padding - kernel_size) / stride) + 1
```

With `ceil_mode=true`:
```
output_size = ceil((input_size + 2*padding - kernel_size) / stride) + 1
```

## Examples

### Basic Usage
```tcl
# Create a 3D volume (batch=1, channels=1, depth=4, height=4, width=4)
set volume [torch::zeros {1 1 4 4 4} float32]

# L2 norm pooling with 2×2×2 kernel
set result [torch::lppool3d $volume 2.0 2]
puts "Output shape: [torch::tensor_shape $result]"  ;# {1 1 2 2 2}
```

### Named Parameter Syntax
```tcl
# Same operation with named parameters
set result [torch::lppool3d -input $volume -normType 2.0 -kernelSize 2]

# With all parameters specified
set result [torch::lppool3d \
    -input $volume \
    -normType 1.0 \
    -kernelSize {2 2 2} \
    -stride {1 1 1} \
    -ceilMode false]
```

### Different Norm Types
```tcl
set volume [torch::ones {1 1 2 2 2} float32]

# L1 norm (sum of absolute values)
set l1_result [torch::lppool3d $volume 1.0 2]

# L2 norm (Euclidean norm)
set l2_result [torch::lppool3d $volume 2.0 2]

# Large p approaches max pooling
set max_like [torch::lppool3d $volume 100.0 2]
```

### Non-uniform Kernels and Strides
```tcl
# Different kernel sizes for each dimension
set result [torch::lppool3d $volume 2.0 {1 2 2}]

# Custom stride
set result [torch::lppool3d $volume 2.0 {2 2 2} {1 1 1}]

# Using named parameters
set result [torch::lppool3d \
    -input $volume \
    -normType 2.0 \
    -kernelSize {2 3 3} \
    -stride {1 2 2}]
```

### Multi-channel and Batch Processing
```tcl
# Multi-channel volume (2 channels)
set multi_channel [torch::zeros {1 2 4 4 4} float32]
set result [torch::lppool3d $multi_channel 2.0 2]
puts "Shape: [torch::tensor_shape $result]"  ;# {1 2 2 2 2}

# Batch processing (3 samples)
set batch [torch::zeros {3 1 4 4 4} float32]
set result [torch::lppool3d $batch 2.0 2]
puts "Shape: [torch::tensor_shape $result]"  ;# {3 1 2 2 2}
```

## Use Cases

### Computer Vision and 3D Analysis
```tcl
# Medical imaging: CT scan volume analysis
set ct_scan [torch::zeros {1 1 64 64 64} float32]
set downsampled [torch::lppool3d $ct_scan 2.0 4 2]  ;# Reduce by factor of 2

# Feature extraction with different norms
set features_l1 [torch::lppool3d $ct_scan 1.0 3]    ;# Sparse features
set features_l2 [torch::lppool3d $ct_scan 2.0 3]    ;# Smooth features
```

### Video Processing
```tcl
# Video frames as 3D tensor (time, height, width)
set video_frames [torch::zeros {1 3 30 128 128} float32]  ;# 30 frames

# Temporal and spatial pooling
set pooled_video [torch::lppool3d $video_frames 2.0 {2 4 4}]
puts "Pooled shape: [torch::tensor_shape $pooled_video]"  ;# {1 3 15 32 32}
```

### Neural Network Layers
```tcl
# 3D CNN feature maps
set feature_maps [torch::zeros {8 32 16 16 16} float32]  ;# Batch of 8

# Progressive downsampling
set stage1 [torch::lppool3d $feature_maps 2.0 2]        ;# 8×32×8×8×8
set stage2 [torch::lppool3d $stage1 2.0 2]              ;# 8×32×4×4×4
```

## Performance Considerations

### Memory Usage
```tcl
# Large volumes require careful memory management
set large_volume [torch::zeros {1 1 128 128 128} float32]

# Use larger strides to reduce memory usage
set efficient [torch::lppool3d $large_volume 2.0 4 4]  ;# Stride = kernel_size
```

### Norm Type Selection
- **L1 norm**: Fastest computation, promotes sparsity
- **L2 norm**: Standard choice, good for most applications
- **Higher norms**: Approach max pooling behavior, more computation

### Kernel Size Guidelines
- Small kernels (2×2×2): Preserve fine details
- Medium kernels (3×3×3 to 5×5×5): Balance detail and computation
- Large kernels (>5): Significant downsampling, global features

## Error Handling

The command provides clear error messages for common issues:

```tcl
# Missing required parameters
catch {torch::lppool3d -input $tensor} error
puts $error  ;# "Required parameters missing or invalid"

# Invalid norm type
catch {torch::lppool3d $tensor 0.0 2} error
puts $error  ;# "Required parameters missing or invalid"

# Invalid tensor
catch {torch::lppool3d nonexistent 2.0 2} error
puts $error  ;# "Invalid input tensor name"

# Unknown parameter
catch {torch::lppool3d -input $tensor -unknown value} error
puts $error  ;# "Unknown parameter: -unknown"
```

## Migration Guide

### From Positional to Named Parameters

**Old Style:**
```tcl
set result [torch::lppool3d $input 2.0 {2 2 2} {1 1 1} true]
```

**New Style:**
```tcl
set result [torch::lppool3d \
    -input $input \
    -normType 2.0 \
    -kernelSize {2 2 2} \
    -stride {1 1 1} \
    -ceilMode true]
```

### Benefits of Named Parameters
1. **Self-documenting**: Parameter names make code more readable
2. **Flexible ordering**: Parameters can be specified in any order
3. **Optional parameters**: Easy to omit optional parameters
4. **Error prevention**: Reduces parameter position mistakes

## Technical Notes

### Input Tensor Requirements
- **5D tensors**: (N, C, D, H, W) - batch, channels, depth, height, width
- **4D tensors**: (C, D, H, W) - channels, depth, height, width
- Minimum size: kernel_size ≤ input_size for each dimension

### Data Type Support
- Works with all floating-point types (float32, float64)
- GPU tensors supported if CUDA is available

### Memory Layout
- Input tensors should be contiguous for optimal performance
- Use `torch::tensor_contiguous` if needed

## Related Commands
- `torch::lppool1d` - 1D Lp pooling
- `torch::lppool2d` - 2D Lp pooling  
- `torch::maxpool3d` - 3D max pooling
- `torch::avgpool3d` - 3D average pooling

## See Also
- [LibTorch Pooling Documentation](https://pytorch.org/docs/stable/nn.functional.html#pooling-functions)
- [3D Pooling in Neural Networks](https://arxiv.org/abs/1506.06204)
- [Lp Pooling Theory](https://papers.nips.cc/paper/4761-generalizing-pooling-functions-in-convolutional-neural-networks-mixed-gated-and-tree.pdf) 