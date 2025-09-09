# torch::denormalize_image

Reverses image normalization by multiplying by standard deviation and adding the mean.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::denormalize_image image mean std ?inplace?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::denormalize_image -image image -mean mean -std std ?-inplace inplace?
```

### CamelCase Alias
```tcl
torch::denormalizeImage ...
```

## Parameters

### Required Parameters
- **image** (tensor): Input image tensor to denormalize
- **mean** (tensor): Mean values used in original normalization (tensor should be broadcastable with image)
- **std** (tensor): Standard deviation values used in original normalization (tensor should be broadcastable with image)

### Optional Parameters
- **inplace** (boolean, default: false): If true, modifies the input tensor in-place; if false, creates a new tensor

## Returns

Returns a tensor handle containing the denormalized image.

## Description

The `torch::denormalize_image` command reverses the normalization process typically applied to images in deep learning preprocessing. It performs the inverse operation of `torch::normalize_image`.

The mathematical operation performed is:
- **Non-inplace**: `output = (image * std) + mean`
- **Inplace**: `image = (image * std) + mean`, then returns the modified input tensor

This is commonly used to convert normalized images back to their original pixel value ranges for visualization or further processing.

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Create test data
set image [torch::tensor_create -data {-0.5 0.0 0.5 1.0} -shape {2 2} -dtype float32]
set mean [torch::tensor_create -data {0.485} -shape {1} -dtype float32]
set std [torch::tensor_create -data {0.229} -shape {1} -dtype float32]

# Denormalize using positional syntax
set denormalized [torch::denormalize_image $image $mean $std]
```

#### Named Parameter Syntax
```tcl
# Denormalize using named parameters
set denormalized [torch::denormalize_image -image $image -mean $mean -std $std]
```

#### CamelCase Alias
```tcl
# Using camelCase alias
set denormalized [torch::denormalizeImage -image $image -mean $mean -std $std]
```

### Multi-Channel Images

```tcl
# Create a 3-channel RGB image (C, H, W format)
set rgb_image [torch::tensor_create -data {0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2} -shape {3 2 2} -dtype float32]

# ImageNet normalization parameters (broadcast-compatible shapes)
set imagenet_mean [torch::tensor_create -data {0.485 0.456 0.406} -shape {3 1 1} -dtype float32]
set imagenet_std [torch::tensor_create -data {0.229 0.224 0.225} -shape {3 1 1} -dtype float32]

# Denormalize
set denormalized [torch::denormalize_image -image $rgb_image -mean $imagenet_mean -std $imagenet_std]
```

### Inplace Operations

```tcl
# Denormalize in-place (modifies original tensor)
set image [torch::tensor_create -data {-1.0 -0.5 0.0 0.5} -shape {2 2} -dtype float32]
set mean [torch::tensor_create -data {0.5} -shape {1} -dtype float32]
set std [torch::tensor_create -data {0.25} -shape {1} -dtype float32]

# Positional syntax with inplace
set result [torch::denormalize_image $image $mean $std 1]

# Named parameter syntax with inplace
set result [torch::denormalize_image -image $image -mean $mean -std $std -inplace 1]
```

### Grayscale Images

```tcl
# Single channel grayscale image
set gray_image [torch::tensor_create -data {-0.2 0.0 0.2 0.4} -shape {1 2 2} -dtype float32]
set mean [torch::tensor_create -data {0.5} -shape {1} -dtype float32]
set std [torch::tensor_create -data {0.3} -shape {1} -dtype float32]

# Denormalize grayscale image
set denormalized [torch::denormalize_image -image $gray_image -mean $mean -std $std]
```

### Batch Processing

```tcl
# Process batch of images
set batch_images [torch::tensor_create -data {/* ... batch data ... */} -shape {8 3 224 224} -dtype float32]
set batch_mean [torch::tensor_create -data {0.485 0.456 0.406} -shape {3 1 1} -dtype float32]
set batch_std [torch::tensor_create -data {0.229 0.224 0.225} -shape {3 1 1} -dtype float32]

# Denormalize entire batch
set denormalized_batch [torch::denormalize_image -image $batch_images -mean $batch_mean -std $batch_std]
```

## Broadcasting Rules

The `mean` and `std` tensors must be broadcastable with the `image` tensor according to PyTorch's broadcasting rules:

- **Single value**: `{1}` broadcasts to any shape
- **Channel-wise**: For RGB images with shape `{3, H, W}`, use mean/std shapes `{3, 1, 1}`
- **Compatible dimensions**: Trailing dimensions of size 1 are automatically broadcast

## Common Use Cases

### Image Visualization
```tcl
# Typical workflow: normalize → process → denormalize → display
set original_image [torch::tensor_create -data {/* image data */} -shape {3 224 224} -dtype float32]

# Normalize for model input
set normalized [torch::normalize_image $original_image $mean $std]

# ... process with neural network ...

# Denormalize for visualization
set display_image [torch::denormalize_image $processed_output $mean $std]
```

### Preprocessing Pipeline Reversal
```tcl
# Reverse standard ImageNet preprocessing
set imagenet_mean [torch::tensor_create -data {0.485 0.456 0.406} -shape {3 1 1} -dtype float32]
set imagenet_std [torch::tensor_create -data {0.229 0.224 0.225} -shape {3 1 1} -dtype float32]

# Denormalize model outputs or intermediate features
set denormalized [torch::denormalize_image -image $normalized_tensor -mean $imagenet_mean -std $imagenet_std]
```

## Error Handling

The command performs validation and provides clear error messages:

```tcl
# Invalid tensor names
catch {torch::denormalize_image invalid_tensor $mean $std} error
# Error: "Invalid image tensor"

# Missing required parameters
catch {torch::denormalize_image -image $image} error
# Error: "Required parameters missing: -mean, -std"

# Invalid parameter names
catch {torch::denormalize_image -invalid_param $image -mean $mean -std $std} error
# Error: "Unknown parameter: -invalid_param"

# Shape incompatibility (handled by PyTorch)
catch {torch::denormalize_image $incompatible_image $mean $std} error
# Error: Broadcasting error message from PyTorch
```

## Data Type Support

The command supports all floating-point tensor types:
- **float32** (Float)
- **float64** (Double)

Integer tensors will be promoted to floating-point for mathematical operations.

## Performance Notes

- **Inplace operations** (`-inplace 1`) are more memory-efficient as they modify the original tensor
- **Broadcasting** is performed automatically for compatible shapes
- **Batch processing** is efficiently handled through vectorized operations
- **Memory usage** scales with tensor size; larger images require more memory

## Technical Details

### Mathematical Formula
```
denormalized_pixel = (normalized_pixel * std) + mean
```

### Broadcasting Behavior
- Mean and std tensors are broadcast to match the image tensor shape
- Common patterns: scalar values, channel-wise values, or spatially-varying values
- PyTorch's broadcasting rules apply

### Memory Management
- Non-inplace operations create new tensors
- Inplace operations modify the input tensor directly
- All operations respect PyTorch's memory layout and optimization

## Migration Guide

### From Positional to Named Parameters

**Old (Positional):**
```tcl
set result [torch::denormalize_image $image $mean $std]
set result [torch::denormalize_image $image $mean $std 1]
```

**New (Named Parameters):**
```tcl
set result [torch::denormalize_image -image $image -mean $mean -std $std]
set result [torch::denormalize_image -image $image -mean $mean -std $std -inplace 1]
```

**Benefits of Named Parameters:**
- Self-documenting code
- Parameter order independence
- Easier to extend with new optional parameters
- Reduced errors from parameter misplacement

## Related Commands

- **torch::normalize_image**: Applies normalization (inverse operation)
- **torch::tensor_create**: Create tensors for mean and std values
- **torch::tensor_shape**: Check tensor dimensions for broadcasting compatibility
- **torch::tensor_dtype**: Verify tensor data types

## See Also

- [torch::normalize_image](normalize_image.md) - Image normalization
- [torch::tensor_create](../tensor_creation/tensor_create.md) - Tensor creation
- [Broadcasting in PyTorch](https://pytorch.org/docs/stable/notes/broadcasting.html)
- [ImageNet preprocessing standards](https://pytorch.org/docs/stable/torchvision/models.html) 