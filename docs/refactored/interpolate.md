# torch::interpolate

Performs tensor interpolation (upsampling/downsampling) using various interpolation algorithms. This function is essential for computer vision tasks, neural networks, and signal processing where you need to resize tensors while preserving their content structure.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::interpolate -input tensor -size size_list ?-mode mode? ?-align_corners align_corners? ?-scale_factor scale_factor_list?
torch::interpolate -tensor tensor -size size_list ?-mode mode? ?-alignCorners align_corners? ?-scaleFactor scale_factor_list?
```

### Positional Parameters (Legacy)
```tcl
torch::interpolate tensor size_list ?mode? ?align_corners? ?scale_factor_list?
```

### CamelCase Alias
```tcl
# Note: "interpolate" has no underscores, so camelCase is the same
torch::interpolate -input tensor -size size_list ?-mode mode? ?-align_corners align_corners?
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` or `-tensor` | tensor | Yes | - | Input tensor to interpolate |
| `-size` | list | Conditional* | - | Target output size as a list of integers |
| `-mode` | string | No | "nearest" | Interpolation algorithm |
| `-align_corners` or `-alignCorners` | integer | No | 0 | Whether to align corner pixels (0=false, 1=true) |
| `-scale_factor` or `-scaleFactor` | list | Conditional* | - | Scaling factors as a list of numbers |

*Either `-size` or `-scale_factor` must be provided, but not both.

## Interpolation Modes

| Mode | Description | Use Cases |
|------|-------------|-----------|
| `nearest` | Nearest neighbor interpolation | Fast upsampling, pixel art, categorical data |
| `linear` | Linear interpolation (1D) | 1D signal processing |
| `bilinear` | Bilinear interpolation (2D) | Image resizing, smooth upsampling |
| `bicubic` | Bicubic interpolation (2D) | High-quality image resizing |
| `trilinear` | Trilinear interpolation (3D) | Video/volumetric data processing |
| `area` | Area interpolation | Downsampling with anti-aliasing |

## Align Corners

The `align_corners` parameter affects how the input and output tensors are aligned:

- **0 (false)**: Corner pixels are treated as the centers of corner pixels
- **1 (true)**: Corner pixels are aligned exactly

**Note**: `align_corners` only works with interpolating modes (`linear`, `bilinear`, `bicubic`, `trilinear`). It's ignored for `nearest` and `area` modes.

## Examples

### Basic Image Upsampling
```tcl
# Create a small image tensor (1 batch, 3 channels, 32x32)
set image [torch::randn -shape {1 3 32 32}]

# Upsample to 64x64 using bilinear interpolation
set upsampled [torch::interpolate -input $image -size {64 64} -mode bilinear]
puts [torch::tensor_shape $upsampled]  ;# Output: 1 3 64 64
```

### Downsampling with Area Interpolation
```tcl
# Create a large image
set large_image [torch::randn -shape {1 3 256 256}]

# Downsample using area interpolation (good for anti-aliasing)
set small_image [torch::interpolate -input $large_image -size {64 64} -mode area]
puts [torch::tensor_shape $small_image]  ;# Output: 1 3 64 64
```

### Using Scale Factors
```tcl
# Create a tensor
set tensor [torch::ones -shape {1 1 10 10}]

# Scale by 2x in both dimensions
set scaled [torch::interpolate -input $tensor -scale_factor {2.0 2.0}]
puts [torch::tensor_shape $scaled]  ;# Output: 1 1 20 20

# Scale by different factors for each dimension
set scaled2 [torch::interpolate -input $tensor -scale_factor {1.5 2.0}]
puts [torch::tensor_shape $scaled2]  ;# Output: 1 1 15 20
```

### 1D Signal Interpolation
```tcl
# Create a 1D signal (batch_size=1, channels=1, length=10)
set signal [torch::randn -shape {1 1 10}]

# Interpolate to length 20 using linear interpolation
set interpolated [torch::interpolate -input $signal -size {20} -mode linear]
puts [torch::tensor_shape $interpolated]  ;# Output: 1 1 20
```

### 3D Volume Interpolation
```tcl
# Create a 3D volume (1 batch, 1 channel, 8x8x8)
set volume [torch::randn -shape {1 1 8 8 8}]

# Interpolate to 16x16x16 using trilinear interpolation
set large_volume [torch::interpolate -input $volume -size {16 16 16} -mode trilinear]
puts [torch::tensor_shape $large_volume]  ;# Output: 1 1 16 16 16
```

### Align Corners Comparison
```tcl
set image [torch::randn -shape {1 1 4 4}]

# Without align_corners (default)
set result1 [torch::interpolate -input $image -size {8 8} -mode bilinear]

# With align_corners
set result2 [torch::interpolate -input $image -size {8 8} -mode bilinear -align_corners 1]

# Results will be slightly different due to different alignment
```

### Batch Processing
```tcl
# Process multiple images at once
set batch_images [torch::randn -shape {4 3 32 32}]  ;# 4 images, 3 channels each

# Resize all images in the batch
set resized_batch [torch::interpolate -input $batch_images -size {64 64} -mode bilinear]
puts [torch::tensor_shape $resized_batch]  ;# Output: 4 3 64 64
```

## Neural Network Integration

### Feature Map Upsampling in CNNs
```tcl
# Typical use in decoder networks or U-Net architectures
set feature_maps [torch::randn -shape {1 256 16 16}]  ;# Deep feature maps

# Upsample for skip connections or output layers
set upsampled_features [torch::interpolate -input $feature_maps -size {32 32} -mode bilinear]
```

### Multi-Scale Feature Extraction
```tcl
set input_image [torch::randn -shape {1 3 224 224}]

# Create multiple scales for feature pyramid
set scale_1 [torch::interpolate -input $input_image -size {112 112} -mode bilinear]
set scale_2 [torch::interpolate -input $input_image -size {56 56} -mode bilinear]
set scale_3 [torch::interpolate -input $input_image -size {28 28} -mode bilinear]
```

## Performance Considerations

### Mode Selection
- **`nearest`**: Fastest, but lowest quality
- **`bilinear`**: Good balance of speed and quality for images
- **`bicubic`**: Higher quality but slower than bilinear
- **`area`**: Best for downsampling to prevent aliasing

### Memory Usage
- Upsampling increases memory usage quadratically for 2D tensors
- Consider processing in smaller batches for very large tensors

### GPU Acceleration
- All interpolation modes are GPU-accelerated when tensors are on CUDA devices
- `bilinear` and `nearest` are typically fastest on GPU

## Common Use Cases

### Computer Vision
```tcl
# Image preprocessing for neural networks
set original [torch::load_image "image.jpg"]
set resized [torch::interpolate -input $original -size {224 224} -mode bilinear]

# Data augmentation with random scales
set scale [expr {0.8 + rand() * 0.4}]  ;# Random scale 0.8-1.2
set h [expr {int(224 * $scale)}]
set w [expr {int(224 * $scale)}]
set augmented [torch::interpolate -input $original -size [list $h $w] -mode bilinear]
```

### Signal Processing
```tcl
# Audio resampling
set audio_signal [torch::randn -shape {1 1 16000}]  ;# 1 second at 16kHz
set resampled [torch::interpolate -input $audio_signal -size {22050} -mode linear]  ;# Resample to 22.05kHz
```

### Medical Imaging
```tcl
# 3D medical volume processing
set ct_scan [torch::randn -shape {1 1 512 512 200}]  ;# CT scan volume
set isotropic [torch::interpolate -input $ct_scan -size {256 256 256} -mode trilinear]  ;# Make isotropic
```

## Error Handling

### Common Errors
- **Missing size and scale_factor**: Must provide either `-size` or `-scale_factor`
- **Invalid mode**: Only supported modes are listed above
- **Incompatible align_corners**: Only works with interpolating modes
- **Invalid tensor dimensions**: Input must be 3D, 4D, or 5D tensor

### Example Error Handling
```tcl
if {[catch {torch::interpolate -input $tensor -size {64 64}} result]} {
    puts "Interpolation failed: $result"
    # Handle error appropriately
}
```

## Migration from Positional Syntax

### Before (Positional)
```tcl
set result [torch::interpolate $tensor {64 64} bilinear 1]
```

### After (Named Parameters)
```tcl
set result [torch::interpolate -input $tensor -size {64 64} -mode bilinear -align_corners 1]
```

## Technical Details

### Supported Input Shapes
- **3D tensors**: `(N, C, L)` for 1D interpolation
- **4D tensors**: `(N, C, H, W)` for 2D interpolation
- **5D tensors**: `(N, C, D, H, W)` for 3D interpolation

Where:
- `N` = batch size
- `C` = number of channels
- `L` = length (1D)
- `H` = height, `W` = width (2D)
- `D` = depth, `H` = height, `W` = width (3D)

### Mathematical Background
- **Bilinear**: Weighted average of 4 nearest pixels
- **Bicubic**: Weighted average of 16 nearest pixels using cubic polynomials
- **Trilinear**: Extension of bilinear to 3D (8 nearest voxels)
- **Area**: Adaptive pooling that preserves the area/volume ratio

## See Also

- `torch::upsample_bilinear` - Specialized bilinear upsampling
- `torch::upsample_nearest` - Specialized nearest neighbor upsampling
- `torch::grid_sample` - Advanced sampling with custom grids
- `torch::adaptive_avg_pool2d` - Adaptive pooling for downsampling 