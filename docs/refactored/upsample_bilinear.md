# torch::upsample_bilinear

Performs bilinear upsampling/downsampling on a 4D input tensor. This operation uses bilinear interpolation to resize the spatial dimensions (height and width) of the input tensor.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::upsample_bilinear -input tensor [-output_size list] [-scale_factor list] [-align_corners bool] [-antialias bool]
torch::upsampleBilinear -input tensor [-output_size list] [-scale_factor list] [-align_corners bool] [-antialias bool]
```

### Positional Syntax (Legacy)
```tcl
torch::upsample_bilinear input size|scale_factor [align_corners] [antialias]
torch::upsampleBilinear input size|scale_factor [align_corners] [antialias]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | tensor | required | Input 4D tensor: [batch_size, channels, height, width] |
| output_size | list | none | Target output size as [height, width]. Mutually exclusive with scale_factor |
| size | list | none | Alias for output_size |
| scale_factor | list | none | Scaling factors as [height_scale, width_scale]. Mutually exclusive with output_size |
| align_corners | bool | false | Whether to align corners when interpolating |
| antialias | bool | false | Whether to apply antialiasing (if supported by PyTorch version) |

**Note**: Either `output_size` (or `size`) **OR** `scale_factor` must be specified, but not both.

## Output

Returns a tensor handle for the result of the bilinear upsampling operation.

**Output Shape**: `[batch_size, channels, output_height, output_width]`

Where:
- If using `output_size`: `output_height = output_size[0]`, `output_width = output_size[1]`
- If using `scale_factor`: `output_height = input_height × scale_factor[0]`, `output_width = input_width × scale_factor[1]`

## Examples

### Basic Upsampling with Output Size
```tcl
# Create input: batch=1, channels=3, height=4, width=4
set input [torch::ones -shape {1 3 4 4} -dtype float32]

# Upsample to 8x8
set upsampled [torch::upsample_bilinear -input $input -output_size {8 8}]
# Output shape: [1, 3, 8, 8]
```

### Upsampling with Scale Factor
```tcl
set input [torch::ones -shape {1 3 4 4} -dtype float32]

# Double the size using scale factor
set upsampled [torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}]
# Output shape: [1, 3, 8, 8]
```

### Asymmetric Scaling
```tcl
set input [torch::ones -shape {1 3 4 4} -dtype float32]

# Different scaling for height and width
set result [torch::upsample_bilinear -input $input -output_size {8 16}]
# Output shape: [1, 3, 8, 16]

# Or using scale factors
set result2 [torch::upsample_bilinear -input $input -scale_factor {2.0 4.0}]
# Output shape: [1, 3, 8, 16]
```

### With Align Corners
```tcl
set input [torch::ones -shape {2 64 32 32} -dtype float32]

# Align corners for more precise interpolation
set upsampled [torch::upsample_bilinear \
    -input $input \
    -output_size {64 64} \
    -align_corners 1]
```

### Downsampling
```tcl
set input [torch::ones -shape {1 3 16 16} -dtype float32]

# Downsample to smaller size
set downsampled [torch::upsample_bilinear -input $input -output_size {8 8}]
# Output shape: [1, 3, 8, 8]

# Or using fractional scale factors
set downsampled2 [torch::upsample_bilinear -input $input -scale_factor {0.5 0.5}]
# Output shape: [1, 3, 8, 8]
```

### Using camelCase Alias
```tcl
set input [torch::ones -shape {1 32 16 16} -dtype float32]

set result [torch::upsampleBilinear -input $input -scale_factor {1.5 1.5}]
```

### Auto-Detection of Size vs Scale Factor
```tcl
set input [torch::ones -shape {1 3 4 4} -dtype float32]

# These are auto-detected as output_size (integers)
set result1 [torch::upsample_bilinear $input {8 8}]

# These are auto-detected as scale_factor (doubles)
set result2 [torch::upsample_bilinear $input {2.0 2.0}]
```

## Mathematical Details

### Bilinear Interpolation
Bilinear interpolation computes output pixel values by taking a weighted average of the four nearest input pixels. For each output pixel at coordinates (y, x):

1. **Map to input coordinates**: 
   - `input_y = y × (input_height - 1) / (output_height - 1)` (if align_corners=true)
   - `input_x = x × (input_width - 1) / (output_width - 1)` (if align_corners=true)

2. **Find surrounding pixels**: Get the four nearest input pixels
3. **Interpolate**: Compute weighted average based on distances

### Align Corners Effect
- **align_corners=false** (default): Uses pixel center alignment
- **align_corners=true**: Uses corner alignment, preserving corner pixel values exactly

### Size vs Scale Factor
```tcl
# These produce the same result for 4x4 input:
torch::upsample_bilinear -input $input -output_size {8 8}
torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}
```

## Use Cases

### 1. Image Upsampling
```tcl
# Increase image resolution
set low_res [torch::ones -shape {1 3 64 64} -dtype float32]
set high_res [torch::upsample_bilinear \
    -input $low_res \
    -output_size {128 128} \
    -align_corners 1]
```

### 2. Feature Map Upsampling in Neural Networks
```tcl
# Upsample feature maps in decoder networks
set features [torch::ones -shape {8 256 8 8} -dtype float32]
set upsampled_features [torch::upsample_bilinear \
    -input $features \
    -scale_factor {2.0 2.0}]
# Output: [8, 256, 16, 16]
```

### 3. Semantic Segmentation
```tcl
# Upsample segmentation maps to match input resolution
set segmap [torch::ones -shape {1 21 32 32} -dtype float32]  # 21 classes
set full_res_segmap [torch::upsample_bilinear \
    -input $segmap \
    -output_size {256 256}]
```

### 4. Data Augmentation
```tcl
# Random scaling for data augmentation
set image [torch::ones -shape {1 3 224 224} -dtype float32]
set scaled_image [torch::upsample_bilinear \
    -input $image \
    -scale_factor {1.2 0.8}]  # Non-uniform scaling
```

### 5. Multi-Scale Processing
```tcl
# Create multiple scales of the same image
set original [torch::ones -shape {1 3 64 64} -dtype float32]

set scale_1_5 [torch::upsample_bilinear -input $original -scale_factor {1.5 1.5}]
set scale_2_0 [torch::upsample_bilinear -input $original -scale_factor {2.0 2.0}]
set scale_0_5 [torch::upsample_bilinear -input $original -scale_factor {0.5 0.5}]
```

## Input Requirements

### Tensor Shape
- **Required**: 4D tensor with shape `[batch_size, channels, height, width]`
- **3D tensors**: Not supported for bilinear interpolation
- **5D tensors**: Not supported for bilinear interpolation

### Data Types
- **Supported**: float32, float64, and other floating-point types
- **Integer types**: May work but floating-point recommended for interpolation

## Error Handling

The command will raise an error for:
- Missing required parameters (`input` and either `output_size` or `scale_factor`)
- Invalid tensor names or non-existent tensors
- Wrong tensor dimensions (not 4D)
- Specifying both `output_size` and `scale_factor`
- Invalid parameter values (negative sizes, etc.)
- Malformed size or scale_factor lists
- Unknown parameter names

## Performance Notes

- **Memory usage**: Increases quadratically with scale factor
- **Speed**: Bilinear is faster than bicubic but slower than nearest neighbor
- **Quality**: Good balance between speed and visual quality
- **Large scale factors**: Consider multiple smaller steps for very large upsampling

## Migration Guide

### From Legacy Syntax
```tcl
# Old positional syntax
set result [torch::upsample_bilinear $input {8 8} 1]

# New named parameter syntax
set result [torch::upsample_bilinear -input $input -output_size {8 8} -align_corners 1]
```

### Parameter Mapping
| Old Position | New Parameter | Notes |
|--------------|---------------|--------|
| 1 | -input | Required |
| 2 | -output_size or -scale_factor | Auto-detected based on type |
| 3 | -align_corners | Optional, default false |
| 4 | -antialias | Optional, default false |

### New Features in Named Syntax
```tcl
# Old: only supported output size
torch::upsample_bilinear $input {8 8}

# New: supports both output size and scale factors
torch::upsample_bilinear -input $input -output_size {8 8}
torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}

# New: explicit parameter names prevent confusion
torch::upsample_bilinear -input $input -size {8 8} -align_corners 1 -antialias 0
```

## Common Patterns

### Encoder-Decoder Networks
```tcl
# Encoder (downsampling)
set encoded [torch::upsample_bilinear -input $input -scale_factor {0.5 0.5}]

# Decoder (upsampling)
set decoded [torch::upsample_bilinear -input $encoded -scale_factor {2.0 2.0}]
```

### Progressive Upsampling
```tcl
# Instead of large single step
set large_step [torch::upsample_bilinear -input $input -scale_factor {8.0 8.0}]

# Use progressive steps for better quality
set step1 [torch::upsample_bilinear -input $input -scale_factor {2.0 2.0}]
set step2 [torch::upsample_bilinear -input $step1 -scale_factor {2.0 2.0}]
set step3 [torch::upsample_bilinear -input $step2 -scale_factor {2.0 2.0}]
```

## See Also

- `torch::upsample_nearest` - Nearest neighbor upsampling (faster)
- `torch::interpolate` - General interpolation with multiple modes
- `torch::tensor_conv_transpose2d` - Learned upsampling via transposed convolution
- `torch::grid_sample` - Arbitrary spatial transformations
- `torch::resize_image` - Image-specific resizing operations 