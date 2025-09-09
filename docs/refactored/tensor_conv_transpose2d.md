# torch::tensor_conv_transpose2d

Applies a 2D transposed convolution (deconvolution) over an input image tensor. This operation is the reverse of a 2D convolution and is commonly used for upsampling and decoder networks.

## Syntax

### Named Parameter Syntax (Recommended)
```tcl
torch::tensor_conv_transpose2d -input tensor -weight tensor [-bias tensor] [-stride int|list] [-padding int|list] [-output_padding int|list] [-groups int] [-dilation int|list]
torch::tensorConvTranspose2d -input tensor -weight tensor [-bias tensor] [-stride int|list] [-padding int|list] [-output_padding int|list] [-groups int] [-dilation int|list]
```

### Positional Syntax (Legacy)
```tcl
torch::tensor_conv_transpose2d input weight [bias] [stride] [padding] [output_padding] [groups] [dilation]
torch::tensorConvTranspose2d input weight [bias] [stride] [padding] [output_padding] [groups] [dilation]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | tensor | required | Input 4D tensor: [batch_size, in_channels, height, width] |
| weight | tensor | required | Weight 4D tensor: [in_channels, out_channels, kernel_height, kernel_width] |
| bias | tensor | none | Bias 1D tensor: [out_channels]. Use "none" or empty to skip |
| stride | int\|list | 1 | Stride for convolution. Single int or list [height, width] |
| padding | int\|list | 0 | Zero-padding added to both sides. Single int or list [height, width] |
| output_padding | int\|list | 0 | Additional size added to one side of output. Single int or list [height, width] |
| groups | int | 1 | Number of blocked connections from input to output channels |
| dilation | int\|list | 1 | Spacing between kernel elements. Single int or list [height, width] |

## Output

Returns a tensor handle for the result of the 2D transposed convolution operation.

**Output Shape**: `[batch_size, out_channels, output_height, output_width]`

Where:
- `output_height = (input_height - 1) × stride[0] - 2 × padding[0] + kernel_height + output_padding[0]`
- `output_width = (input_width - 1) × stride[1] - 2 × padding[1] + kernel_width + output_padding[1]`

## Examples

### Basic Transposed Convolution
```tcl
# Create input: batch=1, channels=3, height=8, width=8
set input [torch::ones -shape {1 3 8 8} -dtype float32]

# Create weight: in_channels=3, out_channels=16, kernel=3x3
set weight [torch::ones -shape {3 16 3 3} -dtype float32]

# Apply transposed convolution
set result [torch::tensor_conv_transpose2d -input $input -weight $weight]
# Output shape: [1, 16, 10, 10]
```

### Upsampling with Stride
```tcl
# Create input for upsampling
set input [torch::ones -shape {1 64 32 32} -dtype float32]
set weight [torch::ones -shape {64 32 4 4} -dtype float32]

# Upsample by factor of 2
set upsampled [torch::tensor_conv_transpose2d -input $input -weight $weight -stride 2 -padding 1]
# Output shape: [1, 32, 64, 64]
```

### With Bias and Custom Parameters
```tcl
set input [torch::ones -shape {2 8 16 16} -dtype float32]
set weight [torch::ones -shape {8 12 3 3} -dtype float32]
set bias [torch::zeros -shape {12} -dtype float32]

set result [torch::tensor_conv_transpose2d \
    -input $input \
    -weight $weight \
    -bias $bias \
    -stride 2 \
    -padding 1 \
    -output_padding 1]
```

### Asymmetric Parameters
```tcl
# Different stride for height and width
set result [torch::tensor_conv_transpose2d \
    -input $input \
    -weight $weight \
    -stride {2 1} \
    -padding {1 2} \
    -dilation {1 2}]
```

### Grouped Transposed Convolution
```tcl
# Input with 8 channels, 2 groups (4 channels per group)
set input [torch::ones -shape {1 8 16 16} -dtype float32]
set weight [torch::ones -shape {8 6 3 3} -dtype float32]

set result [torch::tensor_conv_transpose2d \
    -input $input \
    -weight $weight \
    -groups 2]
# Output: [1, 12, 18, 18] (6 output channels per group × 2 groups)
```

### Using camelCase Alias
```tcl
set input [torch::ones -shape {1 16 8 8} -dtype float32]
set weight [torch::ones -shape {16 32 4 4} -dtype float32]

set result [torch::tensorConvTranspose2d -input $input -weight $weight -stride 2]
```

## Mathematical Details

### Transposed Convolution vs Regular Convolution
- **Regular convolution**: Reduces spatial dimensions (downsampling)
- **Transposed convolution**: Increases spatial dimensions (upsampling)
- The weight tensor shape differs: for transposed conv, it's `[in_channels, out_channels, ...]`

### Output Size Calculation
For 2D transposed convolution:
```
output_height = (input_height - 1) × stride_h - 2 × padding_h + kernel_height + output_padding_h
output_width = (input_width - 1) × stride_w - 2 × padding_w + kernel_width + output_padding_w
```

### Parameter Interpretation
- **stride**: Controls upsampling factor. Larger stride = more upsampling
- **padding**: Reduces output size (opposite of regular convolution)
- **output_padding**: Fine-tunes output size when stride > 1
- **dilation**: Increases effective kernel size
- **groups**: Splits channels into independent groups

## Use Cases

### 1. Image Upsampling
```tcl
# Upsample 32x32 to 64x64
set small_image [torch::ones -shape {1 3 32 32} -dtype float32]
set upconv_weight [torch::ones -shape {3 3 4 4} -dtype float32]

set large_image [torch::tensor_conv_transpose2d \
    -input $small_image \
    -weight $upconv_weight \
    -stride 2 \
    -padding 1]
```

### 2. Decoder Networks
```tcl
# Decoder layer in autoencoder/GAN
set encoded [torch::ones -shape {8 256 4 4} -dtype float32]
set decoder_weight [torch::ones -shape {256 128 4 4} -dtype float32]

set decoded [torch::tensor_conv_transpose2d \
    -input $encoded \
    -weight $decoder_weight \
    -stride 2 \
    -padding 1]
# Output: [8, 128, 8, 8]
```

### 3. Semantic Segmentation
```tcl
# Upsampling in U-Net style architecture
set features [torch::ones -shape {4 512 8 8} -dtype float32]
set upsample_weight [torch::ones -shape {512 256 3 3} -dtype float32]

set upsampled [torch::tensor_conv_transpose2d \
    -input $features \
    -weight $upsample_weight \
    -stride 2 \
    -padding 1 \
    -output_padding 1]
```

## Weight Tensor Shape Requirements

The weight tensor shape for transposed convolution is:
`[input_channels, output_channels, kernel_height, kernel_width]`

This is **different** from regular convolution which uses:
`[output_channels, input_channels, kernel_height, kernel_width]`

## Error Handling

The command will raise an error for:
- Missing required parameters (`input`, `weight`)
- Invalid tensor names
- Dimension mismatches between tensors
- Invalid parameter values (negative stride, etc.)
- Wrong list size for 2D parameters (must be exactly 2 elements)
- Unknown parameter names

## Migration Guide

### From Legacy Syntax
```tcl
# Old positional syntax
set result [torch::tensor_conv_transpose2d $input $weight $bias 2]

# New named parameter syntax
set result [torch::tensor_conv_transpose2d -input $input -weight $weight -bias $bias -stride 2]
```

### Parameter Mapping
| Old Position | New Parameter | Notes |
|--------------|---------------|--------|
| 1 | -input | Required |
| 2 | -weight | Required |
| 3 | -bias | Use "none" to skip |
| 4 | -stride | Now supports pairs |
| 5 | -padding | Now supports pairs |
| 6 | -output_padding | Now supports pairs |
| 7 | -groups | Same as before |
| 8 | -dilation | Now supports pairs |

### New 2D Parameter Features
```tcl
# Old: single stride value
torch::tensor_conv_transpose2d $input $weight none 2

# New: can specify different stride for height/width
torch::tensor_conv_transpose2d -input $input -weight $weight -stride {2 1}
```

## Performance Notes

- Transposed convolution is computationally more expensive than regular convolution
- Memory usage increases significantly with larger stride values
- For large upsampling factors, consider using interpolation followed by regular convolution
- Groups parameter can reduce computation and memory usage

## See Also

- `torch::tensor_conv2d` - Regular 2D convolution
- `torch::tensor_conv_transpose1d` - 1D transposed convolution
- `torch::tensor_conv_transpose3d` - 3D transposed convolution
- `torch::upsample_bilinear` - Alternative upsampling method 