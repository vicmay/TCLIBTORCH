# torch::pixel_shuffle / torch::pixelShuffle

## Description
Rearranges elements in a tensor of shape (*, C × r², H, W) to a tensor of shape (*, C, H × r, W × r), where r is the upscale factor. This operation is particularly useful in super-resolution models.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::pixel_shuffle input upscale_factor
```

### Named Parameter Syntax
```tcl
torch::pixel_shuffle -input tensor -upscaleFactor factor
torch::pixelShuffle -input tensor -upscaleFactor factor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| input / -input / -tensor | tensor | Input tensor of shape (*, C × r², H, W) |
| upscale_factor / -upscaleFactor / -factor | int | Factor to increase spatial resolution by |

## Return Value
Returns a tensor of shape (*, C, H × r, W × r) where r is the upscale factor.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with shape [1, 4, 2, 2]
set input [torch::zeros {1 4 2 2}]
# Apply pixel shuffle with upscale factor 2
# Result will have shape [1, 1, 4, 4]
set output [torch::pixel_shuffle $input 2]
```

### Named Parameter Syntax
```tcl
# Using named parameters
set output [torch::pixel_shuffle -input $input -upscaleFactor 2]

# Alternative parameter names
set output [torch::pixel_shuffle -tensor $input -factor 2]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set output [torch::pixelShuffle -input $input -upscaleFactor 2]
```

## Common Use Cases

1. **Super-Resolution Networks**: Used in the final layer of super-resolution networks to increase spatial dimensions.
```tcl
# Example: Upscaling a feature map
set features [torch::zeros {1 16 32 32}]  # 16 channels
set upscaled [torch::pixel_shuffle $features 2]  # Result: [1 4 64 64]
```

2. **Sub-Pixel Convolution**: Efficient alternative to transposed convolution for upsampling.
```tcl
# Example: Sub-pixel convolution layer
set conv_output [torch::zeros {1 36 8 8}]  # 36 = 4 * 3^2 channels
set upsampled [torch::pixel_shuffle $conv_output 3]  # Result: [1 4 24 24]
```

## Error Cases

1. Missing required parameters:
```tcl
# Error: Missing upscale_factor
torch::pixel_shuffle $input
```

2. Invalid input tensor:
```tcl
# Error: Invalid tensor name
torch::pixel_shuffle "invalid_tensor" 2
```

3. Invalid upscale factor:
```tcl
# Error: Upscale factor must be positive
torch::pixel_shuffle -input $input -upscaleFactor 0
```

## See Also
- [torch::pixel_unshuffle](pixel_unshuffle.md) - Reverse operation of pixel_shuffle
- [torch::upsample_bilinear](upsample_bilinear.md) - Alternative upsampling method
- [torch::interpolate](interpolate.md) - General purpose interpolation
