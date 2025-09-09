# torch::pixel_unshuffle / torch::pixelUnshuffle

## Description
Reverses the pixel shuffle operation, rearranging elements in a tensor of shape (*, C, H × r, W × r) to a tensor of shape (*, C × r², H, W), where r is the downscale factor. This operation is useful in feature extraction and dimensionality reduction.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::pixel_unshuffle input downscale_factor
```

### Named Parameter Syntax
```tcl
torch::pixel_unshuffle -input tensor -downscaleFactor factor
torch::pixelUnshuffle -input tensor -downscaleFactor factor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| input / -input / -tensor | tensor | Input tensor of shape (*, C, H × r, W × r) |
| downscale_factor / -downscaleFactor / -factor | int | Factor to decrease spatial resolution by |

## Return Value
Returns a tensor of shape (*, C × r², H, W) where r is the downscale factor.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with shape [1, 1, 4, 4]
set input [torch::zeros {1 1 4 4}]
# Apply pixel unshuffle with downscale factor 2
# Result will have shape [1, 4, 2, 2]
set output [torch::pixel_unshuffle $input 2]
```

### Named Parameter Syntax
```tcl
# Using named parameters
set output [torch::pixel_unshuffle -input $input -downscaleFactor 2]

# Alternative parameter names
set output [torch::pixel_unshuffle -tensor $input -factor 2]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set output [torch::pixelUnshuffle -input $input -downscaleFactor 2]
```

## Common Use Cases

1. **Feature Extraction**: Used to increase channel dimension while reducing spatial dimensions.
```tcl
# Example: Reducing spatial dimensions for feature extraction
set features [torch::zeros {1 1 32 32}]  # Single channel
set compressed [torch::pixel_unshuffle $features 4]  # Result: [1 16 8 8]
```

2. **Inverse Operation**: Reverse pixel_shuffle operation to recover original tensor dimensions.
```tcl
# Example: Reversing pixel_shuffle operation
set original [torch::zeros {1 4 2 2}]
set shuffled [torch::pixel_shuffle $original 2]  # [1 1 4 4]
set restored [torch::pixel_unshuffle $shuffled 2]  # Back to [1 4 2 2]
```

## Error Cases

1. Missing required parameters:
```tcl
# Error: Missing downscale_factor
torch::pixel_unshuffle $input
```

2. Invalid input tensor:
```tcl
# Error: Invalid tensor name
torch::pixel_unshuffle "invalid_tensor" 2
```

3. Invalid downscale factor:
```tcl
# Error: Downscale factor must be positive
torch::pixel_unshuffle -input $input -downscaleFactor 0
```

## See Also
- [torch::pixel_shuffle](pixel_shuffle.md) - Reverse operation of pixel_unshuffle
- [torch::interpolate](interpolate.md) - General purpose interpolation
- [torch::adaptive_avgpool2d](adaptive_avgpool2d.md) - Alternative downsampling method 