# torch::maxpool1d / torch::maxPool1d

Creates a 1D max pooling layer for neural networks.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::maxpool1d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?
torch::maxPool1d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?
```

### Positional Parameters (Legacy)
```tcl
torch::maxpool1d kernel_size ?stride? ?padding? ?ceil_mode?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-kernelSize` | integer | Required | Size of the pooling window |
| `-stride` | integer | kernelSize | Stride of the pooling window |
| `-padding` | integer | 0 | Zero-padding added to both sides |
| `-ceilMode` | boolean | 0 | When true, use ceil instead of floor for output shape |

## Description

The `torch::maxpool1d` command creates a 1D max pooling layer that applies a sliding window over an input signal and outputs the maximum value within each window. This operation is commonly used to:

- Reduce spatial dimensions
- Create translation invariance
- Decrease computational complexity
- Extract dominant features

The layer operates on input tensors of shape `(batch_size, channels, length)`.

## Return Value

Returns a handle to the created maxpool1d layer module.

## Examples

### Basic Usage
```tcl
# Create a basic maxpool1d layer with kernel size 3
set layer [torch::maxpool1d -kernelSize 3]

# Create input tensor (batch_size=1, channels=1, length=5)
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32]
set input [torch::tensor_reshape $input {1 1 5}]

# Forward pass
set output [torch::layer_forward $layer $input]
puts [torch::tensor_shape $output]  ;# Output: 1 1 1
```

### Advanced Configuration
```tcl
# Create maxpool1d with stride and padding
set layer [torch::maxpool1d -kernelSize 2 -stride 2 -padding 1]

# Create input tensor
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0} -dtype float32]
set input [torch::tensor_reshape $input {1 1 4}]

# Forward pass
set output [torch::layer_forward $layer $input]
puts [torch::tensor_shape $output]  ;# Output: 1 1 3
```

### Using CamelCase Alias
```tcl
# Using camelCase alias
set layer [torch::maxPool1d -kernelSize 2 -stride 1 -padding 1]
```

### Legacy Positional Syntax
```tcl
# Using positional parameters (not recommended for new code)
set layer [torch::maxpool1d 2 1 1]
```

## Output Shape Calculation

The output shape for a given input length L is calculated as:

```
output_length = floor((L + 2 * padding - kernel_size) / stride + 1)
```

If `ceilMode` is true, ceil is used instead of floor.

## Common Use Cases

### Signal Processing
```tcl
# Downsample audio signal
set audio_signal [torch::tensor_reshape $signal {1 1 1000}]
set pooling [torch::maxpool1d -kernelSize 4 -stride 4]
set downsampled [torch::layer_forward $pooling $audio_signal]
```

### Feature Extraction
```tcl
# Extract features from multi-channel signal
set signal [torch::tensor_reshape $data {1 3 100}]  ;# 3 channels
set pool [torch::maxpool1d -kernelSize 5 -stride 2]
set features [torch::layer_forward $pool $signal]
```

## Best Practices

1. **Kernel Size Selection**:
   - Use powers of 2 (2, 4, 8) for standard downsampling
   - Match stride to kernel size for non-overlapping windows
   - Use smaller stride than kernel size for overlapping windows

2. **Padding Usage**:
   - Use padding to control output size
   - Add padding when you want to preserve edge information
   - Set padding = (kernel_size - 1) / 2 to maintain input length

3. **Performance Considerations**:
   - Larger kernel sizes reduce memory usage
   - Smaller strides increase computation time
   - Balance between feature preservation and efficiency

## Error Handling

The command will raise an error if:
- Kernel size is not positive
- Invalid parameter names are used
- Missing parameter values
- Invalid parameter types

## Related Commands

- `torch::maxpool2d` - 2D max pooling
- `torch::maxpool3d` - 3D max pooling
- `torch::avgpool1d` - 1D average pooling
- `torch::layer_forward` - Forward pass through layers

## Migration Guide

### From Legacy to Modern Syntax

```tcl
# Old way (still supported)
set layer [torch::maxpool1d 3 2 1]

# New way (recommended)
set layer [torch::maxpool1d -kernelSize 3 -stride 2 -padding 1]

# Modern camelCase
set layer [torch::maxPool1d -kernelSize 3 -stride 2 -padding 1]
```

### Benefits of Modern Syntax

1. **Self-documenting**: Parameter names make code more readable
2. **Flexible**: Parameter order doesn't matter
3. **Maintainable**: Easier to modify parameters later
4. **Consistent**: Matches modern API patterns

## Notes

1. **Default Stride**: When stride is not specified, it defaults to the kernel size
2. **Memory Usage**: Max pooling doesn't require additional training parameters
3. **Gradient Flow**: During backpropagation, gradients flow only through maximum values
4. **Device Compatibility**: Works on both CPU and CUDA devices
5. **Type Preservation**: Output tensor has same dtype as input tensor 