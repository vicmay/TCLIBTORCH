# torch::adaptive_avgpool1d

Applies a 1D adaptive average pooling over an input signal.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::adaptive_avgpool1d -input tensor -output_size size
torch::adaptiveAvgpool1d -input tensor -output_size size
```

### Positional Parameters (Legacy)
```tcl
torch::adaptive_avgpool1d input output_size
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input`, `-tensor` | string | Input tensor handle | Yes |
| `-output_size`, `-outputSize` | integer | Target output size | Yes |

## Description

Adaptive average pooling applies average pooling with automatically computed kernel size and stride to produce the desired output size. Unlike regular average pooling, you specify the output size directly rather than kernel size and stride.

## Examples

### Named Parameter Syntax
```tcl
# Create a 1D sequence
set data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0}
set tensor [torch::tensor_create $data float32 cpu false]
set input [torch::tensor_reshape $tensor {1 1 8}]  # (batch, channel, length)

# Adaptive pooling to reduce to size 4
set result [torch::adaptive_avgpool1d -input $input -output_size 4]

# Alternative parameter names
set result2 [torch::adaptive_avgpool1d -tensor $input -outputSize 4]

# Using camelCase alias
set result3 [torch::adaptiveAvgpool1d -input $input -output_size 4]
```

### Positional Parameter Syntax (Legacy)
```tcl
set input [torch::tensor_reshape $tensor {1 1 8}]
set result [torch::adaptive_avgpool1d $input 4]
```

## Return Value

Returns a tensor handle with adaptive average pooling applied. The output maintains the same number of dimensions as the input but with the last dimension (length) adjusted to the specified output size.

## Error Handling

- **Missing parameters**: Both input tensor and output_size must be provided
- **Invalid tensor**: The specified tensor handle does not exist
- **Invalid output_size**: Must be a positive integer
- **Unknown parameters**: Using parameter names not recognized by the command

## Compatibility

- ✅ **Backward Compatible**: All existing code using positional syntax continues to work
- ✅ **New Features**: Named parameters provide better readability and maintainability
- ✅ **Modern**: camelCase alias follows contemporary API design patterns

## See Also

- [`torch::adaptive_avgpool2d`](adaptive_avgpool2d.md) - 2D adaptive average pooling
- [`torch::adaptive_avgpool3d`](adaptive_avgpool3d.md) - 3D adaptive average pooling
- [`torch::avgpool1d`](avgpool1d.md) - Regular 1D average pooling 