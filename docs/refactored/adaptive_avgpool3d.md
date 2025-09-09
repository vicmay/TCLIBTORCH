# torch::adaptive_avgpool3d

Applies a 3D adaptive average pooling over an input signal.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::adaptive_avgpool3d -input tensor -output_size size
torch::adaptiveAvgpool3d -input tensor -output_size size
```

### Positional Parameters (Legacy)
```tcl
torch::adaptive_avgpool3d input output_size
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input`, `-tensor` | string | Input tensor handle | Yes |
| `-output_size`, `-outputSize` | int or list | Target output size (single int or list of 3 ints) | Yes |

## Description

Adaptive average pooling for 3D tensors. The output_size can be a single integer (applied to all dimensions) or a list of 3 integers for each spatial dimension.

## Examples

```tcl
# Single size for all dimensions
set result [torch::adaptive_avgpool3d -input $tensor -output_size 2]

# Different sizes per dimension
set result [torch::adaptive_avgpool3d -input $tensor -output_size {2 3 4}]

# Legacy syntax
set result [torch::adaptive_avgpool3d $tensor 2]
```

## Compatibility

- ✅ **Backward Compatible**: Positional syntax continues to work
- ✅ **Named Parameters**: Better readability with `-input` and `-output_size`
- ✅ **camelCase Alias**: Modern `torch::adaptiveAvgpool3d` syntax 