# torch::quantize_per_channel

## Description
Quantizes a tensor per-channel with the given scales and zero points along the specified axis.

## Syntax

### Original (Positional Parameters)
```tcl
torch::quantize_per_channel input scales zero_points axis dtype
```

### New (Named Parameters)
```tcl
torch::quantize_per_channel -input input -scales scales -zeroPoints zero_points -axis axis -dtype dtype
```

### CamelCase Alias
```tcl
torch::quantizePerChannel input scales zero_points axis dtype
torch::quantizePerChannel -input input -scales scales -zeroPoints zero_points -axis axis -dtype dtype
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | tensor | Input tensor to quantize |
| `scales` | tensor | 1D tensor of scales for each channel |
| `zero_points` | tensor | 1D tensor of zero points for each channel |
| `axis` | int | Dimension along which to quantize |
| `dtype` | string | Quantized data type (qint8, quint8, qint32, etc.) |

## Return Value
Returns a quantized tensor handle.

## Examples

### Basic Usage
```tcl
set input [torch::randn [list 2 3 4] -dtype float32]
set scales [torch::ones [list 3] -dtype float32]
set zero_points [torch::zeros [list 3] -dtype int32]
set quantized [torch::quantize_per_channel $input $scales $zero_points 1 qint8]
```

### Named Parameters
```tcl
set input [torch::randn [list 2 3 4] -dtype float32]
set scales [torch::ones [list 3] -dtype float32]
set zero_points [torch::zeros [list 3] -dtype int32]
set quantized [torch::quantize_per_channel -input $input -scales $scales -zeroPoints $zero_points -axis 1 -dtype qint8]
```

### CamelCase Alias
```tcl
set input [torch::randn [list 2 3 4] -dtype float32]
set scales [torch::ones [list 3] -dtype float32]
set zero_points [torch::zeros [list 3] -dtype int32]
set quantized [torch::quantizePerChannel $input $scales $zero_points 1 qint8]
```

## Error Handling
- Throws error if required parameters are missing
- Throws error if input tensor handle is invalid
- Throws error if scales tensor handle is invalid
- Throws error if zero_points tensor handle is invalid
- Throws error if axis is out of bounds
- Throws error if dtype is not a valid quantized type

## Notes
- Currently only supports positional syntax; named parameter support is planned
- Quantization requires specific tensor types and backends in PyTorch
- The input tensor must be compatible with the quantization backend
- The scales and zero_points tensors must have the same size as the specified axis dimension
- Common quantized dtypes: qint8, quint8, qint32

## Migration Guide
When dual syntax support is added, existing code will continue to work:

```tcl
# Old code (will continue to work)
set result [torch::quantize_per_channel $input $scales $zero_points 1 qint8]

# New code (when dual syntax is implemented)
set result [torch::quantize_per_channel -input $input -scales $scales -zeroPoints $zero_points -axis 1 -dtype qint8]

# CamelCase alias (available now)
set result [torch::quantizePerChannel $input $scales $zero_points 1 qint8]
```

## See Also
- [torch::quantize_per_tensor](quantize_per_tensor.md) - Per-tensor quantization
- [torch::dequantize](dequantize.md) - Dequantization
- [torch::q_per_channel_scales](q_per_channel_scales.md) - Get per-channel scales
- [torch::q_per_channel_zero_points](q_per_channel_zero_points.md) - Get per-channel zero points
- [torch::q_per_channel_axis](q_per_channel_axis.md) - Get quantization axis 