# torch::quantize_per_tensor

## Description
Quantizes a tensor per-tensor with the given scale and zero point.

## Syntax

### Original (Positional Parameters)
```tcl
torch::quantize_per_tensor input scale zero_point dtype
```

### New (Named Parameters)
```tcl
torch::quantize_per_tensor -input input -scale scale -zeroPoint zero_point -dtype dtype
```

### CamelCase Alias
```tcl
torch::quantizePerTensor input scale zero_point dtype
torch::quantizePerTensor -input input -scale scale -zeroPoint zero_point -dtype dtype
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | tensor | Input tensor to quantize |
| `scale` | float | Quantization scale factor |
| `zero_point` | int | Quantization zero point |
| `dtype` | string | Quantized data type (qint8, quint8, qint32, etc.) |

## Return Value
Returns a quantized tensor handle.

## Examples

### Basic Usage
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set quantized [torch::quantize_per_tensor $input 0.1 128 qint8]
```

### Named Parameters
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set quantized [torch::quantize_per_tensor -input $input -scale 0.1 -zeroPoint 128 -dtype qint8]
```

### CamelCase Alias
```tcl
set input [torch::randn [list 2 3] -dtype float32]
set quantized [torch::quantizePerTensor $input 0.1 128 qint8]
```

## Error Handling
- Throws error if required parameters are missing
- Throws error if input tensor handle is invalid
- Throws error if scale or zero_point values are invalid
- Throws error if dtype is not a valid quantized type

## Notes
- Currently only supports positional syntax; named parameter support is planned
- Quantization requires specific tensor types and backends in PyTorch
- The input tensor must be compatible with the quantization backend
- Common quantized dtypes: qint8, quint8, qint32

## Migration Guide
When dual syntax support is added, existing code will continue to work:

```tcl
# Old code (will continue to work)
set result [torch::quantize_per_tensor $input 0.1 128 qint8]

# New code (when dual syntax is implemented)
set result [torch::quantize_per_tensor -input $input -scale 0.1 -zeroPoint 128 -dtype qint8]

# CamelCase alias (available now)
set result [torch::quantizePerTensor $input 0.1 128 qint8]
```

## See Also
- [torch::quantize_per_channel](quantize_per_channel.md) - Per-channel quantization
- [torch::dequantize](dequantize.md) - Dequantization
- [torch::q_scale](q_scale.md) - Get quantization scale
- [torch::q_zero_point](q_zero_point.md) - Get quantization zero point 