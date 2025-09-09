# torch::q_per_channel_axis

Gets the quantization axis for a per-channel quantized tensor.

## Syntax

```tcl
torch::q_per_channel_axis quantized_tensor
torch::qPerChannelAxis quantized_tensor    ;# camelCase alias
```

## Description

The `q_per_channel_axis` command returns the axis along which the per-channel quantization was performed for a given quantized tensor. This is typically the channel dimension (e.g., 1 for NCHW format tensors).

## Arguments

* `quantized_tensor` (tensor) - A per-channel quantized tensor

## Return Value

Returns an integer representing the axis along which per-channel quantization was performed.

## Example

```tcl
# Create a float tensor to quantize
set input [torch::randn [list 2 3 4 5]]

# Create scales and zero_points tensors
set scales [torch::ones [list 3]]  ;# One scale per channel
set zero_points [torch::zeros [list 3] -dtype int32]  ;# One zero point per channel

# Quantize along channel dimension (axis 1)
set axis 1
set quantized [torch::quantize_per_channel $input $scales $zero_points $axis qint8]

# Get the quantization axis
set quant_axis [torch::q_per_channel_axis $quantized]
puts "Quantization axis: $quant_axis"  ;# Prints: Quantization axis: 1
```

## Error Conditions

* Returns an error if the input tensor is not a per-channel quantized tensor.

## See Also

* [torch::q_per_channel_scales](q_per_channel_scales.md)
* [torch::q_per_channel_zero_points](q_per_channel_zero_points.md)
* [torch::quantize_per_channel](quantize_per_channel.md) 