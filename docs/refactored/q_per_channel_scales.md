# torch::q_per_channel_scales

Gets the per-channel scale factors for a per-channel quantized tensor.

## Syntax

```tcl
torch::q_per_channel_scales quantized_tensor
torch::qPerChannelScales quantized_tensor    ;# camelCase alias
```

## Description

The `q_per_channel_scales` command returns the scale factors used for per-channel quantization of a tensor. Each scale factor corresponds to a slice of the tensor along the quantization axis.

## Arguments

* `quantized_tensor` (tensor) - A per-channel quantized tensor

## Return Value

Returns a 1-D tensor containing the scale factors used for per-channel quantization. The length of this tensor equals the size of the quantization axis dimension.

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

# Get the scale factors
set quant_scales [torch::q_per_channel_scales $quantized]
puts "Scale factors: [torch::tensor_data $quant_scales]"  ;# Prints: Scale factors: 1 1 1
```

## Error Conditions

* Returns an error if the input tensor is not a per-channel quantized tensor.

## See Also

* [torch::q_per_channel_axis](q_per_channel_axis.md)
* [torch::q_per_channel_zero_points](q_per_channel_zero_points.md)
* [torch::quantize_per_channel](quantize_per_channel.md) 