# torch::q_per_channel_zero_points

Gets the per-channel zero points for a per-channel quantized tensor.

## Syntax

```tcl
torch::q_per_channel_zero_points quantized_tensor
torch::qPerChannelZeroPoints quantized_tensor    ;# camelCase alias
```

## Description

The `q_per_channel_zero_points` command returns the zero points used for per-channel quantization of a tensor. Each zero point corresponds to a slice of the tensor along the quantization axis and represents the quantized value that maps to the real value 0.

## Arguments

* `quantized_tensor` (tensor) - A per-channel quantized tensor

## Return Value

Returns a 1-D tensor containing the zero points used for per-channel quantization. The length of this tensor equals the size of the quantization axis dimension. The values are of type int32.

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

# Get the zero points
set quant_zero_points [torch::q_per_channel_zero_points $quantized]
puts "Zero points: [torch::tensor_data $quant_zero_points]"  ;# Prints: Zero points: 0 0 0
```

## Error Conditions

* Returns an error if the input tensor is not a per-channel quantized tensor.

## See Also

* [torch::q_per_channel_axis](q_per_channel_axis.md)
* [torch::q_per_channel_scales](q_per_channel_scales.md)
* [torch::quantize_per_channel](quantize_per_channel.md) 