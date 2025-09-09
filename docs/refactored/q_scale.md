# torch::q_scale

Gets the scale factor for a quantized tensor.

## Syntax

```tcl
torch::q_scale quantized_tensor
torch::qScale quantized_tensor    ;# camelCase alias
```

## Description

The `q_scale` command returns the scale factor used for quantizing a tensor. This scale factor is used to map quantized integer values back to floating-point values during dequantization.

For a quantized tensor, the relationship between quantized and real values is:
```
real_value = scale * (quantized_value - zero_point)
```

## Arguments

* `quantized_tensor` (tensor) - A quantized tensor

## Return Value

Returns a double-precision floating-point number representing the scale factor used for quantization.

## Example

```tcl
# Create a float tensor to quantize
set input [torch::randn [list 2 3 4]]

# Quantize the tensor per-tensor
set scale 0.1
set zero_point 128
set quantized [torch::quantize_per_tensor $input $scale $zero_point qint8]

# Get the scale factor
set retrieved_scale [torch::q_scale $quantized]
puts "Scale factor: $retrieved_scale"  ;# Should be 0.1

# Using camelCase alias
set retrieved_scale2 [torch::qScale $quantized]
puts "Scale factor (camelCase): $retrieved_scale2"  ;# Should be 0.1
```

## Error Handling

* Throws error if the tensor handle is invalid
* Throws error if the tensor is not quantized

## See Also

* [torch::q_zero_point](q_zero_point.md) - Get zero point of quantized tensor
* [torch::quantize_per_tensor](quantize_per_tensor.md) - Per-tensor quantization
* [torch::dequantize](dequantize.md) - Dequantize a tensor 