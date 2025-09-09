# torch::q_zero_point

Gets the zero point for a quantized tensor.

## Syntax

```tcl
torch::q_zero_point quantized_tensor
torch::qZeroPoint quantized_tensor    ;# camelCase alias
```

## Description

The `q_zero_point` command returns the zero point used for quantizing a tensor. The zero point represents the quantized value that corresponds to the real value 0.

For a quantized tensor, the relationship between quantized and real values is:
```
real_value = scale * (quantized_value - zero_point)
```

When `real_value = 0`, then `quantized_value = zero_point`.

## Arguments

* `quantized_tensor` (tensor) - A quantized tensor

## Return Value

Returns a 64-bit signed integer representing the zero point used for quantization.

## Example

```tcl
# Create a float tensor to quantize
set input [torch::randn [list 2 3 4]]

# Quantize the tensor per-tensor
set scale 0.1
set zero_point 128
set quantized [torch::quantize_per_tensor $input $scale $zero_point qint8]

# Get the zero point
set retrieved_zero_point [torch::q_zero_point $quantized]
puts "Zero point: $retrieved_zero_point"  ;# Should be 128

# Using camelCase alias
set retrieved_zero_point2 [torch::qZeroPoint $quantized]
puts "Zero point (camelCase): $retrieved_zero_point2"  ;# Should be 128
```

## Error Handling

* Throws error if the tensor handle is invalid
* Throws error if the tensor is not quantized

## See Also

* [torch::q_scale](q_scale.md) - Get scale factor of quantized tensor
* [torch::quantize_per_tensor](quantize_per_tensor.md) - Per-tensor quantization
* [torch::dequantize](dequantize.md) - Dequantize a tensor 