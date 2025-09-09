# torch::quantized_add

Performs element-wise addition on quantized tensors with quantization parameters.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::quantized_add -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT [-alpha DOUBLE]
torch::quantizedAdd -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT [-alpha DOUBLE]
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::quantized_add tensor1 tensor2 scale zero_point [alpha]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor1 | Tensor | Required | First input tensor |
| tensor2 | Tensor | Required | Second input tensor |
| scale | Double | Required | Scale factor for quantization |
| zeroPoint | Integer | Required | Zero point offset for quantization |
| alpha | Double | 1.0 | Scalar multiplier for tensor2 (result = tensor1 + alpha * tensor2) |

## Description

The quantized add operation performs element-wise addition on tensors while considering quantization parameters. This function is designed for use with quantized neural networks where computational efficiency is crucial.

The operation computes: `result = tensor1 + alpha * tensor2`

The scale and zero_point parameters are used to maintain proper quantization during the operation, though PyTorch handles the internal quantization mechanics automatically.

## Examples

### Basic Usage
```tcl
# Create input tensors
set tensor1 [torch::tensor_randn -shape {4 5} -dtype float32]
set tensor2 [torch::tensor_randn -shape {4 5} -dtype float32]

# Define quantization parameters
set scale 0.1
set zero_point 0

# Named parameter syntax
set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]

# Legacy positional syntax  
set result [torch::quantized_add $tensor1 $tensor2 $scale $zero_point]

# camelCase alias
set result [torch::quantizedAdd -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]
```

### With Alpha Parameter
```tcl
# Add with scaling factor
set result [torch::quantized_add -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point -alpha 2.0]

# Equivalent legacy syntax
set result [torch::quantized_add $tensor1 $tensor2 $scale $zero_point 2.0]
```

### Different Parameter Orders
```tcl
# Named parameters can be in any order
set result [torch::quantized_add -scale $scale -tensor2 $tensor2 -zeroPoint $zero_point -tensor1 $tensor1 -alpha 0.5]
```

### Quantized Neural Network Example
```tcl
# Typical usage in quantized neural networks
set batch_size 32
set features 128

# Create layer activations
set input_activations [torch::tensor_randn -shape [list $batch_size $features] -dtype float32]
set bias_tensor [torch::tensor_randn -shape [list $features] -dtype float32]

# Quantization parameters for the layer
set scale 0.01
set zero_point 128

# Add bias to activations with quantization
set output [torch::quantized_add -tensor1 $input_activations -tensor2 $bias_tensor -scale $scale -zeroPoint $zero_point]

puts "Output shape: [torch::tensor_shape $output]"
```

### Batch Processing
```tcl
# Process multiple batches with quantized addition
for {set i 0} {$i < 5} {incr i} {
    set batch_tensor1 [torch::tensor_randn -shape {8 16} -dtype float32]
    set batch_tensor2 [torch::tensor_randn -shape {8 16} -dtype float32]
    
    set result [torch::quantized_add -tensor1 $batch_tensor1 -tensor2 $batch_tensor2 -scale 0.05 -zeroPoint 0 -alpha 1.5]
    
    puts "Batch $i processed: $result"
}
```

## Return Value

Returns a tensor containing the result of the quantized addition operation. The output tensor maintains the quantization characteristics as handled internally by PyTorch.

## Notes

- **Quantization Handling**: PyTorch handles quantization mechanics internally; the scale and zero_point parameters inform the operation about quantization characteristics
- **Tensor Compatibility**: Input tensors should be compatible for element-wise operations (same shape or broadcastable)
- **Alpha Parameter**: The alpha parameter scales the second tensor before addition, useful for weighted combinations
- **Performance**: Quantized operations are designed for improved computational efficiency in deployed models

## Error Handling

The function validates:
- Both input tensors must exist and be valid
- Scale parameter must be a valid double precision number
- Zero point parameter must be a valid integer
- Alpha parameter (if provided) must be a valid double precision number
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::quantizedAdd` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::quantized_add $t1 $t2 0.1 0 → torch::quantized_add -tensor1 $t1 -tensor2 $t2 -scale 0.1 -zeroPoint 0
torch::quantized_add $t1 $t2 0.1 0 2.0 → torch::quantized_add -tensor1 $t1 -tensor2 $t2 -scale 0.1 -zeroPoint 0 -alpha 2.0

# Modern camelCase
torch::quantized_add $t1 $t2 0.1 0 → torch::quantizedAdd -tensor1 $t1 -tensor2 $t2 -scale 0.1 -zeroPoint 0
```

## See Also

- `torch::quantized_mul` - Quantized multiplication
- `torch::tensor_add` - Regular tensor addition
- `torch::quantize_per_tensor` - Tensor quantization
- `torch::dequantize` - Tensor dequantization 