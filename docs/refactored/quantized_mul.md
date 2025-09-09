# torch::quantized_mul

Performs element-wise multiplication on quantized tensors with quantization parameters.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::quantized_mul -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT
torch::quantizedMul -tensor1 TENSOR -tensor2 TENSOR -scale DOUBLE -zeroPoint INT
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::quantized_mul tensor1 tensor2 scale zero_point
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor1 | Tensor | Required | First input tensor |
| tensor2 | Tensor | Required | Second input tensor |
| scale | Double | Required | Scale factor for quantization |
| zeroPoint | Integer | Required | Zero point offset for quantization |

## Description

The quantized multiplication operation performs element-wise multiplication on tensors while considering quantization parameters. This function is designed for use with quantized neural networks where computational efficiency is crucial.

The operation computes: `result = tensor1 * tensor2`

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
set result [torch::quantized_mul -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]

# Legacy positional syntax  
set result [torch::quantized_mul $tensor1 $tensor2 $scale $zero_point]

# camelCase alias
set result [torch::quantizedMul -tensor1 $tensor1 -tensor2 $tensor2 -scale $scale -zeroPoint $zero_point]
```

### Different Parameter Orders
```tcl
# Named parameters can be in any order
set result [torch::quantized_mul -scale $scale -tensor2 $tensor2 -zeroPoint $zero_point -tensor1 $tensor1]
```

### Quantized Neural Network Example
```tcl
# Typical usage in quantized neural networks
set batch_size 32
set features 128

# Create weight matrix and input activations
set weights [torch::tensor_randn -shape [list $features $features] -dtype float32]
set activations [torch::tensor_randn -shape [list $batch_size $features] -dtype float32]

# Quantization parameters for the layer
set scale 0.01
set zero_point 128

# Element-wise multiplication with quantization
set output [torch::quantized_mul -tensor1 $activations -tensor2 $weights -scale $scale -zeroPoint $zero_point]

puts "Output shape: [torch::tensor_shape $output]"
```

### Batch Processing with Different Scales
```tcl
# Process multiple batches with different quantization parameters
for {set i 0} {$i < 3} {incr i} {
    set batch_tensor1 [torch::tensor_randn -shape {8 16} -dtype float32]
    set batch_tensor2 [torch::tensor_randn -shape {8 16} -dtype float32]
    
    # Different scale per batch
    set scale [expr {0.1 * ($i + 1)}]
    set zero_point 0
    
    set result [torch::quantized_mul -tensor1 $batch_tensor1 -tensor2 $batch_tensor2 -scale $scale -zeroPoint $zero_point]
    
    puts "Batch $i (scale=$scale) processed: $result"
}
```

### Broadcasting Support
```tcl
# Create tensors with compatible shapes for broadcasting
set tensor_matrix [torch::tensor_randn -shape {4 6} -dtype float32]
set tensor_vector [torch::tensor_randn -shape {6} -dtype float32]

# Quantized multiplication with broadcasting
set result [torch::quantized_mul -tensor1 $tensor_matrix -tensor2 $tensor_vector -scale 0.05 -zeroPoint 64]

puts "Broadcasting result: $result"
```

### Attention Mechanism Example
```tcl
# Quantized attention computation
set query [torch::tensor_randn -shape {32 64} -dtype float32]
set key [torch::tensor_randn -shape {32 64} -dtype float32]

# Quantization parameters for attention
set attention_scale 0.125  # 1/8 for attention scaling
set zero_point 0

# Compute element-wise attention scores
set attention_scores [torch::quantized_mul -tensor1 $query -tensor2 $key -scale $attention_scale -zeroPoint $zero_point]

puts "Attention scores: $attention_scores"
```

## Return Value

Returns a tensor containing the result of the quantized multiplication operation. The output tensor maintains the quantization characteristics as handled internally by PyTorch.

## Notes

- **Quantization Handling**: PyTorch handles quantization mechanics internally; the scale and zero_point parameters inform the operation about quantization characteristics
- **Tensor Compatibility**: Input tensors should be compatible for element-wise operations (same shape or broadcastable)
- **Broadcasting**: Supports PyTorch's broadcasting rules for tensor shapes
- **Performance**: Quantized operations are designed for improved computational efficiency in deployed models
- **Precision**: Unlike regular multiplication, quantized multiplication considers the quantization parameters to maintain numerical precision

## Error Handling

The function validates:
- Both input tensors must exist and be valid
- Scale parameter must be a valid double precision number
- Zero point parameter must be a valid integer
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::quantizedMul` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::quantized_mul $t1 $t2 0.1 0 → torch::quantized_mul -tensor1 $t1 -tensor2 $t2 -scale 0.1 -zeroPoint 0

# Modern camelCase
torch::quantized_mul $t1 $t2 0.1 0 → torch::quantizedMul -tensor1 $t1 -tensor2 $t2 -scale 0.1 -zeroPoint 0
```

## See Also

- `torch::quantized_add` - Quantized addition
- `torch::tensor_mul` - Regular tensor multiplication
- `torch::quantize_per_tensor` - Tensor quantization
- `torch::dequantize` - Tensor dequantization
- `torch::tensor_matmul` - Matrix multiplication 