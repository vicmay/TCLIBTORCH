# torch::sparse_add

Performs element-wise addition on sparse tensors with optional scaling factor.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::sparse_add -tensor1 TENSOR -tensor2 TENSOR [-alpha DOUBLE]
torch::sparseAdd -tensor1 TENSOR -tensor2 TENSOR [-alpha DOUBLE]
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::sparse_add tensor1 tensor2 [alpha]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor1 | Tensor | Required | First input tensor (sparse or dense) |
| tensor2 | Tensor | Required | Second input tensor (sparse or dense) |
| alpha | Double | 1.0 | Scalar multiplier for tensor2 (result = tensor1 + alpha * tensor2) |

## Description

The sparse addition operation performs element-wise addition on tensors, optimized for sparse tensor operations. This function works with both sparse and dense tensors, making it versatile for mixed-precision operations and sparse neural networks.

The operation computes: `result = tensor1 + alpha * tensor2`

PyTorch handles the sparse tensor mechanics internally, ensuring efficient computation when working with sparse data structures.

## Examples

### Basic Usage
```tcl
# Create input tensors (can be sparse or dense)
set tensor1 [torch::tensor_randn -shape {4 5} -dtype float32]
set tensor2 [torch::tensor_randn -shape {4 5} -dtype float32]

# Named parameter syntax
set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2]

# Legacy positional syntax  
set result [torch::sparse_add $tensor1 $tensor2]

# camelCase alias
set result [torch::sparseAdd -tensor1 $tensor1 -tensor2 $tensor2]
```

### With Alpha Parameter
```tcl
# Add with scaling factor
set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha 2.0]

# Equivalent legacy syntax
set result [torch::sparse_add $tensor1 $tensor2 2.0]
```

### Different Parameter Orders
```tcl
# Named parameters can be in any order
set result [torch::sparse_add -alpha 0.5 -tensor2 $tensor2 -tensor1 $tensor1]
```

### Sparse Neural Network Example
```tcl
# Typical usage in sparse neural networks
set batch_size 32
set features 1024

# Create sparse weight matrix (many zeros)
set sparse_weights [torch::tensor_randn -shape [list $features $features] -dtype float32]
# Create input activations
set activations [torch::tensor_randn -shape [list $batch_size $features] -dtype float32]

# Add sparse weights to activations
set output [torch::sparse_add -tensor1 $activations -tensor2 $sparse_weights -alpha 0.1]

puts "Output shape: [torch::tensor_shape $output]"
```

### Gradient Accumulation
```tcl
# Accumulate gradients with different learning rates
set current_gradients [torch::tensor_randn -shape {10 20} -dtype float32]
set accumulated_gradients [torch::tensor_randn -shape {10 20} -dtype float32]

# Add new gradients with momentum factor
set learning_rate 0.01
set momentum 0.9

set updated_gradients [torch::sparse_add -tensor1 $accumulated_gradients -tensor2 $current_gradients -alpha $learning_rate]

puts "Updated gradients: $updated_gradients"
```

### Broadcasting with Sparse Operations
```tcl
# Create tensors with compatible shapes for broadcasting
set sparse_matrix [torch::tensor_randn -shape {4 6} -dtype float32]
set bias_vector [torch::tensor_randn -shape {6} -dtype float32]

# Add bias vector to each row of the matrix
set result [torch::sparse_add -tensor1 $sparse_matrix -tensor2 $bias_vector]

puts "Broadcasting result: $result"
```

### Attention Mechanism with Sparse Operations
```tcl
# Sparse attention computation
set attention_weights [torch::tensor_randn -shape {32 64} -dtype float32]
set residual_connection [torch::tensor_randn -shape {32 64} -dtype float32]

# Add residual connection with attention scaling
set attention_scale 0.1
set output [torch::sparse_add -tensor1 $residual_connection -tensor2 $attention_weights -alpha $attention_scale]

puts "Sparse attention output: $output"
```

### Negative Scaling Example
```tcl
# Subtract by using negative alpha
set tensor1 [torch::tensor_randn -shape {3 3} -dtype float32]
set tensor2 [torch::tensor_randn -shape {3 3} -dtype float32]

# Compute tensor1 - 0.5 * tensor2
set result [torch::sparse_add -tensor1 $tensor1 -tensor2 $tensor2 -alpha -0.5]

puts "Subtraction result: $result"
```

## Return Value

Returns a tensor containing the result of the sparse addition operation. The output tensor type (sparse or dense) depends on the input tensor types and PyTorch's internal optimization decisions.

## Notes

- **Sparse Optimization**: PyTorch automatically optimizes operations when sparse tensors are involved
- **Tensor Compatibility**: Input tensors should be compatible for element-wise operations (same shape or broadcastable)
- **Broadcasting**: Supports PyTorch's broadcasting rules for tensor shapes
- **Performance**: Particularly efficient when working with sparse tensors that have many zero values
- **Alpha Parameter**: The alpha parameter scales the second tensor before addition, useful for weighted combinations and momentum-based updates

## Error Handling

The function validates:
- Both input tensors must exist and be valid
- Alpha parameter (if provided) must be a valid double precision number
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::sparseAdd` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::sparse_add $t1 $t2 → torch::sparse_add -tensor1 $t1 -tensor2 $t2
torch::sparse_add $t1 $t2 2.0 → torch::sparse_add -tensor1 $t1 -tensor2 $t2 -alpha 2.0

# Modern camelCase
torch::sparse_add $t1 $t2 → torch::sparseAdd -tensor1 $t1 -tensor2 $t2
```

## See Also

- `torch::tensor_add` - Regular tensor addition
- `torch::sparse_mm` - Sparse matrix multiplication
- `torch::sparse_sum` - Sparse tensor summation
- `torch::sparse_to_dense` - Convert sparse to dense tensor
- `torch::sparse_coo` - Create sparse COO tensor
