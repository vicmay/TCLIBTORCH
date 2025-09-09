# torch::sparse_log_softmax

Applies the log softmax function to a sparse tensor along a specified dimension.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::sparse_log_softmax -input TENSOR -dim INT
torch::sparseLogSoftmax -input TENSOR -dim INT
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::sparse_log_softmax tensor dim
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | Tensor | Required | Input tensor (sparse or dense) |
| dim | Integer | Required | Dimension along which to compute log softmax |

## Description

The sparse log softmax operation computes the logarithm of the softmax function along a specified dimension of a tensor. This operation is particularly useful in sparse neural networks and when working with sparse probability distributions.

The operation computes:
```
log_softmax(x) = log(softmax(x))
                = log(exp(x) / sum(exp(x)))
                = x - log(sum(exp(x)))
```

The operation is computed in a numerically stable way and is optimized for sparse tensors.

## Examples

### Basic Usage
```tcl
# Create input tensor
set tensor [torch::tensor_randn -shape {3 4} -dtype float32]

# Named parameter syntax
set result [torch::sparse_log_softmax -input $tensor -dim 1]

# Legacy positional syntax  
set result [torch::sparse_log_softmax $tensor 1]

# camelCase alias
set result [torch::sparseLogSoftmax -input $tensor -dim 1]
```

### Different Dimensions
```tcl
# Apply along different dimensions
set result_dim0 [torch::sparse_log_softmax -input $tensor -dim 0]
set result_dim1 [torch::sparse_log_softmax -input $tensor -dim 1]

puts "Result along dim 0: $result_dim0"
puts "Result along dim 1: $result_dim1"
```

### Neural Network Example
```tcl
# Typical usage in neural networks
set batch_size 32
set num_classes 10

# Create logits tensor
set logits [torch::tensor_randn -shape [list $batch_size $num_classes] -dtype float32]

# Apply log_softmax for numerical stability
set log_probs [torch::sparse_log_softmax -input $logits -dim 1]

puts "Log probabilities shape: [torch::tensor_shape $log_probs]"
```

### Working with Sparse Tensors
```tcl
# Create sparse tensor
set values [torch::tensor_randn -shape {5} -dtype float32]
set indices [torch::tensor_create {0 1 2 3 4} {5} int64]
set sparse_tensor [torch::sparse_coo -values $values -indices $indices -size {10}]

# Apply log_softmax
set result [torch::sparse_log_softmax -input $sparse_tensor -dim 0]

puts "Sparse log softmax result: $result"
```

### Negative Dimension Indexing
```tcl
# Using negative dimension for last dimension
set tensor [torch::tensor_randn -shape {2 3 4} -dtype float32]
set result [torch::sparse_log_softmax -input $tensor -dim -1]

puts "Result using negative indexing: $result"
```

## Return Value

Returns a tensor containing the result of the log softmax operation. The output tensor has the same shape as the input tensor.

## Notes

- **Numerical Stability**: The implementation is numerically stable, avoiding overflow/underflow issues
- **Sparsity**: Works efficiently with both sparse and dense tensors
- **Dimension**: The specified dimension must be valid for the input tensor shape
- **Negative Indexing**: Supports negative dimension values for reverse indexing
- **Performance**: Optimized for sparse tensor operations

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Dimension parameter must be a valid integer
- Dimension must be within the valid range for the tensor's shape

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::sparseLogSoftmax` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::sparse_log_softmax $t 1 → torch::sparse_log_softmax -input $t -dim 1

# Modern camelCase
torch::sparse_log_softmax $t 1 → torch::sparseLogSoftmax -input $t -dim 1
```

## See Also

- `torch::sparse_softmax` - Regular softmax for sparse tensors
- `torch::log_softmax` - Log softmax for dense tensors
- `torch::sparse_add` - Sparse tensor addition
- `torch::sparse_mm` - Sparse matrix multiplication
- `torch::sparse_to_dense` - Convert sparse to dense tensor
``` 