# torch::unsqueeze_multiple

Adds multiple dimensions of size 1 to a tensor at specified positions.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::unsqueeze_multiple -tensor TENSOR -dims DIMS
torch::unsqueezeMultiple -tensor TENSOR -dims DIMS
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::unsqueeze_multiple tensor dims
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor | Tensor | Required | Input tensor to unsqueeze |
| dims | List of Integers | Required | List of dimension positions where size-1 dimensions will be added |

## Description

The unsqueeze multiple operation adds dimensions of size 1 to a tensor at specified positions. This is useful for broadcasting operations, reshaping tensors for different neural network layers, and preparing tensors for matrix operations that require specific dimensionality.

**Important Note**: The dimensions are processed in descending order internally to avoid index shifting issues. For a tensor with `n` dimensions, you can add new dimensions at positions 0 through n (inclusive). When multiple dimensions are added, the valid position range changes after each addition.

## Examples

### Basic Usage
```tcl
# Create a tensor with shape [3, 4]
set tensor [torch::tensor_randn -shape {3 4} -dtype float32]

# Named parameter syntax - add dimension at position 0
set result [torch::unsqueeze_multiple -tensor $tensor -dims {0}]

# Legacy positional syntax - same operation
set result [torch::unsqueeze_multiple $tensor {0}]

# camelCase alias
set result [torch::unsqueezeMultiple -tensor $tensor -dims {0}]
```

### Add Multiple Dimensions
```tcl
# Create tensor with shape [2, 3]
set tensor [torch::tensor_randn -shape {2 3} -dtype float32]

# Add dimensions at positions 0 and 2 - becomes [1, 2, 1, 3]
set result [torch::unsqueeze_multiple -tensor $tensor -dims {0 2}]

# Equivalent legacy syntax
set result [torch::unsqueeze_multiple $tensor {0 2}]
```

### Different Parameter Orders
```tcl
# Named parameters can be in any order
set result [torch::unsqueeze_multiple -dims {1} -tensor $tensor]
```

### Neural Network Preparation Example
```tcl
# Prepare tensor for batch processing
set features 128
set sequence_length 10

# Create feature vector
set feature_vector [torch::tensor_randn -shape [list $features] -dtype float32]

# Add batch and sequence dimensions for RNN input
# Shape becomes [1, 1, 128] for [batch_size, seq_len, features]
set rnn_input [torch::unsqueeze_multiple -tensor $feature_vector -dims {0 1}]

puts "Original shape: [torch::tensor_shape $feature_vector]"
puts "RNN input shape: [torch::tensor_shape $rnn_input]"
```

### Broadcasting Preparation
```tcl
# Prepare tensors for broadcasting operations
set vector [torch::tensor_randn -shape {5} -dtype float32]
set matrix [torch::tensor_randn -shape {3 5} -dtype float32]

# Add dimension to vector for broadcasting with matrix
# Shape [5] becomes [1, 5] for broadcasting with [3, 5]
set broadcastable_vector [torch::unsqueeze_multiple -tensor $vector -dims {0}]

# Now can broadcast: [1, 5] with [3, 5] -> [3, 5]
set result [torch::tensor_add -tensor1 $matrix -tensor2 $broadcastable_vector]

puts "Broadcasting result shape: [torch::tensor_shape $result]"
```

### Attention Mechanism Example
```tcl
# Prepare tensors for attention computation
set batch_size 32
set seq_len 16
set embed_dim 64

# Create query vector for single position
set query [torch::tensor_randn -shape [list $embed_dim] -dtype float32]

# Add batch and sequence dimensions
# Shape [64] becomes [1, 1, 64] for [batch, seq, embed]
set query_expanded [torch::unsqueeze_multiple -tensor $query -dims {0 1}]

puts "Query expanded shape: [torch::tensor_shape $query_expanded]"
```

### Convolutional Layer Preparation
```tcl
# Prepare image data for CNN processing
set height 28
set width 28

# Create grayscale image data
set image_data [torch::tensor_randn -shape [list $height $width] -dtype float32]

# Add batch and channel dimensions
# Shape [28, 28] becomes [1, 1, 28, 28] for [batch, channels, height, width]
set cnn_input [torch::unsqueeze_multiple -tensor $image_data -dims {0 1}]

puts "CNN input shape: [torch::tensor_shape $cnn_input]"
```

### Scalar to Tensor Conversion
```tcl
# Convert scalar to higher-dimensional tensor
set scalar_value [torch::tensor_create -data 42.0 -dtype float32]

# Add multiple dimensions to create a 3D tensor with shape [1, 1, 1]
set tensor_3d [torch::unsqueeze_multiple -tensor $scalar_value -dims {0 1 2}]

puts "3D tensor shape: [torch::tensor_shape $tensor_3d]"
```

### Batch Processing Pipeline
```tcl
# Process multiple samples with different dimensionalities
for {set i 0} {$i < 3} {incr i} {
    # Create sample with different base shapes
    set base_shape [list [expr {2 + $i}] [expr {3 + $i}]]
    set sample [torch::tensor_randn -shape $base_shape -dtype float32]
    
    # Add batch dimension to each sample
    set batched_sample [torch::unsqueeze_multiple -tensor $sample -dims {0}]
    
    puts "Sample $i - Original: [torch::tensor_shape $sample], Batched: [torch::tensor_shape $batched_sample]"
}
```

### Time Series Data Preparation
```tcl
# Prepare time series data for RNN processing
set time_steps 100
set features 50

# Create time series data [time_steps, features]
set time_series [torch::tensor_randn -shape [list $time_steps $features] -dtype float32]

# Add batch dimension for processing
# Shape [100, 50] becomes [1, 100, 50] for [batch, time, features]
set rnn_ready [torch::unsqueeze_multiple -tensor $time_series -dims {0}]

puts "Time series shape: [torch::tensor_shape $rnn_ready]"
```

### Matrix Operations Setup
```tcl
# Prepare for batch matrix operations
set matrix_a [torch::tensor_randn -shape {3 4} -dtype float32]
set matrix_b [torch::tensor_randn -shape {4 5} -dtype float32]

# Add batch dimensions for batch matrix multiplication
set batch_a [torch::unsqueeze_multiple -tensor $matrix_a -dims {0}]
set batch_b [torch::unsqueeze_multiple -tensor $matrix_b -dims {0}]

# Now can perform batch matrix multiplication
set result [torch::tensor_bmm -tensor1 $batch_a -tensor2 $batch_b]

puts "Batch matmul result shape: [torch::tensor_shape $result]"
```

## Return Value

Returns a new tensor with the specified dimensions added. The resulting tensor contains the same data but with additional dimensions of size 1 at the specified positions.

## Notes

- **Dimension Indexing**: Dimensions are 0-indexed. For a tensor with `n` dimensions, valid positions are 0 through n (inclusive)
- **Processing Order**: Dimensions are processed in descending order internally to prevent index shifting issues
- **Shape Changes**: The total number of elements remains the same, but the tensor shape changes
- **Broadcasting**: Adding dimensions is commonly used to make tensors compatible for broadcasting operations
- **Memory Efficiency**: The operation typically returns a view of the original tensor when possible

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Dimensions list must not be empty
- Specified dimensions must be within the valid range for the tensor
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::unsqueezeMultiple` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::unsqueeze_multiple $tensor {0} → torch::unsqueeze_multiple -tensor $tensor -dims {0}
torch::unsqueeze_multiple $tensor {0 2} → torch::unsqueeze_multiple -tensor $tensor -dims {0 2}

# Modern camelCase
torch::unsqueeze_multiple $tensor {0} → torch::unsqueezeMultiple -tensor $tensor -dims {0}
```

## See Also

- `torch::squeeze_multiple` - Remove multiple dimensions of size 1
- `torch::tensor_reshape` - Change tensor shape arbitrarily
- `torch::tensor_view` - Create a view with different shape
- `torch::tensor_unsqueeze` - Add single dimension
- `torch::tensor_squeeze` - Remove single dimension
- `torch::tensor_expand` - Expand tensor to larger size
