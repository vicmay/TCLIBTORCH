# torch::squeeze_multiple

Removes multiple dimensions of size 1 from a tensor, either all size-1 dimensions or specific dimensions.

## Syntax

### New Syntax (Named Parameters)
```tcl
torch::squeeze_multiple -tensor TENSOR [-dims DIMS]
torch::squeezeMultiple -tensor TENSOR [-dims DIMS]
```

### Legacy Syntax (Positional Parameters) 
```tcl
torch::squeeze_multiple tensor [dims]
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| tensor | Tensor | Required | Input tensor to squeeze |
| dims | List of Integers | None | List of dimension indices to squeeze (must be size 1). If not provided, all size-1 dimensions are squeezed |

## Description

The squeeze multiple operation removes dimensions of size 1 from a tensor. This is useful for removing unnecessary dimensions and reshaping tensors for different operations. When specific dimensions are provided, only those dimensions are squeezed (and they must have size 1). When no dimensions are specified, all dimensions of size 1 are automatically removed.

**Important Note**: When squeezing multiple specific dimensions, the operation is performed sequentially. This means that after squeezing an earlier dimension, the indices of later dimensions may shift. It's recommended to either squeeze dimensions in descending order or use the "squeeze all" mode when possible.

## Examples

### Basic Usage
```tcl
# Create a tensor with multiple size-1 dimensions
set tensor [torch::tensor_randn -shape {1 4 1 5 1} -dtype float32]

# Named parameter syntax - squeeze all size-1 dimensions
set result [torch::squeeze_multiple -tensor $tensor]

# Legacy positional syntax - squeeze all size-1 dimensions
set result [torch::squeeze_multiple $tensor]

# camelCase alias
set result [torch::squeezeMultiple -tensor $tensor]
```

### Squeeze Specific Dimensions
```tcl
# Create tensor with shape [1, 4, 1, 5, 1]
set tensor [torch::tensor_randn -shape {1 4 1 5 1} -dtype float32]

# Squeeze dimensions 0 and 2 (which have size 1)
set result [torch::squeeze_multiple -tensor $tensor -dims {0 2}]

# Equivalent legacy syntax
set result [torch::squeeze_multiple $tensor {0 2}]
```

### Different Parameter Orders
```tcl
# Named parameters can be in any order
set result [torch::squeeze_multiple -dims {0 2} -tensor $tensor]
```

### Neural Network Layer Example
```tcl
# Common usage in neural networks to remove batch dimensions
set batch_size 1
set features 128
set sequence_length 10

# Create output from a layer with singleton batch dimension
set layer_output [torch::tensor_randn -shape [list $batch_size $sequence_length $features] -dtype float32]

# Remove the singleton batch dimension for further processing
set squeezed_output [torch::squeeze_multiple -tensor $layer_output -dims {0}]

puts "Original shape: [torch::tensor_shape $layer_output]"
puts "Squeezed shape: [torch::tensor_shape $squeezed_output]"
```

### Attention Mechanism Example
```tcl
# Remove extra dimensions from attention outputs
set batch_size 1
set num_heads 8
set seq_len 32
set head_dim 64

# Attention output with singleton dimensions
set attention_output [torch::tensor_randn -shape [list $batch_size 1 $num_heads $seq_len $head_dim] -dtype float32]

# Remove singleton dimensions (batch and extra dimension)
set cleaned_output [torch::squeeze_multiple -tensor $attention_output -dims {0 1}]

puts "Attention output shape: [torch::tensor_shape $cleaned_output]"
```

### Automatic Squeeze All
```tcl
# When you want to remove all size-1 dimensions automatically
set tensor_with_ones [torch::tensor_randn -shape {1 3 1 1 4 1} -dtype float32]

# Remove all size-1 dimensions automatically
set compact_tensor [torch::squeeze_multiple -tensor $tensor_with_ones]

puts "Original shape: [torch::tensor_shape $tensor_with_ones]"
puts "Compact shape: [torch::tensor_shape $compact_tensor]"
```

### Batch Processing
```tcl
# Process multiple tensors with size-1 dimensions
for {set i 0} {$i < 3} {incr i} {
    # Create tensors with different patterns of size-1 dimensions
    set tensor [torch::tensor_randn -shape [list 1 [expr {$i + 2}] 1 4] -dtype float32]
    
    # Squeeze all size-1 dimensions
    set squeezed [torch::squeeze_multiple -tensor $tensor]
    
    puts "Batch $i - Original: [torch::tensor_shape $tensor], Squeezed: [torch::tensor_shape $squeezed]"
}
```

### Conditional Squeezing
```tcl
# Squeeze different dimensions based on tensor properties
set tensor [torch::tensor_randn -shape {1 1 3 4} -dtype float32]

# Get tensor shape to decide which dimensions to squeeze
set shape [torch::tensor_shape $tensor]

# Build list of dimensions to squeeze (size-1 dimensions)
set dims_to_squeeze {}
for {set i 0} {$i < [llength $shape]} {incr i} {
    if {[lindex $shape $i] == 1} {
        lappend dims_to_squeeze $i
    }
}

if {[llength $dims_to_squeeze] > 0} {
    set result [torch::squeeze_multiple -tensor $tensor -dims $dims_to_squeeze]
} else {
    set result $tensor
}

puts "Squeezed tensor shape: [torch::tensor_shape $result]"
```

## Return Value

Returns a new tensor with the specified dimensions (or all size-1 dimensions) removed. The resulting tensor contains the same data but with a modified shape.

## Notes

- **Dimension Indexing**: Dimensions are 0-indexed. When squeezing multiple specific dimensions, be aware that indices may shift after each squeeze operation
- **Size-1 Requirement**: Only dimensions with size 1 can be squeezed. Attempting to squeeze a dimension with size > 1 will result in an error
- **Shape Preservation**: The total number of elements in the tensor remains the same, only the shape changes
- **Sequential Processing**: When multiple dimensions are specified, they are squeezed one by one, which can affect dimension indices
- **Memory Efficiency**: The operation typically returns a view of the original tensor when possible

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Specified dimensions must be within the valid range for the tensor
- Dimensions to be squeezed must have size 1
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::squeezeMultiple` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::squeeze_multiple $tensor → torch::squeeze_multiple -tensor $tensor
torch::squeeze_multiple $tensor {0 2} → torch::squeeze_multiple -tensor $tensor -dims {0 2}

# Modern camelCase
torch::squeeze_multiple $tensor → torch::squeezeMultiple -tensor $tensor
```

## See Also

- `torch::unsqueeze_multiple` - Add multiple dimensions of size 1
- `torch::tensor_reshape` - Change tensor shape arbitrarily
- `torch::tensor_view` - Create a view with different shape
- `torch::tensor_squeeze` - Squeeze single dimension
- `torch::tensor_unsqueeze` - Add single dimension
