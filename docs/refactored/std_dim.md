# torch::std_dim

Computes the standard deviation along a specified dimension of a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::std_dim tensor dim ?unbiased? ?keepdim?
```

### Named Parameter Syntax (New)
```tcl
torch::std_dim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?
torch::std_dim -tensor tensor -dim dim ?-unbiased bool? ?-keepdim bool?
```

### CamelCase Alias
```tcl
torch::stdDim tensor dim ?unbiased? ?keepdim?
torch::stdDim -input tensor -dim dim ?-unbiased bool? ?-keepdim bool?
torch::stdDim -tensor tensor -dim dim ?-unbiased bool? ?-keepdim bool?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensor` / `-input` / `-tensor` | string | Required | Name of the input tensor |
| `dim` / `-dim` | integer | Required | Dimension along which to compute standard deviation |
| `unbiased` / `-unbiased` | boolean | true | Whether to use unbiased estimation (true) or biased estimation (false) |
| `keepdim` / `-keepdim` | boolean | false | Whether to keep the reduced dimension in the output |

## Returns

Returns a new tensor handle containing the standard deviation values along the specified dimension.

## Description

The `torch::std_dim` command computes the standard deviation of elements along a specified dimension of a tensor. The standard deviation measures the amount of variation or dispersion in the data.

**Key Features:**
- **Unbiased vs Biased**: When `unbiased=true` (default), the standard deviation is computed using N-1 in the denominator (unbiased estimator). When `unbiased=false`, it uses N (biased estimator).
- **Dimension Preservation**: When `keepdim=true`, the reduced dimension is kept with size 1. When `keepdim=false` (default), the dimension is removed.
- **Mathematical Formula**: For unbiased estimation: σ = √(Σ(x - μ)² / (N-1)), where μ is the mean and N is the number of elements.

## Examples

### Basic Usage
```tcl
# Create a tensor with sample data
set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]

# Named parameter syntax - compute standard deviation along dimension 0
set result [torch::std_dim -input $tensor -dim 0]
set values [torch::tensor_to_list $result]
puts "Standard deviation: [lindex $values 0]"

# Legacy positional syntax
set result [torch::std_dim $tensor 0]
set values [torch::tensor_to_list $result]
puts "Standard deviation: [lindex $values 0]"

# camelCase alias
set result [torch::stdDim -input $tensor -dim 0]
```

### Unbiased vs Biased Estimation
```tcl
set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]

# Unbiased estimation (default, uses N-1)
set unbiased_result [torch::std_dim -input $tensor -dim 0 -unbiased 1]
set unbiased_val [lindex [torch::tensor_to_list $unbiased_result] 0]

# Biased estimation (uses N)
set biased_result [torch::std_dim -input $tensor -dim 0 -unbiased 0]
set biased_val [lindex [torch::tensor_to_list $biased_result] 0]

puts "Unbiased std: $unbiased_val"
puts "Biased std: $biased_val"
puts "Unbiased > Biased: [expr {$unbiased_val > $biased_val}]"
```

### Multi-dimensional Tensors
```tcl
# Create a 2D tensor
set tensor [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float32 -device cpu -requiresGrad true]

# Compute std along dimension 0 (rows)
set result_dim0 [torch::std_dim -input $tensor -dim 0]
set values_dim0 [torch::tensor_to_list $result_dim0]
puts "Std along dim 0: $values_dim0"

# Compute std along dimension 1 (columns)
set result_dim1 [torch::std_dim -input $tensor -dim 1]
set values_dim1 [torch::tensor_to_list $result_dim1]
puts "Std along dim 1: $values_dim1"
```

### Keeping Dimensions
```tcl
set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]

# Without keeping dimension (default)
set result1 [torch::std_dim -input $tensor -dim 0 -keepdim 0]
set shape1 [torch::tensor_shape $result1]
puts "Shape without keepdim: $shape1"

# With keeping dimension
set result2 [torch::std_dim -input $tensor -dim 0 -keepdim 1]
set shape2 [torch::tensor_shape $result2]
puts "Shape with keepdim: $shape2"
```

### Statistical Analysis Example
```tcl
# Simulate multiple samples
set samples {}
for {set i 0} {$i < 5} {incr i} {
    set sample [torch::tensorCreate -data [list [expr {rand() * 10}] [expr {rand() * 10}] [expr {rand() * 10}]] -dtype float32 -device cpu -requiresGrad true]
    lappend samples $sample
}

# Stack samples into a 2D tensor
set stacked [torch::tensorStack -tensors $samples -dim 0]

# Compute statistics
set mean [torch::mean_dim -input $stacked -dim 0]
set std [torch::std_dim -input $stacked -dim 0]

puts "Mean across samples: [torch::tensor_to_list $mean]"
puts "Std across samples: [torch::tensor_to_list $std]"
```

### Neural Network Batch Normalization Example
```tcl
# Simulate batch normalization statistics computation
set batch_size 4
set features 3

# Create batch data
set batch_data [torch::tensorCreate -data {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0} {10.0 11.0 12.0}} -dtype float32 -device cpu -requiresGrad true]

# Compute batch statistics (along batch dimension)
set batch_mean [torch::mean_dim -input $batch_data -dim 0 -keepdim 1]
set batch_std [torch::std_dim -input $batch_data -dim 0 -keepdim 1 -unbiased 0]

puts "Batch mean: [torch::tensor_to_list $batch_mean]"
puts "Batch std: [torch::tensor_to_list $batch_std]"

# Apply batch normalization
set normalized [torch::tensor_sub -input $batch_data -other $batch_mean]
set normalized [torch::tensor_div -input $normalized -other $batch_std]
```

### Data Analysis Pipeline
```tcl
# Load or create dataset
set dataset [torch::tensorCreate -data {{1.2 3.4 5.6} {2.1 4.3 6.5} {0.9 2.8 4.7} {3.2 5.1 7.3}} -dtype float32 -device cpu -requiresGrad true]

# Compute descriptive statistics
set mean [torch::mean_dim -input $dataset -dim 0]
set std [torch::std_dim -input $dataset -dim 0]
set min_val [torch::tensor_min -input $dataset -dim 0]
set max_val [torch::tensor_max -input $dataset -dim 0]

puts "Feature statistics:"
puts "  Mean: [torch::tensor_to_list $mean]"
puts "  Std:  [torch::tensor_to_list $std]"
puts "  Min:  [torch::tensor_to_list $min_val]"
puts "  Max:  [torch::tensor_to_list $max_val]"

# Z-score normalization
set z_scores [torch::tensor_sub -input $dataset -other $mean]
set z_scores [torch::tensor_div -input $z_scores -other $std]
```

## Return Value

Returns a new tensor handle containing the standard deviation values. The shape of the result depends on:
- The input tensor shape
- The specified dimension
- The `keepdim` parameter

## Notes

- **Dimension Indexing**: Dimensions are 0-indexed
- **Unbiased Estimation**: Default behavior uses N-1 in denominator for unbiased estimation
- **Numerical Stability**: For single-element tensors, the result may be NaN due to division by zero
- **Memory Efficiency**: The operation typically returns a new tensor rather than a view
- **Gradient Support**: The operation supports automatic differentiation when `requiresGrad=true`

## Error Handling

The function validates:
- Input tensor must exist and be valid
- Dimension must be within the valid range for the tensor
- Boolean parameters must be valid boolean values
- Parameter values must be provided for named syntax

## Compatibility

✅ **Backward Compatible**: All existing code using positional parameters continues to work  
✅ **Named Parameters**: New code can use clearer `-parameter value` syntax  
✅ **camelCase**: Modern `torch::stdDim` alias available  

## Migration Guide

```tcl
# Old style → New style
torch::std_dim $tensor 0 → torch::std_dim -input $tensor -dim 0
torch::std_dim $tensor 1 0 1 → torch::std_dim -input $tensor -dim 1 -unbiased 0 -keepdim 1

# Modern camelCase
torch::std_dim $tensor 0 → torch::stdDim -input $tensor -dim 0
```

## See Also

- `torch::mean_dim` - Compute mean along dimension
- `torch::var_dim` - Compute variance along dimension
- `torch::tensor_min` - Find minimum values
- `torch::tensor_max` - Find maximum values
- `torch::tensor_sum` - Sum elements along dimension 