# torch::tensor_var

Computes the variance of elements in a tensor along specified dimensions.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_var tensor_handle ?dim? ?unbiased? ?keepdim?
```

### Named Parameters (New)
```tcl
torch::tensor_var -input tensor_handle ?-dim dim? ?-unbiased unbiased? ?-keepdim keepdim?
```

### CamelCase Alias
```tcl
torch::tensorVar -input tensor_handle ?-dim dim? ?-unbiased unbiased? ?-keepdim keepdim?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | tensor_handle | required | Input tensor |
| `dim` | integer or list | all dimensions | Dimension(s) along which to compute variance |
| `unbiased` | boolean | true | If true, use Bessel's correction (n-1), otherwise use n |
| `keepdim` | boolean | false | Whether to keep the reduced dimensions |

## Return Value

Returns a tensor handle containing the variance values.

## Examples

### Basic Usage
```tcl
# Create a tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float]

# Compute variance of all elements
set var [torch::tensor_var $t]
puts "Variance: [torch::tensor_item $var]"

# Clean up
torch::tensor_destroy $t
torch::tensor_destroy $var
```

### Along Specific Dimension
```tcl
# Create a 2D tensor
set t [torch::tensor_create -data {{1.0 2.0 3.0} {4.0 5.0 6.0}} -dtype float]

# Compute variance along dimension 0 (rows)
set var_dim0 [torch::tensor_var $t -dim 0]
puts "Variance along dim 0: [torch::tensor_to_list $var_dim0]"

# Compute variance along dimension 1 (columns)
set var_dim1 [torch::tensor_var $t -dim 1]
puts "Variance along dim 1: [torch::tensor_to_list $var_dim1]"

# Clean up
torch::tensor_destroy $t
torch::tensor_destroy $var_dim0
torch::tensor_destroy $var_dim1
```

### With Named Parameters
```tcl
# Create a tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float]

# Compute variance with biased estimator
set var_biased [torch::tensor_var -input $t -unbiased false]
puts "Biased variance: [torch::tensor_item $var_biased]"

# Compute variance and keep dimensions
set var_keepdim [torch::tensor_var -input $t -keepdim true]
puts "Variance with keepdim: [torch::tensor_to_list $var_keepdim]"

# Clean up
torch::tensor_destroy $t
torch::tensor_destroy $var_biased
torch::tensor_destroy $var_keepdim
```

### Using CamelCase Alias
```tcl
# Create a tensor
set t [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float]

# Use camelCase alias
set var [torch::tensorVar -input $t -dim 0 -unbiased true -keepdim false]
puts "Variance: [torch::tensor_item $var]"

# Clean up
torch::tensor_destroy $t
torch::tensor_destroy $var
```

## Migration from Positional to Named Parameters

### Before (Positional)
```tcl
# Old positional syntax
set var [torch::tensor_var $tensor 0 true false]
```

### After (Named)
```tcl
# New named syntax
set var [torch::tensor_var -input $tensor -dim 0 -unbiased true -keepdim false]
```

## Error Handling

The command will throw an error if:
- The input tensor handle is invalid
- The specified dimension is out of bounds
- Required parameters are missing

## Notes

- When `dim` is not specified, variance is computed over all elements
- The `unbiased` parameter determines whether to use Bessel's correction (n-1) or not (n)
- When `keepdim` is true, the output tensor retains the reduced dimensions with size 1
- For scalar tensors, the result is also a scalar tensor
- The command supports both CPU and CUDA tensors 