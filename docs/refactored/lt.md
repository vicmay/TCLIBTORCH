# torch::lt

Performs element-wise less-than comparison between two tensors.

## Syntax

### Current Syntax
```tcl
torch::lt tensor1 tensor2
```

### Named Parameter Syntax  
```tcl
torch::lt -input1 tensor1 -input2 tensor2
```

### camelCase Alias
```tcl
torch::Lt -input1 tensor1 -input2 tensor2
```

All syntaxes are fully supported and equivalent.

## Parameters

### Named Parameters
- `-input1` (required): First tensor for comparison
- `-input2` (required): Second tensor for comparison  

### Alternative Parameter Names
- `-tensor1` (alternative to `-input1`): First tensor for comparison
- `-tensor2` (alternative to `-tensor2`): Second tensor for comparison

### Positional Parameters
When using positional syntax, the parameters are provided in the following order:
1. `tensor1` (required): First tensor for comparison
2. `tensor2` (required): Second tensor for comparison

## Return Value

Returns a new tensor handle containing the boolean result of the element-wise less-than comparison.

## Examples

### Basic Usage
```tcl
# Create test tensors
set a [torch::tensor_create {1.0 2.0 3.0} float32]
set b [torch::tensor_create {2.0 2.0 2.0} float32]

# Using positional syntax
set result1 [torch::lt $a $b]
;# Returns tensor with elements [true, false, false]

# Using named parameters
set result2 [torch::lt -input1 $a -input2 $b]
;# Same as above

# Using camelCase alias
set result3 [torch::Lt -input1 $a -input2 $b]
;# Same as above
```

### Matrix Comparison
```tcl
# Create 2x2 matrices
set matrix1 [torch::tensor_create {1.0 3.0 2.0 4.0} float32]
set matrix1 [torch::tensor_reshape $matrix1 {2 2}]
set matrix2 [torch::tensor_create {2.0 2.0 3.0 3.0} float32]
set matrix2 [torch::tensor_reshape $matrix2 {2 2}]

# Compare matrices
set result [torch::lt $matrix1 $matrix2]
;# Returns tensor with shape [2, 2] containing boolean values
```

### Different Data Types
```tcl
# Integer tensors
set a [torch::tensor_create {1 2 3} int32]
set b [torch::tensor_create {2 2 2} int32]
set result [torch::lt $a $b]

# Float64 tensors
set a [torch::tensor_create {1.0 2.0 3.0} float64]
set b [torch::tensor_create {2.0 2.0 2.0} float64]
set result [torch::lt $a $b]
```

### Using Alternative Parameter Names
```tcl
set a [torch::tensor_create {1.0 2.0 3.0} float32]
set b [torch::tensor_create {2.0 2.0 2.0} float32]

# Using -tensor1 and -tensor2
set result [torch::lt -tensor1 $a -tensor2 $b]

# Mixed parameter names
set result [torch::lt -input1 $a -tensor2 $b]
```

## Broadcasting

The `torch::lt` function supports broadcasting. Tensors with different shapes can be compared if they are broadcastable according to PyTorch broadcasting rules.

```tcl
# Broadcasting example
set a [torch::tensor_create {1.0 2.0 3.0} float32]
set b [torch::tensor_create {2.0} float32]
set result [torch::lt $a $b]
;# Compares each element of a with 2.0
```

## Notes

- The function returns a boolean tensor with the same shape as the broadcasted input tensors
- Both input tensors must have compatible data types
- The result tensor will have boolean (uint8) data type
- Element-wise comparison: `result[i] = input1[i] < input2[i]`
- Broadcasting follows PyTorch's standard broadcasting rules

## Error Handling

The function will throw an error if:
- Required parameters are missing
- Invalid tensor handles are provided
- Unknown parameter names are used
- Parameter values are missing

## Migration Guide

### From Positional to Named Parameters
```tcl
# Old (still supported)
set result [torch::lt $tensor1 $tensor2]

# New named parameter syntax
set result [torch::lt -input1 $tensor1 -input2 $tensor2]

# Or using camelCase alias
set result [torch::Lt -input1 $tensor1 -input2 $tensor2]
```

## Related Functions

- `torch::le` - Less than or equal to comparison
- `torch::gt` - Greater than comparison
- `torch::ge` - Greater than or equal to comparison
- `torch::eq` - Equal to comparison
- `torch::ne` - Not equal to comparison

## Technical Details

- **Function**: Element-wise less-than comparison
- **Backend**: Calls PyTorch's `tensor.lt()` method
- **Broadcasting**: Supported according to PyTorch rules
- **Memory**: Creates new tensor, doesn't modify input tensors
- **Thread Safety**: Safe for concurrent use with different tensors 