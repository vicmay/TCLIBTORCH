# torch::gt

Element-wise greater-than comparison of tensors.

## Syntax

### Positional Syntax (Original)
```tcl
torch::gt tensor1 tensor2
```

### Named Parameter Syntax (New)
```tcl
torch::gt -input1 tensor1 -input2 tensor2
torch::gt -tensor1 tensor1 -tensor2 tensor2
```

### camelCase Alias
```tcl
torch::Gt tensor1 tensor2
torch::Gt -input1 tensor1 -input2 tensor2
```

## Description

The `torch::gt` command performs element-wise greater-than comparison between two tensors. It returns a tensor of the same shape as the input tensors, containing boolean values (1 for true, 0 for false) indicating whether each element of the first tensor is greater than the corresponding element of the second tensor.

## Parameters

### Positional Parameters
- `tensor1` - First input tensor handle
- `tensor2` - Second input tensor handle

### Named Parameters
- `-input1` or `-tensor1` - First input tensor handle
- `-input2` or `-tensor2` - Second input tensor handle

## Returns

A tensor handle containing the element-wise comparison results. The output tensor has the same shape as the input tensors and contains boolean values (1 for true, 0 for false).

## Examples

### Basic Usage

```tcl
# Create test tensors
set a [torch::tensor_create {1 2 3 4} float32]
set b [torch::tensor_create {0 2 4 4} float32]

# Positional syntax
set result1 [torch::gt $a $b]
# Returns tensor with values: {1 0 0 0}

# Named parameter syntax
set result2 [torch::gt -input1 $a -input2 $b]
# Returns tensor with values: {1 0 0 0}

# camelCase alias
set result3 [torch::Gt $a $b]
# Returns tensor with values: {1 0 0 0}
```

### Using Different Parameter Names

```tcl
# Alternative named parameter syntax
set result4 [torch::gt -tensor1 $a -tensor2 $b]

# Mixed parameter names
set result5 [torch::gt -input1 $a -tensor2 $b]
```

### Broadcasting Support

```tcl
# Broadcasting works with different shaped tensors
set scalar [torch::tensor_create {2} float32]
set vector [torch::tensor_create {1 2 3 4} float32]

set result [torch::gt $vector $scalar]
# Returns tensor with values: {0 0 1 1}
```

## Mathematical Operation

For tensors A and B with elements a_i and b_i respectively:

```
result_i = 1 if a_i > b_i, 0 otherwise
```

## Backward Compatibility

The original positional syntax is fully supported and will continue to work:

```tcl
# This will always work
set result [torch::gt $tensor1 $tensor2]
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::gt $a $b

# New equivalent syntax
torch::gt -input1 $a -input2 $b
```

### From snake_case to camelCase

```tcl
# Original command
torch::gt $a $b

# camelCase alias
torch::Gt $a $b
```

## Error Handling

The command provides clear error messages for common mistakes:

```tcl
# Missing arguments
torch::gt
# Error: Usage: torch::gt tensor1 tensor2 | torch::gt -input1 tensor1 -input2 tensor2

# Invalid tensor handle
torch::gt invalid_tensor $b
# Error: Invalid tensor name for input1

# Unknown parameter
torch::gt -unknown $a -input2 $b
# Error: Unknown parameter: -unknown. Valid parameters are: -input1, -tensor1, -input2, -tensor2
```

## Performance Notes

- Both syntaxes have identical performance characteristics
- The named parameter syntax adds minimal parsing overhead
- Broadcasting follows PyTorch's broadcasting rules

## Related Commands

- `torch::lt` - Element-wise less-than comparison
- `torch::le` - Element-wise less-than-or-equal comparison
- `torch::ge` - Element-wise greater-than-or-equal comparison
- `torch::eq` - Element-wise equality comparison
- `torch::ne` - Element-wise inequality comparison

## See Also

- [Basic Tensor Operations](../tensor_operations.md)
- [Comparison Operations](../comparison_operations.md)
- [Broadcasting Rules](../broadcasting.md) 