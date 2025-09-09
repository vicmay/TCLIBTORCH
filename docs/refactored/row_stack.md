# torch::row_stack

Stack tensors row-wise (alias for vstack). This command combines multiple tensors by stacking them vertically (along the first dimension).

## Syntax

```tcl
torch::row_stack tensor_list
torch::row_stack tensor1 tensor2 ?tensor3 ...?
torch::row_stack -tensors tensor_list
torch::row_stack -inputs tensor_list
```

The command also supports a camelCase alias: `torch::rowStack`

## Arguments

* `tensor_list` (positional) or `-tensors/-inputs tensor_list` (named): List of tensors to stack
* Multiple tensors can be provided as separate arguments in positional syntax

## Return Value

Returns a new tensor with the input tensors stacked along the first dimension (row-wise).

## Examples

```tcl
# Create test tensors
set tensor1 [torch::tensor_create -data {1 2 3} -shape {1 3}]
set tensor2 [torch::tensor_create -data {4 5 6} -shape {1 3}]

# Using positional syntax with list
set result1 [torch::row_stack [list $tensor1 $tensor2]]
# Result tensor shape: {2 3}, values: {1 2 3 4 5 6}

# Using positional syntax with multiple arguments
set result2 [torch::row_stack $tensor1 $tensor2]
# Result tensor shape: {2 3}, values: {1 2 3 4 5 6}

# Using named parameter syntax
set result3 [torch::row_stack -tensors [list $tensor1 $tensor2]]
# Result tensor shape: {2 3}, values: {1 2 3 4 5 6}

# Using camelCase alias
set result4 [torch::rowStack -inputs [list $tensor1 $tensor2]]
# Result tensor shape: {2 3}, values: {1 2 3 4 5 6}
```

## Error Conditions

* If no arguments are provided
* If an invalid tensor name is provided
* If tensors cannot be stacked (incompatible shapes)
* If an unknown parameter is provided in named syntax

## See Also

* `torch::vstack` - Equivalent command (row_stack is an alias for vstack)
* `torch::column_stack` - Stack tensors column-wise
* `torch::hstack` - Stack tensors horizontally
* `torch::dstack` - Stack tensors depth-wise 