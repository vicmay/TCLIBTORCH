# torch::round

Rounds each element of the input tensor to the nearest integer.

## Syntax

```tcl
torch::round tensor
torch::round -input tensor
```

The command also supports a camelCase alias: `torch::Round`

## Arguments

* `tensor` (positional) or `-input tensor` (named): Input tensor to round

## Return Value

Returns a new tensor with each element rounded to the nearest integer.

## Examples

```tcl
# Create a test tensor
set t [torch::tensor_create -data {1.6 2.1 3.7 4.2 5.9 6.3} -shape {2 3}]

# Using positional syntax
set rounded [torch::round $t]
# Result tensor contains: {2 2 4 4 6 6}

# Using named parameter syntax
set rounded [torch::round -input $t]
# Result tensor contains: {2 2 4 4 6 6}

# Using camelCase alias
set rounded [torch::Round -input $t]
# Result tensor contains: {2 2 4 4 6 6}
```

## Error Conditions

* Invalid tensor name
* Missing input tensor
* Unknown parameter

## See Also

* `torch::ceil` - Round up to nearest integer
* `torch::floor` - Round down to nearest integer
* `torch::trunc` - Truncate decimal part 