# torch::silu

## Description
Applies the SiLU (Sigmoid Linear Unit) activation function, also known as Swish, element-wise.

The SiLU function is defined as:
```
silu(x) = x * sigmoid(x)
```

where sigmoid(x) = 1 / (1 + exp(-x))

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::silu tensor
```

### Named Parameter Syntax
```tcl
torch::silu -input tensor
```

### camelCase Alias
```tcl
torch::siLU tensor
torch::siLU -input tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor/-input | tensor | The input tensor to apply the SiLU activation function to. |

## Return Value
Returns a new tensor with the SiLU activation function applied element-wise.

## Examples

### Basic Usage with Positional Syntax
```tcl
# Create a tensor with values [1.0, 2.0, -1.0, -2.0]
set x [torch::tensor_create -data {1.0 2.0 -1.0 -2.0} -dtype float32]

# Apply SiLU activation
set result [torch::silu $x]

# Result will be approximately [0.731, 1.762, -0.269, -0.238]
```

### Using Named Parameter Syntax
```tcl
set x [torch::tensor_create -data {1.0 2.0 -1.0 -2.0} -dtype float32]
set result [torch::silu -input $x]
```

### Using camelCase Alias
```tcl
set x [torch::tensor_create -data {1.0 2.0 -1.0 -2.0} -dtype float32]
set result [torch::siLU -input $x]
```

## Error Handling

The command will raise an error in the following cases:
- If no input tensor is provided
- If the input tensor is invalid
- If an unknown parameter is provided

## See Also
- `torch::sigmoid` - The sigmoid activation function
- `torch::relu` - The ReLU activation function
- `torch::gelu` - The GELU activation function 