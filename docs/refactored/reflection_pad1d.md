# torch::reflection_pad1d

Pads a 1D tensor using reflection of the border elements.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::reflection_pad1d tensor padding

# Named parameter syntax
torch::reflection_pad1d -input tensor -padding {left right}

# camelCase alias
torch::reflectionPad1d -input tensor -padding {left right}
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor / -input | tensor | Input tensor to be padded (1D or batched 1D) |
| padding / -padding | list | List of two integers {left right} specifying padding sizes |

## Return Value

Returns a new tensor with the specified reflection padding applied.

## Description

The `reflection_pad1d` command pads a 1D tensor by reflecting the border elements. For example, if you have a tensor `[1 2 3]` and pad it with `{2 1}`, the result will be `[2 1 1 2 3 3]`. The padding is done as follows:

1. Left padding: Reflects the leftmost elements
2. Right padding: Reflects the rightmost elements

The operation supports both single tensors and batched tensors. For batched tensors, the padding is applied to the last dimension.

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create a 1D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Add reflection padding (2 left, 1 right)
set padded [torch::reflection_pad1d $input {2 1}]
# Result: tensor([2.0, 1.0, 1.0, 2.0, 3.0, 3.0])
```

### Using Named Parameters
```tcl
# Create a 1D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Add reflection padding using named parameters
set padded [torch::reflection_pad1d -input $input -padding {2 1}]
# Result: tensor([2.0, 1.0, 1.0, 2.0, 3.0, 3.0])
```

### Using camelCase Alias
```tcl
# Create a 1D tensor
set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]

# Add reflection padding using camelCase alias
set padded [torch::reflectionPad1d -input $input -padding {2 1}]
# Result: tensor([2.0, 1.0, 1.0, 2.0, 3.0, 3.0])
```

### Edge Cases

#### Zero Padding
```tcl
set input [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
set padded [torch::reflection_pad1d $input {0 0}]
# Result: tensor([1.0, 2.0, 3.0]) - unchanged
```

#### Single Element Tensor
```tcl
set input [torch::tensor_create -data {1.0} -shape {1} -dtype float32]
set padded [torch::reflection_pad1d $input {1 1}]
# Result: tensor([1.0, 1.0, 1.0])
```

#### Negative Values
```tcl
set input [torch::tensor_create -data {-1.0 -2.0 -3.0} -shape {3} -dtype float32]
set padded [torch::reflection_pad1d $input {1 1}]
# Result: tensor([-2.0, -1.0, -2.0, -3.0, -2.0])
```

#### Large Padding
```tcl
set input [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
set padded [torch::reflection_pad1d $input {3 3}]
# Result: tensor([2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0])
```

## Error Handling

The command will raise an error in the following cases:
- If no input tensor is provided
- If no padding values are provided
- If the padding list does not contain exactly 2 values
- If the padding values are not valid integers
- If the input tensor handle is invalid

## See Also

- [torch::reflection_pad2d](reflection_pad2d.md) - 2D reflection padding
- [torch::reflection_pad3d](reflection_pad3d.md) - 3D reflection padding
- [torch::replication_pad1d](replication_pad1d.md) - 1D replication padding 