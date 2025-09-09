# torch::replication_pad1d / torch::replicationPad1d

## Description
Pads a 3D tensor using replication of the border elements. This operation extends the tensor by repeating its edge values, which is particularly useful for tasks like signal processing, audio data augmentation, and 1D convolutional neural networks where edge artifacts need to be minimized.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::replication_pad1d tensor padding
```

### New Syntax (Named Parameters)
```tcl
torch::replication_pad1d -input tensor -padding {left right}
torch::replication_pad1d -tensor tensor -pad {left right}
```

### CamelCase Alias
```tcl
torch::replicationPad1d tensor padding
torch::replicationPad1d -input tensor -padding {left right}
```

## Parameters

### Positional Parameters
1. `tensor` (tensor): Input tensor of shape (batch_size, channels, width)
2. `padding` (list): A list of two integers specifying the padding size: {left right}

### Named Parameters
- `-input` or `-tensor` (tensor): Input tensor of shape (batch_size, channels, width)
- `-padding` or `-pad` (list): A list of two integers specifying the padding size: {left right}

## Return Value
Returns a new tensor with the same shape as the input tensor except for the width dimension, which is increased by the sum of left and right padding.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a 3D tensor (1, 1, 6)
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32]
set input [torch::tensor_reshape $tensor {1 1 6}]

# Add padding of 1 on both sides
set result [torch::replication_pad1d $input {1 1}]
# Result shape: (1, 1, 8)
```

### Named Parameter Syntax
```tcl
# Using named parameters
set result [torch::replicationPad1d -input $input -padding {1 1}]

# Using parameter aliases
set result [torch::replicationPad1d -tensor $input -pad {1 1}]
```

### Uneven Padding
```tcl
# Add more padding on the left
set result [torch::replicationPad1d -input $input -padding {2 1}]
# Result shape: (1, 1, 9)
```

## Error Handling

The command will throw an error in the following cases:
- Input tensor is not 3D (batch_size, channels, width)
- Padding list does not contain exactly 2 values
- Padding values are negative
- Invalid tensor name
- Missing required parameters
- Unknown parameters

## See Also
- `torch::replication_pad2d` - 2D replication padding
- `torch::replication_pad3d` - 3D replication padding
- `torch::reflection_pad1d` - 1D reflection padding
- `torch::constant_pad1d` - 1D constant padding 