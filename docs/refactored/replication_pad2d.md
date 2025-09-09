# torch::replication_pad2d / torch::replicationPad2d

## Description
Pads a 4D tensor using replication of the border elements. This operation extends the tensor by repeating its edge values, which is particularly useful for tasks like image processing, data augmentation, and 2D convolutional neural networks where edge artifacts need to be minimized.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::replication_pad2d tensor padding
```

### New Syntax (Named Parameters)
```tcl
torch::replication_pad2d -input tensor -padding {left right top bottom}
torch::replication_pad2d -tensor tensor -pad {left right top bottom}
```

### CamelCase Alias
```tcl
torch::replicationPad2d tensor padding
torch::replicationPad2d -input tensor -padding {left right top bottom}
```

## Parameters

### Positional Parameters
1. `tensor` (tensor): Input tensor of shape (batch_size, channels, height, width)
2. `padding` (list): A list of four integers specifying the padding size: {left right top bottom}

### Named Parameters
- `-input` or `-tensor` (tensor): Input tensor of shape (batch_size, channels, height, width)
- `-padding` or `-pad` (list): A list of four integers specifying the padding size: {left right top bottom}

## Return Value
Returns a new tensor with the same shape as the input tensor except for the height and width dimensions, which are increased by the sum of top/bottom and left/right padding respectively.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a 4D tensor (1, 1, 3, 4)
set data [list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
set tensor [torch::tensor_create -data $data -dtype float32]
set input [torch::tensor_reshape $tensor {1 1 3 4}]

# Add padding of 1 on all sides
set result [torch::replication_pad2d $input {1 1 1 1}]
# Result shape: (1, 1, 5, 6)
```

### Named Parameter Syntax
```tcl
# Using named parameters
set result [torch::replicationPad2d -input $input -padding {1 1 1 1}]

# Using parameter aliases
set result [torch::replicationPad2d -tensor $input -pad {1 1 1 1}]
```

### Uneven Padding
```tcl
# Add more padding on the left and top
set result [torch::replicationPad2d -input $input -padding {2 1 2 1}]
# Result shape: (1, 1, 6, 7)
```

## Error Handling

The command will throw an error in the following cases:
- Input tensor is not 4D (batch_size, channels, height, width)
- Padding list does not contain exactly 4 values
- Padding values are negative
- Invalid tensor name
- Missing required parameters
- Unknown parameters

## See Also
- `torch::replication_pad1d` - 1D replication padding
- `torch::replication_pad3d` - 3D replication padding
- `torch::reflection_pad2d` - 2D reflection padding
- `torch::constant_pad2d` - 2D constant padding 