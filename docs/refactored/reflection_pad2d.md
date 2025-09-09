# torch::reflection_pad2d / torch::reflectionPad2d

Pads a tensor using the reflection of the input boundary.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::reflection_pad2d tensor padding
```

### Named Parameter Syntax
```tcl
torch::reflectionPad2d -input tensor -padding {left right top bottom}
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor/input | tensor | The input tensor (3D or 4D) |
| padding | list | A list of 4 integers: {left right top bottom} |

## Alternative Parameter Names
- `-input` or `-tensor` for the input tensor
- `-padding` or `-pad` for the padding values

## Description

This command applies reflection padding to a tensor. The padding values specify how many pixels to add on each side. The input tensor must be 3D (CHW) or 4D (NCHW).

The reflection padding mode uses the input boundaries to fill the padding area. For example, padding the tensor `[1 2 3]` with 2 elements on both sides would result in `[3 2 1 2 3 2 1]`.

## Return Value

Returns a new tensor with the specified padding applied.

## Examples

### Using Positional Syntax
```tcl
# Create a 3x3 tensor
set t [torch::tensor_create {
    {1 2 3}
    {4 5 6}
    {7 8 9}
} float32]

# Add 1 pixel padding on all sides
set padded [torch::reflection_pad2d $t {1 1 1 1}]
# Result will be 5x5
```

### Using Named Parameter Syntax
```tcl
# Same operation with named parameters
set padded [torch::reflectionPad2d -input $t -padding {1 1 1 1}]

# Uneven padding (2 left, 1 right, 2 top, 1 bottom)
set uneven [torch::reflectionPad2d \
    -input $t \
    -padding {2 1 2 1}]
```

## Error Conditions

The command will return an error if:
- The input tensor is invalid or missing
- The padding list does not contain exactly 4 values
- The padding values are not valid integers
- The input tensor has incorrect dimensions (must be 3D or 4D)

## See Also

- [torch::reflection_pad1d](reflection_pad1d.md)
- [torch::reflection_pad3d](reflection_pad3d.md)
- [torch::replication_pad2d](replication_pad2d.md)
- [torch::constant_pad2d](constant_pad2d.md) 