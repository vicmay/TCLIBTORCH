# torch::maxpool3d / torch::maxPool3d

Creates a 3D max pooling layer for neural networks.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::maxpool3d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?
torch::maxPool3d -kernelSize value ?-stride value? ?-padding value? ?-ceilMode value?
```

### Positional Parameters (Legacy)
```tcl
torch::maxpool3d kernel_size ?stride? ?padding? ?ceil_mode?
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-kernelSize` | integer | Required | Size of the pooling window |
| `-stride` | integer | kernelSize | Stride of the pooling window |
| `-padding` | integer | 0 | Zero-padding added to all three sides |
| `-ceilMode` | boolean | 0 | When true, use ceil instead of floor for output shape |

## Description

The `maxpool3d` command creates a 3D max pooling layer that operates on 5D tensors (batch_size, channels, depth, height, width). It applies max pooling over a 3D input signal composed of several input planes.

The layer takes a 5D tensor as input and applies max pooling across its spatial dimensions (depth, height, width). For each window, it outputs the maximum value in that window.

## Examples

### Basic Usage with Named Parameters
```tcl
# Create a max pooling layer with kernel size 3
set layer [torch::maxpool3d -kernelSize 3]

# Create a layer with custom stride and padding
set layer [torch::maxpool3d -kernelSize 3 -stride 2 -padding 1]

# Create a layer with ceil mode enabled
set layer [torch::maxpool3d -kernelSize 3 -stride 2 -padding 1 -ceilMode 1]
```

### Legacy Positional Syntax
```tcl
# Basic usage with kernel size only
set layer [torch::maxpool3d 3]

# With stride and padding
set layer [torch::maxpool3d 3 2 1]

# With all parameters
set layer [torch::maxpool3d 3 2 1 1]
```

### Forward Pass Example
```tcl
# Create input tensor (batch_size=1, channels=1, depth=2, height=2, width=2)
set input [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0} -dtype float32]
set input [torch::tensor_reshape $input {1 1 2 2 2}]

# Create max pooling layer
set layer [torch::maxpool3d -kernelSize 2]

# Forward pass
set output [torch::layer_forward $layer $input]
```

## Return Value

Returns a handle to the created max pooling layer. This handle can be used with `torch::layer_forward` to apply the pooling operation to input tensors.

## Error Conditions

The command will raise an error if:
- No kernel size is provided
- Kernel size is less than or equal to 0
- Invalid parameter names are used
- Parameter values are missing
- Invalid parameter values are provided (e.g., non-integer kernel size)

## See Also

* [torch::maxpool1d](maxpool1d.md)
* [torch::maxpool2d](maxpool2d.md)
* [torch::avgpool3d](avgpool3d.md)