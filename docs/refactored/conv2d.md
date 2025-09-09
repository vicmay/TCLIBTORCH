# torch::conv2d / torch::conv2dLayer

Creates a 2D convolutional neural network layer.

## Syntax

```tcl
# Positional syntax (backward compatibility)
torch::conv2d in_channels out_channels kernel_size ?stride? ?padding? ?bias?

# Named parameter syntax
torch::conv2d -inChannels value -outChannels value -kernelSize value ?-stride value? ?-padding value? ?-bias value?

# camelCase alias
torch::conv2dLayer -inChannels value -outChannels value -kernelSize value ?-stride value? ?-padding value? ?-bias value?
```

## Parameters

| Parameter   | Type    | Default | Description                                                |
|-------------|---------|--------|------------------------------------------------------------|
| inChannels  | integer | -       | Number of channels in the input image                      |
| outChannels | integer | -       | Number of channels produced by the convolution             |
| kernelSize  | integer | -       | Size of the convolving kernel                              |
| stride      | integer | 1       | Stride of the convolution                                  |
| padding     | integer | 0       | Zero-padding added to both sides of the input              |
| bias        | boolean | 1       | If true, adds a learnable bias to the output               |

## Return Value

Returns a handle to the created conv2d layer module.

## Description

The `torch::conv2d` command creates a 2D convolutional layer that applies a 2D convolution over an input signal composed of several input planes.

The command supports both positional and named parameter syntax for backward compatibility.

## Examples

### Positional Syntax

```tcl
# Create a basic conv2d layer with 3 input channels, 16 output channels, and 3x3 kernel
set conv [torch::conv2d 3 16 3]

# Create a conv2d layer with stride and padding
set conv_with_stride [torch::conv2d 3 16 3 2 1]

# Create a conv2d layer without bias
set conv_no_bias [torch::conv2d 3 16 3 1 0 0]

# Use the conv2d layer in forward pass
set input_tensor [torch::ones [list 2 3 32 32]]  # batch_size=2, channels=3, height=32, width=32
set output_tensor [torch::layer_forward $conv $input_tensor]
```

### Named Parameter Syntax

```tcl
# Create a conv2d layer with named parameters
set conv [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3]

# Create a conv2d layer with all parameters
set conv_full [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -stride 2 -padding 1 -bias 1]

# Create a conv2d layer without bias
set conv_no_bias [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -bias 0]

# Parameters can be in any order
set conv [torch::conv2d -bias 1 -outChannels 16 -stride 1 -inChannels 3 -kernelSize 3 -padding 1]
```

### camelCase Alias

```tcl
# Use the camelCase alias
set conv [torch::conv2dLayer -inChannels 3 -outChannels 16 -kernelSize 3]
```

## Output Shape Calculation

For a square input with size \(H_{in} \times W_{in}\), the output size \(H_{out} \times W_{out}\) is calculated as:

```
H_out = floor((H_in + 2 * padding - kernel_size) / stride + 1)
W_out = floor((W_in + 2 * padding - kernel_size) / stride + 1)
```

## Error Handling

The command will throw an error if:
- Required parameters are missing
- Parameters have invalid values (e.g., negative or zero values for inChannels, outChannels, or kernelSize)
- Unknown parameters are provided
- A parameter value is missing

## Migration from Positional to Named Syntax

To migrate from the positional to the named parameter syntax:

```tcl
# Old syntax
set conv [torch::conv2d 3 16 3 2 1]

# New syntax
set conv [torch::conv2d -inChannels 3 -outChannels 16 -kernelSize 3 -stride 2 -padding 1]
```

## See Also

- `torch::layer_forward` - Forward pass through a layer
- `torch::conv1d` - 1D convolution layer
- `torch::conv3d` - 3D convolution layer
- `torch::maxpool2d` - 2D max pooling layer 