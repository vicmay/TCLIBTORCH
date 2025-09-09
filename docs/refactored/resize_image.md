# torch::resize_image / torch::resizeImage

## Description
Resizes an image tensor using interpolation. This operation is particularly useful for tasks like image preprocessing, data augmentation, and adapting images to specific input sizes for neural networks.

## Syntax

### Original Syntax (Positional Parameters)
```tcl
torch::resize_image image size ?mode? ?align_corners?
```

### New Syntax (Named Parameters)
```tcl
torch::resize_image -input tensor -size {height width} ?-mode mode? ?-alignCorners bool?
torch::resize_image -tensor tensor -size {height width} ?-mode mode? ?-alignCorners bool?
torch::resize_image -image tensor -size {height width} ?-mode mode? ?-alignCorners bool?
```

### CamelCase Alias
```tcl
torch::resizeImage tensor size ?mode? ?align_corners?
torch::resizeImage -input tensor -size {height width} ?-mode mode? ?-alignCorners bool?
```

## Parameters

### Positional Parameters
1. `image` (tensor): Input tensor of shape (batch_size, channels, height, width)
2. `size` (list): A list of two integers specifying the output size: {height width}
3. `mode` (string, optional): Interpolation mode. Valid values are:
   - "nearest" - Nearest neighbor interpolation
   - "bilinear" - Bilinear interpolation (default)
   - "bicubic" - Bicubic interpolation
4. `align_corners` (boolean, optional): If true, the corner pixels of the input and output tensors are aligned, and thus preserving the values at the corner pixels. Default: false

### Named Parameters
- `-input`, `-tensor`, or `-image` (tensor): Input tensor of shape (batch_size, channels, height, width)
- `-size` (list): A list of two integers specifying the output size: {height width}
- `-mode` (string, optional): Interpolation mode. Same values as above. Default: "bilinear"
- `-align_corners` or `-alignCorners` (boolean, optional): Same as above. Default: false

## Return Value
Returns a new tensor with the same shape as the input tensor except for the height and width dimensions, which are resized to the specified size.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a 4D tensor (1, 1, 3, 4)
set data [list 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0]
set tensor [torch::tensor_create -data $data -dtype float32]
set input [torch::tensor_reshape $tensor {1 1 3 4}]

# Resize to 6x8
set result [torch::resize_image $input {6 8}]
# Result shape: (1, 1, 6, 8)
```

### Named Parameter Syntax
```tcl
# Using named parameters
set result [torch::resizeImage -input $input -size {6 8}]

# Using parameter aliases
set result [torch::resizeImage -tensor $input -size {6 8}]
set result [torch::resizeImage -image $input -size {6 8}]
```

### Different Interpolation Modes
```tcl
# Nearest neighbor interpolation
set result [torch::resizeImage -input $input -size {6 8} -mode nearest]

# Bicubic interpolation with aligned corners
set result [torch::resizeImage -input $input -size {6 8} -mode bicubic -alignCorners 1]
```

## Error Handling

The command will throw an error in the following cases:
- Input tensor is not 4D (batch_size, channels, height, width)
- Size list does not contain exactly 2 values
- Invalid mode specified (must be "nearest", "bilinear", or "bicubic")
- Invalid tensor name
- Missing required parameters
- Unknown parameters

## See Also
- `torch::interpolate` - General interpolation function
- `torch::upsample_bilinear` - Bilinear upsampling
- `torch::normalize_image` - Image normalization
- `torch::denormalize_image` - Image denormalization 