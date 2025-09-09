# torch::fold

## Overview
Combines sliding blocks from an input tensor into a single output tensor. This is the reverse operation of `torch::unfold` and is commonly used in operations like col2im or for combining patches back into images.

## Syntax

### Positional Syntax (Backward Compatibility)
```tcl
torch::fold input output_size kernel_size ?dilation? ?padding? ?stride?
```

### Named Parameter Syntax (Recommended)
```tcl
torch::fold -input tensor -outputSize {h w} -kernelSize {h w} [-dilation {h w}] [-padding {h w}] [-stride {h w}]
```

### camelCase Alias
```tcl
torch::Fold -input tensor -outputSize {h w} -kernelSize {h w} [-dilation {h w}] [-padding {h w}] [-stride {h w}]
```

## Parameters

### Required Parameters
- **input** (string): Input tensor handle containing the sliding blocks
- **output_size** / **outputSize** (list): Output spatial dimensions as `{height width}`
- **kernel_size** / **kernelSize** (list): Size of the sliding blocks as `{height width}`

### Optional Parameters
- **dilation** (list): Dilation factor for the sliding blocks. Default: `{1 1}`
- **padding** (list): Padding applied to the input. Default: `{0 0}`
- **stride** (list): Stride for the sliding blocks. Default: `{1 1}`

## Parameter Aliases
- `-input` and `-tensor` are equivalent
- `-output_size` and `-outputSize` are equivalent
- `-kernel_size` and `-kernelSize` are equivalent

## Return Value
Returns a tensor handle containing the combined output tensor.

## Examples

### Basic Usage
```tcl
# Create input tensor with sliding blocks
set input [torch::unfold $image {3 3} {3 3}]

# Basic fold operation
set result [torch::fold $input {28 28} {3 3}]
```

### Named Parameter Syntax
```tcl
# Using named parameters
set result [torch::fold -input $input -outputSize {28 28} -kernelSize {3 3}]

# Using camelCase alias
set result [torch::Fold -input $input -outputSize {28 28} -kernelSize {3 3}]
```

### With Optional Parameters
```tcl
# With dilation, padding, and stride
set result [torch::fold -input $input \
    -outputSize {28 28} \
    -kernelSize {3 3} \
    -dilation {2 2} \
    -padding {1 1} \
    -stride {2 2}]
```

### Positional Syntax
```tcl
# Basic positional syntax
set result [torch::fold $input {28 28} {3 3}]

# With all parameters
set result [torch::fold $input {28 28} {3 3} {1 1} {0 0} {1 1}]
```

## Mathematical Description

The fold operation combines sliding blocks extracted from an input tensor back into a single output tensor. Given:
- Input tensor with shape `(N, C × kernel_h × kernel_w, L)`
- Output size `(output_h, output_w)`
- Kernel size `(kernel_h, kernel_w)`

The operation reconstructs the original tensor by placing the sliding blocks at their corresponding positions in the output tensor.

## Input/Output Shapes

### Input Shape
- Input tensor: `(N, C × kernel_h × kernel_w, L)`
  - `N`: batch size
  - `C`: number of channels
  - `kernel_h × kernel_w`: flattened kernel dimensions
  - `L`: number of sliding blocks

### Output Shape
- Output tensor: `(N, C, output_h, output_w)`
  - `N`: batch size
  - `C`: number of channels
  - `output_h, output_w`: specified output dimensions

## Common Use Cases

### 1. Image Reconstruction from Patches
```tcl
# Extract patches from image
set patches [torch::unfold $image {8 8} {8 8}]

# Reconstruct image from patches
set reconstructed [torch::fold $patches {224 224} {8 8}]
```

### 2. Convolution Implementation
```tcl
# Manual convolution using unfold/fold
set patches [torch::unfold $input {3 3} {3 3}]
set conv_result [torch::matmul $weights $patches]
set output [torch::fold $conv_result {28 28} {1 1}]
```

### 3. Patch-based Processing
```tcl
# Process image patches and recombine
set patches [torch::unfold $image {16 16} {16 16}]
set processed [process_patches $patches]
set result [torch::fold $processed {256 256} {16 16}]
```

## Error Handling

The function validates all parameters and provides clear error messages:

```tcl
# Invalid tensor handle
catch {torch::fold "invalid" {28 28} {3 3}} error
# Error: Invalid input tensor name

# Invalid output size format
catch {torch::fold $input {28} {3 3}} error
# Error: Output size must be list of 2 ints

# Invalid kernel size format
catch {torch::fold $input {28 28} {3}} error
# Error: Kernel size must be list of 2 ints

# Missing required parameters
catch {torch::fold -input $input} error
# Error: Required parameters missing: input tensor, output_size, and kernel_size required
```

## Performance Considerations

- The fold operation is memory-efficient as it reconstructs the output tensor directly
- For large tensors, consider the memory requirements of the output tensor
- GPU acceleration is available when input tensors are on CUDA devices

## Relationship to Other Operations

- **torch::unfold**: Inverse operation that extracts sliding blocks
- **torch::conv2d**: Uses similar concepts for convolution operations
- **torch::col2im**: PyTorch's internal operation that fold is based on

## Data Type Support

Supports all standard PyTorch data types:
- Float: `float32`, `float64`
- Integer: `int32`, `int64`
- Complex: `complex64`, `complex128`
- Half precision: `float16`
- Boolean: `bool`

## Device Compatibility

- **CPU**: Full support
- **CUDA**: Full support with GPU acceleration
- **Mixed**: Automatic device handling based on input tensor device

## See Also

- [`torch::unfold`](unfold.md) - Extract sliding blocks from tensors
- [`torch::conv2d`](conv2d.md) - 2D convolution operation
- [`torch::col2im`](col2im.md) - Column-to-image transformation
- [`torch::im2col`](im2col.md) - Image-to-column transformation

---

*This documentation covers both the legacy positional syntax and the new named parameter syntax. The named parameter syntax is recommended for new code due to improved readability and maintainability.* 