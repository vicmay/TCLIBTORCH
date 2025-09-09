# torch::ifftshift

Performs an inverse FFT shift operation on a tensor. This operation is the inverse of `torch::fftshift`, undoing the centering of the zero-frequency component.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::ifftshift -input tensor ?-dim dimension?
torch::ifftshift -tensor tensor ?-dimension dimension?
```

### Positional Parameters (Legacy)
```tcl
torch::ifftshift tensor ?dim?
```

### CamelCase Alias
```tcl
torch::ifftShift -input tensor ?-dim dimension?
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input` or `-tensor` | tensor | Yes | Input tensor to shift |
| `-dim` or `-dimension` | integer | No | Dimension along which to shift (default: all dimensions) |

## Returns

Returns a new tensor with the inverse FFT shift applied.

## Description

The `torch::ifftshift` function performs an inverse FFT shift operation, which is commonly used in signal processing to undo the centering of the zero-frequency component that was done by `torch::fftshift`.

For even-length arrays, this operation shifts the zero-frequency component from the center back to the beginning. For odd-length arrays, it performs the corresponding inverse shift.

The operation is equivalent to:
- If `dim` is specified: shifts only along that dimension
- If `dim` is not specified: shifts along all dimensions

## Examples

### Basic Usage

```tcl
# Create a test tensor
set tensor [torch::tensor_create {1 2 3 4} float32]

# Apply inverse FFT shift (named parameters)
set result [torch::ifftshift -input $tensor]
puts [torch::tensor_data $result]
;# Output: {3.0 4.0 1.0 2.0}

# Using positional syntax (legacy)
set result2 [torch::ifftshift $tensor]
puts [torch::tensor_data $result2]
;# Output: {3.0 4.0 1.0 2.0}

# Using camelCase alias
set result3 [torch::ifftShift -input $tensor]
puts [torch::tensor_data $result3]
;# Output: {3.0 4.0 1.0 2.0}
```

### 2D Tensor Example

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1 2} {3 4}} float32]

# Apply inverse FFT shift to all dimensions
set result [torch::ifftshift -input $tensor]
puts [torch::tensor_data $result]
;# Output: {4.0 3.0 2.0 1.0}
```

### Dimension-Specific Shift

```tcl
# Create a 2D tensor
set tensor [torch::tensor_create {{1 2 3 4} {5 6 7 8}} float32]

# Apply inverse FFT shift only to dimension 1 (columns)
set result [torch::ifftshift -input $tensor -dim 1]
puts [torch::tensor_data $result]
;# Output: {3.0 4.0 1.0 2.0 7.0 8.0 5.0 6.0}

# Apply inverse FFT shift only to dimension 0 (rows)
set result2 [torch::ifftshift -input $tensor -dim 0]
puts [torch::tensor_data $result2]
;# Output: {5.0 6.0 7.0 8.0 1.0 2.0 3.0 4.0}
```

### Inverse Relationship with fftshift

```tcl
# Demonstrate inverse relationship
set original [torch::tensor_create {1 2 3 4 5 6 7 8} float32]

# Apply fftshift then ifftshift
set shifted [torch::fftshift $original]
set restored [torch::ifftshift $shifted]

puts "Original: [torch::tensor_data $original]"
puts "Restored: [torch::tensor_data $restored]"
;# Both should be identical
```

## Parameter Aliases

The function supports multiple parameter names for flexibility:

- `-input` and `-tensor`: Both specify the input tensor
- `-dim` and `-dimension`: Both specify the dimension to shift

## Error Handling

The function provides clear error messages for invalid inputs:

```tcl
# Missing input tensor
catch {torch::ifftshift} msg
puts $msg
;# Output: Required parameters missing: input tensor required

# Invalid dimension
set tensor [torch::tensor_create {1 2 3 4} float32]
catch {torch::ifftshift $tensor invalid_dim} msg
puts $msg
;# Output: Invalid dimension: must be integer

# Unknown parameter
catch {torch::ifftshift -input $tensor -unknown_param value} msg
puts $msg
;# Output: Unknown parameter: -unknown_param
```

## Technical Details

### Algorithm
1. For each dimension to be shifted:
   - Calculate shift amount as negative half the dimension size: `-(size / 2)`
   - Use `torch::roll` to perform the circular shift
2. If no dimension is specified, apply to all dimensions sequentially

### Performance
- The operation is implemented using PyTorch's roll function for efficiency
- Time complexity: O(n) where n is the number of elements in the tensor
- Space complexity: O(n) for the output tensor

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::ifftshift $tensor
torch::ifftshift $tensor $dim

# New named parameter syntax
torch::ifftshift -input $tensor
torch::ifftshift -input $tensor -dim $dim
```

### Advantages of Named Parameters
- **Clarity**: Parameter names make the code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Extensibility**: Easy to add new parameters without breaking existing code
- **Error Prevention**: Less likely to pass arguments in wrong order

## Related Functions

- `torch::fftshift`: Forward FFT shift (inverse of ifftshift)
- `torch::roll`: General tensor rolling operation
- `torch::fft::fft`: Fast Fourier Transform
- `torch::fft::ifft`: Inverse Fast Fourier Transform

## Mathematical Background

FFT shift operations are commonly used in signal processing and image processing:

- **Signal Processing**: Centers the zero-frequency component for visualization
- **Image Processing**: Moves the DC component to the center of the spectrum
- **Convolution**: Prepares signals for frequency-domain operations

The inverse FFT shift undoes this centering, which is essential when:
- Converting back from frequency domain to time domain
- Preparing data for inverse FFT operations
- Restoring original signal ordering

## See Also

- [torch::fftshift](fftshift.md) - Forward FFT shift
- [torch::roll](roll.md) - General tensor rolling
- [Signal Processing Guide](../guides/signal_processing.md) - Complete signal processing workflow 