# torch::tensor_ifft2d

Computes the 2-dimensional Inverse Fast Fourier Transform (IFFT) of a tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_ifft2d tensor ?dims?
```

### Named Parameters (New)
```tcl
torch::tensor_ifft2d -tensor tensor -dims {d0 d1}
```

### CamelCase Alias
```tcl
torch::tensorIfft2d tensor ?dims?
torch::tensorIfft2d -tensor tensor -dims {d0 d1}
```

## Parameters

| Parameter | Type   | Required | Default | Description                                 |
|-----------|--------|----------|---------|---------------------------------------------|
| `tensor`  | string | Yes      | -       | Name of the input tensor                    |
| `dims`    | list   | No       | last 2  | List of 2 dimensions to compute IFFT over   |

## Description

The `torch::tensor_ifft2d` command computes the 2D Inverse Fast Fourier Transform of a tensor. The IFFT is the mathematical inverse of the FFT, transforming a signal from the frequency domain back to the time domain over two dimensions.

- **Input**: A 2D or higher tensor (typically complex, but can be real)
- **Output**: A complex tensor with the same shape as the input
- **Default behavior**: If no dims are specified, IFFT is computed over the last two dimensions

## Examples

### Basic Usage
```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Compute 2D IFFT using positional syntax
set ifft_result [torch::tensor_ifft2d $t]

# Compute 2D IFFT using named parameters
set ifft_result [torch::tensor_ifft2d -tensor $t]

# Compute 2D IFFT using camelCase alias
set ifft_result [torch::tensorIfft2d $t]
```

### Specifying Dimensions
```tcl
# IFFT over specific dims (e.g., 0 and 1)
set ifft_result [torch::tensor_ifft2d $t {0 1}]
set ifft_result [torch::tensor_ifft2d -tensor $t -dims {0 1}]
set ifft_result [torch::tensorIfft2d -tensor $t -dims {0 1}]
```

### FFT2D-IFFT2D Round Trip
```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Apply FFT2D then IFFT2D - should recover original signal
set fft_result [torch::tensor_fft2d $t]
set ifft_result [torch::tensor_ifft2d $fft_result]

# Get the shape of the result
set shape [torch::tensor_shape $ifft_result]
puts "IFFT2D result shape: $shape"
```

## Error Handling

The command provides clear error messages for various error conditions:

- **Invalid tensor name**: When the specified tensor doesn't exist
- **dims not a list of 2**: When dims is not a list of exactly 2 integers
- **Unknown parameter**: When using named parameters with invalid names
- **Missing tensor parameter**: When no tensor is provided

## Migration Guide

### From Old Syntax to New Syntax
**Old (Positional Only):**
```tcl
torch::tensor_ifft2d tensor_name {0 1}
```
**New (Named Parameters):**
```tcl
torch::tensor_ifft2d -tensor tensor_name -dims {0 1}
```
**New (CamelCase):**
```tcl
torch::tensorIfft2d tensor_name {0 1}
torch::tensorIfft2d -tensor tensor_name -dims {0 1}
```

### Backward Compatibility
The old positional syntax is fully supported and will continue to work:
```tcl
torch::tensor_ifft2d my_tensor
```

## Mathematical Background

The 2D IFFT is the inverse operation of the 2D FFT. For a 2D signal of size M×N, the 2D IFFT computes:

```
x[m,n] = (1/(M*N)) * Σ Σ X[k,l] * e^(j*2π*(k*m/M + l*n/N))
```

where:
- `x[m,n]` is the (m,n)-th time sample
- `X[k,l]` is the (k,l)-th frequency component
- `M` and `N` are the dimensions of the signal

The 2D IFFT transforms a signal from the 2D frequency domain back to the 2D time domain.

## Related Commands

- `torch::tensor_fft2d` - 2D Fast Fourier Transform
- `torch::tensor_ifft` - 1D Inverse Fast Fourier Transform
- `torch::tensor_rfft` - Real-valued Fast Fourier Transform
- `torch::tensor_irfft` - Inverse Real-valued Fast Fourier Transform

## Notes

- The IFFT result is typically a complex tensor, even if the input is real
- The output has the same shape as the input
- For real input signals, the IFFT result may have conjugate symmetry
- The IFFT is most efficient when the size of each dimension is a power of 2
- IFFT2D is the mathematical inverse of FFT2D: `IFFT2D(FFT2D(x)) ≈ x`

## Applications

- 2D signal reconstruction from frequency domain
- Image processing and restoration
- 2D spectral analysis and filtering
- Digital image processing
- Scientific computing and analysis
- Computer vision applications 