# torch::tensor_fft2d

Computes the 2-dimensional Fast Fourier Transform (FFT) of a tensor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_fft2d tensor ?dims?
```

### Named Parameters (New)
```tcl
torch::tensor_fft2d -tensor tensor -dims {d0 d1}
```

### CamelCase Alias
```tcl
torch::tensorFft2d tensor ?dims?
torch::tensorFft2d -tensor tensor -dims {d0 d1}
```

## Parameters

| Parameter | Type   | Required | Default | Description                                 |
|-----------|--------|----------|---------|---------------------------------------------|
| `tensor`  | string | Yes      | -       | Name of the input tensor                    |
| `dims`    | list   | No       | last 2  | List of 2 dimensions to compute FFT over    |

## Description

The `torch::tensor_fft2d` command computes the 2D Fast Fourier Transform of a tensor. The FFT is performed over two dimensions, either specified or defaulting to the last two.

- **Input**: A 2D or higher tensor
- **Output**: A complex tensor with the same shape as the input
- **Default behavior**: If no dims are specified, FFT is computed over the last two dimensions

## Examples

### Basic Usage
```tcl
# Create a 2D tensor
set t [torch::tensor_create {{1.0 2.0} {3.0 4.0}} float32 cpu true]

# Compute 2D FFT using positional syntax
set fft_result [torch::tensor_fft2d $t]

# Compute 2D FFT using named parameters
set fft_result [torch::tensor_fft2d -tensor $t]

# Compute 2D FFT using camelCase alias
set fft_result [torch::tensorFft2d $t]
```

### Specifying Dimensions
```tcl
# FFT over specific dims (e.g., 0 and 1)
set fft_result [torch::tensor_fft2d $t {0 1}]
set fft_result [torch::tensor_fft2d -tensor $t -dims {0 1}]
set fft_result [torch::tensorFft2d -tensor $t -dims {0 1}]
```

### Error Handling
- **Invalid tensor name**: When the specified tensor doesn't exist
- **dims not a list of 2**: When dims is not a list of exactly 2 integers
- **Unknown parameter**: When using named parameters with invalid names
- **Missing tensor parameter**: When no tensor is provided

## Migration Guide

### From Old Syntax to New Syntax
**Old (Positional Only):**
```tcl
torch::tensor_fft2d tensor_name {0 1}
```
**New (Named Parameters):**
```tcl
torch::tensor_fft2d -tensor tensor_name -dims {0 1}
```
**New (CamelCase):**
```tcl
torch::tensorFft2d tensor_name {0 1}
torch::tensorFft2d -tensor tensor_name -dims {0 1}
```

### Backward Compatibility
The old positional syntax is fully supported and will continue to work:
```tcl
torch::tensor_fft2d my_tensor
```

## Notes
- The FFT result is a complex tensor, even if the input is real
- The output has the same shape as the input
- For real input signals, the FFT result has conjugate symmetry
- The FFT is most efficient when the size of each dimension is a power of 2

## Related Commands
- `torch::tensor_fft`    - 1D Fast Fourier Transform
- `torch::tensor_ifft2d` - 2D Inverse Fast Fourier Transform
- `torch::tensor_rfft`   - Real-valued Fast Fourier Transform
- `torch::tensor_irfft`  - Inverse Real-valued Fast Fourier Transform 