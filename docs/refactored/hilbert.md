# torch::hilbert

Compute the Hilbert transform of a signal.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::hilbert -input tensor
torch::hilbert -tensor tensor
```

### Positional Parameters (Legacy)
```tcl
torch::hilbert tensor
```

## Parameters

### Named Parameters
- **`-input`** or **`-tensor`** (tensor): Input signal tensor for which to compute the Hilbert transform

### Positional Parameters
1. **`tensor`** (tensor): Input signal tensor for which to compute the Hilbert transform

## Returns

A tensor containing the Hilbert transform of the input signal. The returned tensor has the same shape as the input tensor.

## Description

The `torch::hilbert` command computes the Hilbert transform of a signal using Fast Fourier Transform (FFT). The Hilbert transform is useful in signal processing for creating analytic signals and computing instantaneous amplitude and phase information.

The Hilbert transform of a signal creates a version of the signal that is 90 degrees out of phase with the original. When combined with the original signal, it forms an analytic signal that has no negative frequency components.

## Mathematical Background

The Hilbert transform H{x(t)} of a signal x(t) is defined as:

```
H{x(t)} = (1/π) ∫ x(τ)/(t-τ) dτ
```

In the frequency domain, the Hilbert transform is equivalent to:
1. Taking the FFT of the signal
2. Multiplying by -j*sgn(ω) where sgn is the sign function
3. Taking the inverse FFT

The implementation uses the following approach:
- Apply FFT to the input signal
- Create a filter that zeros negative frequencies and doubles positive frequencies
- Apply inverse FFT to get the Hilbert transform

## Examples

### Basic Usage with Named Parameters

```tcl
# Create a simple sinusoidal signal
set signal [torch::tensor_create {0.0 1.0 0.0 -1.0 0.0 1.0 0.0 -1.0} float32 cpu true]

# Compute Hilbert transform using named parameters
set hilbert_result [torch::hilbert -input $signal]

# Check result shape (should match input)
set shape [torch::tensor_shape $hilbert_result]
puts "Hilbert transform shape: $shape"  ;# Output: Hilbert transform shape: 8
```

### Using Alternative Parameter Name

```tcl
# Same functionality with alternative parameter name
set signal [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set hilbert_result [torch::hilbert -tensor $signal]
```

### Legacy Positional Syntax

```tcl
# Backward compatibility with positional parameters
set signal [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu true]
set hilbert_result [torch::hilbert $signal]
```

### Signal Processing Applications

```tcl
# Example: Computing analytic signal
set real_signal [torch::tensor_create {1.0 0.707 0.0 -0.707 -1.0 -0.707 0.0 0.707} float32 cpu true]
set hilbert_transform [torch::hilbert -input $real_signal]

# The analytic signal is: real_signal + j * hilbert_transform
# This allows computation of instantaneous amplitude and phase
```

### Different Signal Types

```tcl
# Short signal
set short_signal [torch::tensor_create {1.0 2.0} float32 cpu true]
set result [torch::hilbert -input $short_signal]

# Longer signal
set long_signal [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0} float32 cpu true]
set result [torch::hilbert -input $long_signal]
```

### Error Handling

```tcl
# Missing parameters
if {[catch {torch::hilbert} error]} {
    puts "Error: $error"
}

# Invalid parameter name
if {[catch {torch::hilbert -input $signal -unknown "value"} error]} {
    puts "Error: $error"
}

# Invalid tensor handle
if {[catch {torch::hilbert -input "invalid_handle"} error]} {
    puts "Error: $error"
}
```

## Parameter Validation

The command validates that:
- An input tensor is provided
- Named parameters come in pairs when using named syntax
- Parameter names are recognized
- Input tensor handle is valid

## Implementation Details

The Hilbert transform is computed using the following algorithm:

1. **FFT**: Apply Fast Fourier Transform to the input signal
2. **Filter Creation**: Create a filter H where:
   - H[0] = 1 (DC component unchanged)
   - H[k] = 2 for k = 1, 2, ..., N/2-1 (positive frequencies doubled)
   - H[N/2] = 1 for even N (Nyquist frequency unchanged)
   - H[k] = 0 for k = N/2+1, ..., N-1 (negative frequencies zeroed)
3. **Apply Filter**: Multiply FFT result by the filter
4. **IFFT**: Apply inverse FFT and take the real part

This approach effectively:
- Preserves the DC component
- Doubles positive frequency components
- Zeros negative frequency components
- Returns the real part of the analytic signal's imaginary component

## Performance Considerations

- **Computational Complexity**: O(N log N) due to FFT operations
- **Memory Usage**: O(N) for intermediate FFT results
- **Signal Length**: Works best with power-of-2 lengths for optimal FFT performance
- **Precision**: Uses complex FFT internally, returns real result

## Applications

The Hilbert transform is commonly used for:

1. **Analytic Signal Creation**: Combining real signal with its Hilbert transform
2. **Instantaneous Phase/Amplitude**: Computing envelope and phase information
3. **Single Sideband Modulation**: In communication systems
4. **Feature Extraction**: In signal analysis and machine learning
5. **Quadrature Detection**: In radar and communication systems

## Mathematical Properties

- **Linearity**: H{ax + by} = aH{x} + bH{y}
- **Involution**: H{H{x}} = -x (applying Hilbert transform twice gives negative of original)
- **Orthogonality**: The Hilbert transform is orthogonal to the original signal
- **Energy Preservation**: The energy of the Hilbert transform equals the energy of the original signal

## See Also

- [`torch::fft`](fft.md) - Fast Fourier Transform
- [`torch::ifft`](ifft.md) - Inverse Fast Fourier Transform
- [`torch::fftshift`](fftshift.md) - Shift zero-frequency component to center
- [`torch::ifftshift`](ifftshift.md) - Inverse FFT shift
- [`torch::spectrogram`](spectrogram.md) - Time-frequency analysis

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::hilbert $signal

# New named parameter syntax
torch::hilbert -input $signal
```

### Benefits of Named Parameters

1. **Clarity**: Parameter name makes the purpose explicit
2. **Consistency**: Matches other signal processing commands
3. **Extensibility**: Easier to add optional parameters in future versions
4. **Error Prevention**: Reduces mistakes in parameter usage

## Version History

- **v1.0**: Initial implementation with positional parameters
- **v2.0**: Added dual syntax support with named parameters
- **v2.0**: Enhanced error handling and validation
- **v2.0**: Improved documentation with mathematical background

## Notes

- The current implementation uses a simplified approach suitable for most signal processing applications
- For specialized applications requiring different Hilbert transform variants, consider using the underlying FFT operations directly
- The input signal should be real-valued for meaningful results
- The transform preserves the signal length and sampling characteristics 