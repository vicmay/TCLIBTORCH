# torch::time_stretch

Applies time stretching to a Short-Time Fourier Transform (STFT) matrix. This operation changes the duration of audio signals while preserving their pitch by manipulating the time-frequency representation.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::time_stretch stft_matrix rate
```

### Named Parameter Syntax (New)
```tcl
torch::time_stretch -input stft_matrix -rate value
```

### CamelCase Alias
```tcl
torch::timeStretch -input stft_matrix -rate value
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `stft_matrix` / `-input` | string | Yes | Name of the STFT matrix tensor |
| `rate` / `-rate` | double | Yes | Time stretching rate. Values > 1.0 speed up, < 1.0 slow down |

## Return Value

Returns a string handle to the resulting time-stretched STFT matrix.

## Description

The time stretching operation modifies the duration of audio signals by manipulating their time-frequency representation:

- **Rate > 1.0**: Speeds up the audio (shorter duration)
- **Rate < 1.0**: Slows down the audio (longer duration)
- **Rate = 1.0**: No change in duration

The implementation uses phase vocoder techniques:
1. Extracts magnitude and phase from the STFT matrix
2. Interpolates both magnitude and phase along the time dimension
3. Reconstructs the time-stretched STFT matrix

This is commonly used in audio processing applications such as:
- Audio speed adjustment
- Music tempo modification
- Speech rate control
- Audio effects and transformations

## Examples

### Basic Usage

```tcl
# Create STFT matrix (simplified example)
set stft [torch::tensor_create -data {{1.0 2.0 3.0 4.0} {5.0 6.0 7.0 8.0}}]

# Speed up by factor of 2 (shorter duration)
set fast [torch::time_stretch $stft 2.0]
puts [torch::tensor_shape $fast]
# Output: 2 2

# Slow down by factor of 0.5 (longer duration)
set slow [torch::time_stretch $stft 0.5]
puts [torch::tensor_shape $slow]
# Output: 2 8
```

### Using Named Parameters

```tcl
set stft [torch::tensor_create -data {{1.0 2.0 3.0 4.0} {5.0 6.0 7.0 8.0}}]

# Use named parameter syntax
set result [torch::time_stretch -input $stft -rate 1.5]
puts [torch::tensor_shape $result]
# Output: 2 2
```

### Using CamelCase Alias

```tcl
set stft [torch::tensor_create -data {{1.0 2.0 3.0 4.0} {5.0 6.0 7.0 8.0}}]

# Use camelCase alias
set result [torch::timeStretch -input $stft -rate 0.75]
puts [torch::tensor_shape $result]
# Output: 2 5
```

### Different Stretching Rates

```tcl
set stft [torch::tensor_create -data {{1.0 2.0 3.0 4.0} {5.0 6.0 7.0 8.0}}]

# No change
set unchanged [torch::time_stretch $stft 1.0]
puts [torch::tensor_shape $unchanged]
# Output: 2 4

# Moderate speed up
set faster [torch::time_stretch $stft 1.5]
puts [torch::tensor_shape $faster]
# Output: 2 2

# Extreme slow down
set slower [torch::time_stretch $stft 0.1]
puts [torch::tensor_shape $slower]
# Output: 2 40
```

### Working with Real STFT Data

```tcl
# Create a more realistic STFT matrix
set rows 64
set cols 128
set stft_data {}
for {set i 0} {$i < $rows} {incr i} {
    set row {}
    for {set j 0} {$j < $cols} {incr j} {
        lappend row [expr {sin($i * 0.1) * cos($j * 0.1)}]
    }
    lappend stft_data $row
}
set stft [torch::tensor_create -data $stft_data]

# Apply time stretching
set stretched [torch::time_stretch $stft 0.8]
puts "Original shape: [torch::tensor_shape $stft]"
puts "Stretched shape: [torch::tensor_shape $stretched]"
# Output: Original shape: 64 128
# Output: Stretched shape: 64 160
```

## Migration Guide

### From Positional to Named Syntax

**Old (Positional):**
```tcl
torch::time_stretch $stft_matrix 2.0
```

**New (Named Parameters):**
```tcl
torch::time_stretch -input $stft_matrix -rate 2.0
```

### Benefits of Named Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Documentation**: Self-documenting code

## Error Handling

The command will throw an error in the following cases:

- **Missing required parameters**: `Required parameters missing: input tensor required, rate must be positive`
- **Invalid tensor name**: `Invalid tensor`
- **Invalid rate value**: `Invalid rate value`
- **Non-positive rate**: `rate must be positive`
- **Unknown parameter**: `Unknown parameter: -invalid. Valid parameters are: -input, -rate`
- **Missing parameter value**: `Missing value for parameter`
- **Interpolation error**: `Input and output sizes should be greater than 0` (for very large rates)

### Error Examples

```tcl
# Missing arguments
torch::time_stretch
# Error: Required parameters missing: input tensor required, rate must be positive

# Invalid tensor
torch::time_stretch invalid_tensor 2.0
# Error: Invalid tensor

# Invalid rate
set stft [torch::tensor_create -data {{1.0 2.0} {3.0 4.0}}]
torch::time_stretch $stft invalid
# Error: Invalid rate value

# Zero rate
torch::time_stretch $stft 0.0
# Error: rate must be positive

# Negative rate
torch::time_stretch $stft -1.0
# Error: rate must be positive

# Unknown parameter
torch::time_stretch -input $stft -invalid 2.0
# Error: Unknown parameter: -invalid. Valid parameters are: -input, -rate

# Very large rate (causes interpolation error)
torch::time_stretch $stft 10.0
# Error: Input and output sizes should be greater than 0
```

## Notes

- **Backward Compatibility**: The positional syntax is fully supported for backward compatibility
- **STFT Input**: Expects a 2D tensor representing an STFT matrix (frequency bins × time frames)
- **Rate Range**: While any positive rate is accepted, very large rates (> 5.0) may cause interpolation errors
- **Phase Handling**: The current implementation uses simplified phase reconstruction
- **Audio Quality**: For high-quality audio processing, consider using more sophisticated phase vocoder implementations
- **Memory Usage**: Creates a new tensor, so ensure sufficient memory for large STFT matrices
- **Performance**: The operation is computationally intensive for large matrices

## Related Commands

- `torch::spectrogram` - Create spectrograms from audio signals
- `torch::pitch_shift` - Shift pitch of audio signals
- `torch::fft_shift` - Shift FFT output
- `torch::ifft_shift` - Inverse FFT shift
- `torch::hilbert` - Compute analytic signal 