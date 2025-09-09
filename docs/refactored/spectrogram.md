# torch::spectrogram

Computes the power spectrogram of an input signal using the Short-Time Fourier Transform (STFT).

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::spectrogram tensor ?n_fft? ?hop_length? ?win_length? ?window?
```

### Modern Syntax (Named Parameters)
```tcl
torch::spectrogram -input tensor ?-nFft value? ?-hopLength value? ?-winLength value? ?-window tensor?
torch::spectrogram -input tensor ?-n_fft value? ?-hop_length value? ?-win_length value? ?-window tensor?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| input/tensor | tensor | Input signal tensor | Required |
| n_fft/nFft | integer | Size of FFT | 400 |
| hop_length/hopLength | integer | Number of samples between successive frames | 200 |
| win_length/winLength | integer | Window size | 400 |
| window | tensor | Window function tensor | Hann window |

## Description

The `spectrogram` command computes the power spectrogram of an input signal using the Short-Time Fourier Transform (STFT). The spectrogram represents the signal's frequency content over time.

The process involves:
1. Applying a sliding window to the signal
2. Computing the FFT for each window
3. Taking the magnitude squared of the complex STFT result

The resulting spectrogram has dimensions:
- Frequency bins: n_fft//2 + 1 (for real input)
- Time frames: Approximately (signal_length - n_fft) / hop_length + 1

## Examples

### Basic Usage
```tcl
# Create a test signal (1 second of 440 Hz sine wave at 16kHz)
set t [torch::tensor_arange 0 16000 1]
set freq 440.0
set sample_rate 16000.0
set t [torch::tensor_mul $t [torch::tensor_create [expr {2.0 * 3.14159 * $freq / $sample_rate}]]]
set signal [torch::tensor_sin $t]

# Compute spectrogram with default parameters (positional syntax)
set spec [torch::spectrogram $signal]

# Compute spectrogram with named parameters
set spec [torch::spectrogram -input $signal -nFft 512 -hopLength 256]
```

### Custom Window
```tcl
# Create a custom window
set window [torch::tensor_hamming_window 512]

# Using positional syntax
set spec [torch::spectrogram $signal 512 256 512 $window]

# Using named parameters
set spec [torch::spectrogram -input $signal -nFft 512 -hopLength 256 -winLength 512 -window $window]
```

### High Resolution Analysis
```tcl
# Higher frequency resolution (larger n_fft)
set spec [torch::spectrogram -input $signal -nFft 2048 -hopLength 512]

# Higher time resolution (smaller hop_length)
set spec [torch::spectrogram -input $signal -nFft 512 -hopLength 128]
```

## Error Handling

The command will raise an error if:
- The input tensor is invalid or not provided
- n_fft is not positive
- hop_length is not positive
- win_length is not positive
- The window tensor is invalid (if provided)

## Performance Notes

- Larger n_fft values give better frequency resolution but worse time resolution
- Smaller hop_length values give better time resolution but increase computation time
- The window function affects the frequency leakage and resolution trade-off
- For real-time applications, consider using smaller n_fft and hop_length values

## See Also

- `torch::stft` - Short-time Fourier transform
- `torch::melscale_fbanks` - Mel-scale filter banks
- `torch::mfcc` - Mel-frequency cepstral coefficients 