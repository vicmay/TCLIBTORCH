# torch::melscale_fbanks / torch::melscaleFbanks

**Category:** Signal Processing  
**Aliases:** `torch::melscale_fbanks`, `torch::melscaleFbanks`

---

## Description

Creates a bank of triangular mel-scale filters. These filters are used to convert a power spectrogram into a mel-scale spectrogram, which better approximates human auditory perception.

---

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::melscale_fbanks n_freqs n_mels sample_rate ?f_min? ?f_max?
```
- `n_freqs`: Number of frequency bins (typically FFT size / 2 + 1)
- `n_mels`: Number of mel bands to generate
- `sample_rate`: Sample rate of the audio signal in Hz
- `f_min` (optional): Minimum frequency in Hz (default: 0.0)
- `f_max` (optional): Maximum frequency in Hz (default: sample_rate/2)

### Named Parameter Syntax (Modern)
```tcl
torch::melscale_fbanks -nFreqs int -nMels int -sampleRate double ?-fMin double? ?-fMax double?
```
- `-nFreqs`: Number of frequency bins
- `-nMels`: Number of mel bands to generate
- `-sampleRate`: Sample rate of the audio signal in Hz
- `-fMin` (optional): Minimum frequency in Hz (default: 0.0)
- `-fMax` (optional): Maximum frequency in Hz (default: sample_rate/2)

### CamelCase Alias
```tcl
torch::melscaleFbanks ...
```
Supports both syntaxes above.

---

## Parameters

| Name       | Type   | Required | Default        | Description                                 |
|------------|--------|----------|----------------|---------------------------------------------|
| nFreqs     | int    | Yes      |                | Number of frequency bins                    |
| nMels      | int    | Yes      |                | Number of mel bands to generate             |
| sampleRate | double | Yes      |                | Sample rate of the audio signal in Hz       |
| fMin       | double | No       | 0.0            | Minimum frequency in Hz                     |
| fMax       | double | No       | sampleRate/2   | Maximum frequency in Hz                     |

---

## Examples

### 1. Positional Syntax
```tcl
# Create mel filter banks with 64 frequency bins and 20 mel bands
set mel_filters [torch::melscale_fbanks 64 20 16000]

# Create mel filter banks with custom frequency range
set mel_filters [torch::melscale_fbanks 64 20 16000 50.0 7000.0]
```

### 2. Named Parameter Syntax
```tcl
# Basic usage with named parameters
set mel_filters [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000]

# With custom frequency range
set mel_filters [torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0 -fMax 7000.0]
```

### 3. CamelCase Alias
```tcl
# Using camelCase alias
set mel_filters [torch::melscaleFbanks -nFreqs 64 -nMels 20 -sampleRate 16000]
```

### 4. Audio Processing Pipeline Example
```tcl
# Create a spectrogram from audio signal
set audio [torch::tensor_create -data $audio_data -dtype float32]
set spec [torch::spectrogram $audio 512 256]

# Apply mel filter banks
set mel_filters [torch::melscaleFbanks -nFreqs 257 -nMels 80 -sampleRate 16000]
set mel_spec [torch::tensor_matmul $mel_filters $spec]

# Convert to log scale
set log_mel_spec [torch::tensor_log $mel_spec]
```

---

## Error Handling

- Missing required parameters: Returns error message indicating required parameters
- Invalid parameter values: Returns error if n_freqs, n_mels, or sample_rate are not positive
- Unknown parameters: Returns error with the name of the unknown parameter

---

## Migration Guide

| Old (Positional)                                  | New (Named)                                                                    |
|---------------------------------------------------|--------------------------------------------------------------------------------|
| `torch::melscale_fbanks 64 20 16000`              | `torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000`               |
| `torch::melscale_fbanks 64 20 16000 50.0`         | `torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0`    |
| `torch::melscale_fbanks 64 20 16000 50.0 7000.0`  | `torch::melscale_fbanks -nFreqs 64 -nMels 20 -sampleRate 16000 -fMin 50.0 -fMax 7000.0` |

Both syntaxes are fully supported for backward compatibility.

---

## Return Value

Returns a tensor of shape `[n_mels, n_freqs]` containing the mel filter bank. Each row represents a triangular filter centered on a particular mel frequency.
