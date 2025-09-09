# torch::mfcc

Computes Mel-frequency cepstral coefficients (MFCCs) from a mel spectrogram.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::mfcc spectrogram ?n_mfcc? ?dct_type?
```

### Named Parameters (New Syntax)
```tcl
torch::mfcc -spectrogram tensor ?-nMfcc value? ?-dctType value?
torch::mfcc -spectrogram tensor ?-n_mfcc value? ?-dct_type value?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `spectrogram` / `-spectrogram` | string | Name of the mel spectrogram tensor | Required |
| `n_mfcc` / `-nMfcc` / `-n_mfcc` | integer | Number of MFCC coefficients to return | 13 |
| `dct_type` / `-dctType` / `-dct_type` | integer | Type of DCT to use (currently only type 2 is supported) | 2 |

## Returns

Returns a string handle to the tensor containing the MFCC coefficients. The output tensor has the same batch dimensions as the input, with the feature dimension changed to `n_mfcc`.

## Description

The `torch::mfcc` command computes Mel-frequency cepstral coefficients from a mel spectrogram. The process involves:

1. Taking the logarithm of the mel spectrogram
2. Applying a Discrete Cosine Transform (DCT) to decorrelate the features
3. Returning the first `n_mfcc` coefficients

MFCCs are commonly used features in audio processing, particularly for speech recognition and music analysis.

## Examples

### Basic Usage

```tcl
# Create a mel spectrogram (typically from an audio signal)
set mel_spec [torch::melspectrogram $waveform $sample_rate]

# Compute MFCCs with default parameters (positional syntax)
set mfccs [torch::mfcc $mel_spec]

# Compute MFCCs with named parameters
set mfccs [torch::mfcc -spectrogram $mel_spec -nMfcc 20]
```

### Custom Parameters

```tcl
# Create a mel spectrogram
set mel_spec [torch::melspectrogram $waveform $sample_rate]

# Compute 8 MFCCs (positional syntax)
set mfccs [torch::mfcc $mel_spec 8]

# Compute 8 MFCCs with named parameters
set mfccs [torch::mfcc -spectrogram $mel_spec -nMfcc 8]

# Compute 8 MFCCs with snake_case parameters
set mfccs [torch::mfcc -spectrogram $mel_spec -n_mfcc 8]
```

## Error Handling

The command will raise an error in the following cases:
- The spectrogram tensor name is invalid
- The n_mfcc value is not a valid integer
- The dct_type value is not a valid integer
- Invalid parameter name is used

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old syntax (still supported)
set mfccs [torch::mfcc $mel_spec 20]

# New syntax
set mfccs [torch::mfcc -spectrogram $mel_spec -nMfcc 20]
```

## See Also

- `torch::melspectrogram` - Compute a mel spectrogram from an audio waveform
- `torch::stft` - Short-time Fourier transform
- `torch::spectrogram` - Compute a spectrogram from an audio waveform
