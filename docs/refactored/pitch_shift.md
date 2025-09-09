# torch::pitch_shift / torch::pitchShift

Shifts the pitch of an audio waveform by a specified number of semitones.

## Syntax

### Legacy Syntax (Positional Parameters)
```tcl
torch::pitch_shift waveform sample_rate n_steps
torch::pitchShift waveform sample_rate n_steps  ;# camelCase alias
```

### Modern Syntax (Named Parameters)
```tcl
torch::pitch_shift -waveform tensor -sampleRate value -nSteps value
torch::pitchShift -waveform tensor -sampleRate value -nSteps value  ;# camelCase alias
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| waveform/tensor | tensor | The input audio waveform tensor |
| sample_rate/sampleRate | float | The sampling rate of the audio in Hz |
| n_steps/nSteps | float | Number of semitones to shift (positive = up, negative = down) |

## Description

The `pitch_shift` command modifies the pitch of an audio waveform by a specified number of semitones while preserving its duration. This is commonly used in audio processing for tasks like musical transposition or voice modification.

The pitch shift is performed using time stretching and resampling:
1. The waveform is time-stretched by a factor of 1/rate
2. The stretched waveform is resampled by the rate factor
3. The rate is calculated as 2^(n_steps/12), where n_steps is the number of semitones

A positive n_steps value shifts the pitch up, while a negative value shifts it down. For example:
- n_steps = 12.0: Shift up one octave
- n_steps = -12.0: Shift down one octave
- n_steps = 7.0: Shift up a perfect fifth
- n_steps = -5.0: Shift down a perfect fourth

## Examples

### Shifting Up One Octave
```tcl
# Using positional syntax
set shifted [torch::pitch_shift $waveform 44100.0 12.0]

# Using named syntax
set shifted [torch::pitch_shift -waveform $waveform -sampleRate 44100.0 -nSteps 12.0]

# Using camelCase alias
set shifted [torch::pitchShift -waveform $waveform -sampleRate 44100.0 -nSteps 12.0]
```

### Shifting Down One Octave
```tcl
# Using positional syntax
set shifted [torch::pitch_shift $waveform 44100.0 -12.0]

# Using named syntax
set shifted [torch::pitch_shift -waveform $waveform -sampleRate 44100.0 -nSteps -12.0]
```

### Shifting Up a Perfect Fifth
```tcl
# Shift up by 7 semitones (perfect fifth)
set shifted [torch::pitch_shift $waveform 44100.0 7.0]
```

## Error Handling

The command will raise an error if:
- The waveform tensor is missing or invalid
- The sample rate is missing or not positive
- The n_steps parameter is missing

## See Also

- `torch::time_stretch` - Time stretching without pitch modification
- `torch::spectrogram` - Compute spectrogram of audio signal
- `torch::stft` - Short-time Fourier transform 