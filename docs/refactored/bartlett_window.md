# torch::bartlett_window / torch::bartlettWindow

Generates a Bartlett window function tensor for signal processing applications.

## Aliases
- `torch::bartlett_window`  (legacy snake-case)
- `torch::bartlettWindow`  (camelCase modern)

---

## Positional Syntax (back-compat)
```tcl
set window [torch::bartlett_window <windowLength> ?dtype? ?device? ?periodic?]
```

## Named Parameter Syntax (recommended)
```tcl
set window [torch::bartlettWindow \
    -window_length    <int> \
    -dtype           <string>        ;# default: float32 \
    -device          <string>        ;# default: cpu \
    -periodic        <0|1> ]         ;# default: 1
```
Parameter aliases:
* `-window_length` / `-windowLength` / `-length`

### Parameters
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **-window_length** | int | — | Length of the window (must be positive) |
| **-dtype** | string | float32 | Data type: float32, float64, double, float, int32, int64 |
| **-device** | string | cpu | Device: cpu, cuda |
| **-periodic** | bool (0/1) | 1 | Whether window is periodic (1) or symmetric (0) |

### Returns
Tensor handle containing the Bartlett window values. Shape: `(window_length,)`

The Bartlett window (triangular window) ranges from 0 to 1, with peak at center.

---

## Examples
```tcl
# Create basic Bartlett window
set window [torch::bartlett_window 5]

# Positional (legacy)
set window1 [torch::bartlett_window 8 float64 cpu 0]

# Named (preferred)
set window2 [torch::bartlettWindow -windowLength 8 -dtype float64 -device cpu -periodic 0]

# Common signal processing usage
set window [torch::bartlettWindow -length 1024 -dtype float32]
set windowed_signal [torch::tensor_mul $signal $window]
```

---

## Mathematical Properties
The Bartlett window is defined as:
- **Symmetric (periodic=0)**: Triangular shape with zeros at endpoints
- **Periodic (periodic=1)**: Triangular shape suitable for DFT/FFT
- **Peak value**: Always 1.0 at the center
- **Edge values**: 0.0 for symmetric, small positive for periodic

## Use Cases
- **Signal windowing**: Reducing spectral leakage in FFT analysis
- **Filter design**: Creating smooth transitions in digital filters  
- **Spectral analysis**: Preparing signals for frequency domain processing
- **Audio processing**: Windowing audio frames for analysis

---

## Error Messages
| Situation | Message |
|-----------|---------|
| Missing required params | `Named parameters require pairs: -param value` |
| Invalid window length | `Required parameters: window_length must be positive` |
| Unknown parameter | `Unknown parameter: -foo` |
| Invalid dtype | `Unsupported dtype: invalid_type` |

---

## Migration Guide
Old:
```tcl
set window [torch::bartlett_window 512]
```
New:
```tcl
set window [torch::bartlettWindow -windowLength 512]
```

With additional options:
```tcl
# Old (limited options)
set window [torch::bartlett_window 512]

# New (full control)
set window [torch::bartlettWindow -windowLength 512 -dtype float64 -periodic 0]
```

---

## Tests
Validated by `tests/refactored/bartlett_window_test.tcl` (positional, named, alias, equivalence, errors, edge cases).

## Compatibility
✅ Positional syntax retained • ✅ Named parameters added • ✅ camelCase alias registered 

## Related Commands

- **[torch::blackman_window](blackman_window.md)**: Blackman window function
- **[torch::hamming_window](hamming_window.md)**: Hamming window function  
- **[torch::hann_window](hann_window.md)**: Hann window function
- **[torch::kaiser_window](kaiser_window.md)**: Kaiser window function
- **[torch::tensor_mul](tensor_mul.md)**: Element-wise multiplication for windowing 