# `torch::gamma` / `torch::gamma`  
Sample random numbers from a Gamma distribution.

## Summary
Generates a tensor of the given shape filled with random samples drawn from a Gamma distribution with provided `alpha` (shape) and `beta` (rate) parameters.

- **Backward-compatible positional syntax** remains fully supported.  
- **Modern named-parameter syntax** is now available for clearer, self-documenting calls.

## Usage

### Positional Syntax (legacy)
```tcl
# torch::gamma size alpha beta ?dtype? ?device?
set t [torch::gamma {3 4} 2.0 3.0 float32 cpu]
```

### Named-Parameter Syntax (recommended)
```tcl
# torch::gamma -size {shape} -alpha value -beta value ?-dtype type? ?-device dev?
set t [torch::gamma -size {3 4} -alpha 2.0 -beta 3.0 -dtype float32 -device cpu]
```

Both calls above return a handle to a newly-allocated tensor filled with Gamma-distributed values.

## Parameters
| Name | Positional Index | Type | Default | Description |
|------|------------------|------|---------|-------------|
| `size` | 1 | list(int) | — | Shape of the output tensor. |
| `alpha` | 2 | double | — | Shape parameter (>0). |
| `beta` | 3 | double | — | Rate parameter (>0). |
| `dtype` | 4 / `-dtype` | string | `float32` | Element data type. |
| `device` | 5 / `-device` | string | `cpu` | Device on which to allocate the tensor (`cpu`, `cuda`, ...). |

## Examples
```tcl
# 1-D tensor on CPU
set t1 [torch::gamma {5} 2.0 1.0]

# 2-D tensor on CUDA using named parameters
set t2 [torch::gamma -size {3 4} -alpha 0.5 -beta 2.0 -device cuda]

# Specify double precision
set t3 [torch::gamma -size {10} -alpha 3.0 -beta 0.8 -dtype float64]
```

## Error Handling
The command raises informative errors when:
- Required parameters are missing.
- `alpha` or `beta` are ≤ 0.
- Unknown named parameters are supplied.
- Invalid `dtype` or `device` strings are provided.

Example:
```tcl
# Missing beta parameter
catch { torch::gamma -size {3} -alpha 2.0 } msg
puts $msg  ;# -> "Required parameters missing or invalid: size, alpha > 0, beta > 0 must be provided"
```

## Migration Guide
| Legacy Call | Modern Equivalent |
|-------------|-------------------|
| `torch::gamma {2 2} 1.0 2.0` | `torch::gamma -size {2 2} -alpha 1.0 -beta 2.0` |

## Test Coverage
See `tests/refactored/gamma_test.tcl` for a comprehensive suite that validates:
1. Basic functionality for both syntaxes
2. Parameter validation and error handling
3. Data type / device flexibility
4. Syntax consistency

---
**Status:** Dual-syntax compatible / camelCase alias N/A (command name unchanged). 