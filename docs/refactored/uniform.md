# torch::uniform

Generate Uniform Random Distribution

---

## Overview

Returns a tensor filled with random numbers from a uniform distribution on the interval `[low, high)`.

---

## Syntax

### Positional (Backward Compatible)
```tcl
torch::uniform size low high ?dtype? ?device?
```

### Named Parameters (Modern)
```tcl
torch::uniform -size SIZE -low LOW -high HIGH ?-dtype DTYPE? ?-device DEVICE?
```

---

## Parameters
| Name   | Type    | Required | Description                                    |
|--------|---------|----------|------------------------------------------------|
| size   | List    | Yes      | Shape of the output tensor                     |
| low    | Float   | Yes      | Lower bound of the uniform distribution        |
| high   | Float   | Yes      | Upper bound of the uniform distribution        |
| dtype  | String  | No       | Data type (default: "float32")                 |
| device | String  | No       | Device to place tensor on (default: "cpu")     |

---

## Return Value
A tensor of the specified shape filled with random numbers from the uniform distribution on `[low, high)`.

---

## Examples

### Positional Syntax
```tcl
# Basic usage
set uniform [torch::uniform {2 3} 0.0 1.0]

# With custom dtype
set uniform [torch::uniform {2 2} 0.0 1.0 float64]

# With custom range
set uniform [torch::uniform {3 3} -5.0 5.0]
```

### Named Parameter Syntax
```tcl
# Basic usage
set uniform [torch::uniform -size {2 3} -low 0.0 -high 1.0]

# With custom dtype
set uniform [torch::uniform -size {2 2} -low 0.0 -high 1.0 -dtype float64]

# With custom device
set uniform [torch::uniform -size {2 2} -low 0.0 -high 1.0 -device cpu]

# With custom range
set uniform [torch::uniform -size {3 3} -low -10.0 -high 10.0]
```

---

## Migration Guide
- **Old code using positional syntax will continue to work.**
- **New code should use named parameters for clarity and maintainability.**

| Old (positional)                    | New (named)                                                |
|-------------------------------------|------------------------------------------------------------|
| torch::uniform {2 3} 0.0 1.0       | torch::uniform -size {2 3} -low 0.0 -high 1.0              |
| torch::uniform {2 2} 0.0 1.0 float64 | torch::uniform -size {2 2} -low 0.0 -high 1.0 -dtype float64 |

---

## Error Handling
- Missing or invalid parameters will produce a clear error message.
- Both syntaxes are validated for required arguments.
- `low` must be less than `high`.
- Invalid dtype or device specifications will result in error.

---

## See Also
- [torch::normal](normal.md)
- [torch::rand](rand.md)
- [torch::randn](randn.md) 