# `torch::gelu` / `torch::gelu`

> Gaussian Error Linear Unit (GELU) activation function.
>
> The GELU non-linearity is defined as:
>
> \[ \text{gelu}(x) = 0.5 x \left(1 + \text{erf}\left( \frac{x}{\sqrt{2}} \right) \right) \]
>
> It is commonly used in transformer and modern neural network architectures.

---

## 🆕 Named-parameter Syntax (Recommended)

```tcl
# Usage: torch::gelu -input tensorHandle
set result [torch::gelu -input $tensor]
```

### Parameters
| Name  | Type   | Required | Description                  |
|-------|--------|----------|------------------------------|
| `-input` | tensor handle | ✅ Yes | Input tensor to apply GELU |

### Example
```tcl
set x [torch::ones {2 3}]
set y [torch::gelu -input $x]
```

---

## ♻️ Positional Syntax (Backward-compatible)

```tcl
# Usage: torch::gelu tensorHandle
set result [torch::gelu $tensor]
```

This form is preserved for full backward compatibility with existing scripts.

---

## 🔀 camelCase Alias

`torch::gelu` is already in lowercase and camelCase friendly, therefore the alias equals the original name.

---

## ✅ Notes

* Both syntaxes are fully equivalent and yield identical results.
* The output tensor shares the same shape, dtype, and device as the input unless otherwise modified later.
* Gradients are propagated automatically if `requires_grad` is set on the input tensor.

---

## ⚠️ Error Handling

| Error Condition | Message |
|-----------------|---------|
| Missing `-input` value (named syntax) | `Missing value for option` |
| Unknown parameter | `Unknown parameter: -foo` |
| Invalid tensor handle | `Invalid tensor name` |

---

## 🧪 Tests

Comprehensive tests covering both syntaxes, error handling, and edge cases reside in `tests/refactored/gelu_test.tcl`.

---

## 🔄 Migration Guide

Old scripts using positional syntax continue to work:

```tcl
# Legacy positional
set y [torch::gelu $x]
```

For new code, prefer the more explicit named-parameter style:

```tcl
# Modern named parameters
set y [torch::gelu -input $x]
``` 