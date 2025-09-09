# torch::adaptive_maxpool3d / torch::adaptiveMaxpool3d

Applies 3-D adaptive max-pooling over an input tensor.

## Aliases
- `torch::adaptive_maxpool3d`  (snake-case, legacy)
- `torch::adaptiveMaxpool3d`   (camelCase, preferred)

Both aliases invoke the **same** function and support **two syntaxes**.

---

## 1. Positional Syntax (backward-compatible)
```tcl
set output [torch::adaptive_maxpool3d <inputTensor> <outputSize>]
```
* **inputTensor** — tensor handle with shape `(N, C, D, H, W)`
* **outputSize**  — either single integer *s* (produces `(s, s, s)`), or a list `{d h w}`

---

## 2. Named Parameter Syntax (recommended)
```tcl
set output [torch::adaptiveMaxpool3d \
    -input        <inputTensor> \
    -output_size  <int>|{d h w} ]
```
Parameter aliases:
- `-input` / `-tensor`
- `-output_size` / `-outputSize`

### Parameters
| Name | Type | Default | Description |
|------|------|---------|-------------|
| **-input** | tensor handle | — | Input tensor `(N, C, D, H, W)` |
| **-output_size** | int or int list (length 3) | — | Target output spatial size. `2` ➜ `(2,2,2)`; `{3 2 1}` ➜ `(3,2,1)` |

### Returns
Tensor handle pointing to pooled output `(N, C, *)`.

---

## Examples
```tcl
# Create a 3-D volume tensor (4×4×4)
set vol [torch::tensor_create {1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16} float32 cpu false]
set vol [torch::tensor_reshape $vol {1 1 4 4 4}]

# Positional syntax
set pooled1 [torch::adaptive_maxpool3d $vol 2]

# Named syntax (preferred)
set pooled2 [torch::adaptiveMaxpool3d -input $vol -output_size 2]
```

---

## Error Handling
| Condition | Message |
|-----------|---------|
| Missing required parameters | `Required parameters: input tensor and positive output_size` |
| Invalid tensor handle | `Invalid input tensor name` |
| Bad `-output_size` value | `Output size must be an int or list of 3 ints` |
| Unknown parameter | `Unknown parameter: -foo` |

---

## Migration Guide
Legacy code:
```tcl
set out [torch::adaptive_maxpool3d $x 2]
```
Modern equivalent (no functional change):
```tcl
set out [torch::adaptiveMaxpool3d -input $x -output_size 2]
```

---

## Test Coverage
See `tests/refactored/adaptive_maxpool3d_test.tcl` for comprehensive test cases covering positional syntax, named parameters, camelCase alias, edge cases, and error handling. 