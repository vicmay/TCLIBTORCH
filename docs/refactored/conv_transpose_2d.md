# `torch::conv_transpose_2d` / `torch::convTranspose2d`

2-D transposed convolution (also called de-convolution) operation.
Supports both legacy positional syntax and the new named-parameter + camelCase API.

---
## 📜 Legacy Positional Syntax (Backward-Compatible)
```tcl
# torch::conv_transpose_2d input weight ?bias? ?stride? ?padding? ?output_padding? ?groups? ?dilation?
set y [torch::conv_transpose_2d $x $w $b {2 2} {1 1} {0 0} 1 {1 1}]
```

## 🆕 Named-Parameter Syntax
```tcl
set y [torch::conv_transpose_2d \
    -input          $x \
    -weight         $w \
    -bias           $b            ;# optional
    -stride         {2 2}         ;# default {1 1}
    -padding        {1 1}         ;# default {0 0}
    -output_padding {0 0}         ;# default {0 0}
    -groups         1             ;# default 1
    -dilation       {1 1}         ;# default {1 1}
]
```

## 🆕 camelCase Alias
`torch::convTranspose2d` is a direct alias and accepts **both** syntaxes:
```tcl
# Positional with camelCase alias
set y [torch::convTranspose2d $x $w]
# Named-parameter with camelCase alias
set y [torch::convTranspose2d -input $x -weight $w -stride {2 2}]
```

---
## Parameters
| Name | Positional Index | Named Flag | Type | Default | Description |
|------|------------------|------------|------|---------|-------------|
| input | 1 | `-input` | tensor | – | Input tensor `N x C_in x H x W` |
| weight | 2 | `-weight` | tensor | – | Convolution weight tensor `C_in x C_out/groups x kH x kW` |
| bias | 3 | `-bias` | tensor/"none" | none | Optional bias tensor `C_out` |
| stride | 4 | `-stride` | int / list(2) | `{1 1}` | Stride of the convolution |
| padding | 5 | `-padding` | int / list(2) | `{0 0}` | Zero-padding added to both sides |
| output_padding | 6 | `-output_padding` | int / list(2) | `{0 0}` | Additional size added to one side of output shape |
| groups | 7 | `-groups` | int | `1` | Number of blocked connections |
| dilation | 8 | `-dilation` | int / list(2) | `{1 1}` | Spacing between kernel elements |

All list parameters accept either a scalar or a 2-element list.

---
## Return Value
A new tensor representing the transposed convolution result.

---
## Examples
### Basic Usage (Positional)
```tcl
set x [torch::randn -shape {1 1 4 4}]
set w [torch::randn -shape {1 1 3 3}]
set y [torch::conv_transpose_2d $x $w]
```

### Basic Usage (Named)
```tcl
set x [torch::randn -shape {1 1 4 4}]
set w [torch::randn -shape {1 1 3 3}]
set y [torch::conv_transpose_2d -input $x -weight $w -stride {2 2}]
```

### camelCase Alias
```tcl
set y [torch::convTranspose2d -input $x -weight $w]
```

---
## Migration Guide
1. Replace positional arguments with explicit `-flag value` pairs.
2. Optionally switch to the camelCase command name.
3. Behaviour remains identical; you can migrate incrementally.

---
## Error Handling
The command validates:
* Presence of `input` and `weight`.
* Tensor names existing in the global storage.
* Correct list lengths (2) for stride/padding/output_padding/dilation.
* Integer validation for `groups`.
Clear TCL errors are thrown for invalid usage.

---
## Test Coverage
See `tests/refactored/conv_transpose_2d_test.tcl` for:
* Positional & named syntax basic functionality
* camelCase alias equivalence
* Parameter validation & error cases
* Data type variations and edge cases

---
© LibTorch TCL Extension – Dual Syntax API Modernization 