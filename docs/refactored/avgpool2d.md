# torch::avgpool2d / torch::avgPool2d

Average pooling 2-D layer constructor. Returns a handle to a newly created `AvgPool2d` module that can be applied to tensors via `torch::layer_forward`.

---

## Positional Syntax (Backward Compatible)
```tcl
set pool [torch::avgpool2d <kernel_size> ?<stride>? ?<padding>?]
```
* `kernel_size` (int, required) – Size of the pooling window.
* `stride` (int, optional) – Stride of the window. Defaults to `kernel_size` when omitted.
* `padding` (int, optional) – Implicit zero-padding on both sides. Defaults to `0`.

## Named Parameter Syntax (New)
```tcl
set pool [torch::avgPool2d \
    -kernelSize <int> \
    ?-stride <int>? \
    ?-padding <int>?
]
```
### Options
* `-kernelSize` (int, required) – Pooling window size.
* `-stride` (int, optional) – Stride of the window. Defaults to `-kernelSize` value.
* `-padding` (int, optional) – Zero padding. Defaults to `0`.

The new syntax can also be accessed via snake-case command name:
```tcl
set pool [torch::avgpool2d -kernelSize 2]
```

## camelCase Alias
The command is additionally registered as `torch::avgPool2d` for consistency with named parameters.

---

## Examples

### Basic Usage
```tcl
# Positional
set pool [torch::avgpool2d 2]

# Named parameters
set pool2 [torch::avgPool2d -kernelSize 2 -stride 2]

# Forward pass
set x  [torch::randn {1 1 4 4} float32 cpu false]
set y  [torch::layer_forward $pool  $x]
set y2 [torch::layer_forward $pool2 $x]
```

### Migration
Old code continues to work. To migrate:
```tcl
# BEFORE
set pool [torch::avgpool2d 2 2]

# AFTER – equivalent
set pool [torch::avgPool2d -kernelSize 2 -stride 2]
```

---

## Error Handling
• Missing required `-kernelSize` raises an error.
• Unknown parameters trigger a descriptive exception.

---

## Return Value
Returns a string handle pointing to the created `AvgPool2d` module stored internally.

---

## See Also
* `torch::maxpool2d` / `torch::maxPool2d` – Max-pooling counterpart.
* `torch::layer_forward` – Execute a module on an input tensor. 