# torch::avgpool3d / torch::avgPool3d

3-D average pooling layer constructor. Returns handle to `AvgPool3d` module for use with `torch::layer_forward`.

---

## Positional Syntax (backward compatible)
```tcl
set pool [torch::avgpool3d <input> <kernel_size> ?<stride>? ?<padding>? ?<count_include_pad>?]
```

Arguments:
1. `input` – tensor handle
2. `kernel_size` – int or list of 3 ints
3. `stride` – defaults to `kernel_size`
4. `padding` – defaults to `0`
5. `count_include_pad` – 1/0, default 1

## Named Parameter Syntax (new)
```tcl
set pool [torch::avgPool3d \
    -input <tensor> \
    -kernelSize <int|list> \
    ?-stride <int|list>? \
    ?-padding <int|list>? \
    ?-countIncludePad <0|1>?]
```

* `-input` (required) – tensor handle
* `-kernelSize` (required) – int or list {kD kH kW}
* `-stride` – defaults to `-kernelSize`
* `-padding` – defaults to 0
* `-countIncludePad` – whether to include zero-padding in averaging (default 1)

Both snake_case and camelCase names accept either syntax.

## Examples
```tcl
set x [torch::randn -shape {1 1 8 8 8}]

# Positional
set pool1 [torch::avgpool3d $x 2]
set y1   [torch::layer_forward $pool1 $x]

# Named
set pool2 [torch::avgPool3d -input $x -kernelSize 2 -stride 2 -padding 0]
set y2   [torch::layer_forward $pool2 $x]
```

## Migration
Positional code continues to work; migrate by replacing ordered args with named options and optionally camelCase command.

## Error Handling
Missing `-kernelSize` or invalid dimensions produce descriptive errors.

## Return Value
String handle to created module. 