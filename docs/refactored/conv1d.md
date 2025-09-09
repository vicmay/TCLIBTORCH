# torch::conv1d

1-D convolution operation supporting both positional and named parameter syntax.

---

## Positional Syntax (back-compat)
```tcl
set y [torch::conv1d <input> <weight> ?<bias>|none? ?<stride>? ?<padding>? ?<dilation>? ?<groups>?]
```
Arguments (defaults match PyTorch):
* `input` – handle of 3-D input tensor `(N, C_in, L)`
* `weight` – handle of 3-D weight tensor `(C_out, C_in/groups, K)`
* `bias` – handle of bias tensor `(C_out)` or `none` (default `none`)
* `stride` (int, default 1)
* `padding` (int, default 0)
* `dilation` (int, default 1)
* `groups` (int, default 1)

## Named Parameter Syntax (modern)
```tcl
set y [torch::conv1d \
    -input  <tensor> \
    -weight <tensor> \
    ?-bias <tensor|none>? \
    ?-stride <int>? \
    ?-padding <int>? \
    ?-dilation <int>? \
    ?-groups <int>?]
```

## Examples
```tcl
# Positional
set y1 [torch::conv1d $x $w $b 2 1]

# Named (no bias)
set y2 [torch::conv1d -input $x -weight $w -bias none -stride 2 -padding 1]
```

## Returns
A new tensor handle representing the convolution result `(N, C_out, L_out)`. 