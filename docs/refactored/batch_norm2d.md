# torch::batchnorm2d / torch::batchNorm2d

2-D batch normalization layer constructor supporting both legacy positional and modern named syntax.

---

## Positional Syntax (backward compatible)
```tcl
set bn [torch::batchnorm2d <num_features> ?<eps>? ?<momentum>? ?<affine>? ?<track_running_stats>?]
```
* `num_features` (int, required) – number of feature channels.
* `eps` (float, default 1e-5) – numerical stability term.
* `momentum` (float, default 0.1).
* `affine` (bool, default true) – if false, scale/shift not learned.
* `track_running_stats` (bool, default true).

## Named Parameter Syntax (new)
```tcl
set bn [torch::batchNorm2d \
    -numFeatures <int> \
    ?-eps <float>? \
    ?-momentum <float>? \
    ?-affine <bool>? \
    ?-trackRunningStats <bool>?]
```
Booleans accept `1/0 true/false`.

## Examples
```tcl
# Positional
set bn1 [torch::batchnorm2d 64]

# Named
set bn2 [torch::batchNorm2d -numFeatures 64 -eps 1e-4 -momentum 0.05]
```

## Migration
Swap ordered arguments for named options and optionally switch to camelCase command.

## Return Value
Returns a handle to the created `BatchNorm2d` module for use with `torch::layer_forward`. 