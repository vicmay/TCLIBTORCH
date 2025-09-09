# torch::tensor_where

Selects elements from two tensors based on a condition tensor (elementwise if-then-else).

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_where condition x y
```

### Named Parameters (New)
```tcl
torch::tensor_where -condition cond_tensor -x x_tensor -y y_tensor
```

### CamelCase Alias
```tcl
torch::tensorWhere -condition cond_tensor -x x_tensor -y y_tensor
```

## Parameters

| Parameter   | Type          | Required | Description                                 |
|-------------|---------------|----------|---------------------------------------------|
| condition   | tensor_handle | Yes      | Boolean tensor (dtype bool) for selection    |
| x           | tensor_handle | Yes      | Tensor to select from when condition is true |
| y           | tensor_handle | Yes      | Tensor to select from when condition is false|

- All tensors must be broadcastable to the same shape.
- The `condition` tensor must have dtype `bool`.

## Return Value

Returns a tensor handle containing the selected elements.

## Examples

### Basic Usage
```tcl
# Create tensors
set cond [torch::tensor_create -data {1 0 1 0} -dtype bool]
set x [torch::tensor_create -data {10 20 30 40} -dtype float32]
set y [torch::tensor_create -data {1 2 3 4} -dtype float32]

# Elementwise selection
set out [torch::tensor_where $cond $x $y]
puts "Result: [torch::tensor_to_list $out]"  ;# 10.0 2.0 30.0 4.0
```

### Named Parameters
```tcl
set cond [torch::tensor_create -data {0 1 0 1} -dtype bool]
set x [torch::tensor_create -data {5 6 7 8} -dtype float32]
set y [torch::tensor_create -data {0 0 0 0} -dtype float32]
set out [torch::tensor_where -condition $cond -x $x -y $y]
puts "Result: [torch::tensor_to_list $out]"  ;# 0.0 6.0 0.0 8.0
```

### CamelCase Alias
```tcl
set cond [torch::tensor_create -data {1 0 1 0} -dtype bool]
set x [torch::tensor_create -data {100 200 300 400} -dtype float32]
set y [torch::tensor_create -data {10 20 30 40} -dtype float32]
set out [torch::tensorWhere -condition $cond -x $x -y $y]
puts "Result: [torch::tensor_to_list $out]"  ;# 100.0 20.0 300.0 40.0
```

### Broadcasting
```tcl
set cond [torch::tensor_create -data {{1 0} {0 1}} -dtype bool]
set x [torch::tensor_create -data {10 20} -dtype float32]
set y [torch::tensor_create -data {1 2} -dtype float32]
set out [torch::tensor_where $cond $x $y]
puts "Result: [torch::tensor_to_list $out]"  ;# 10.0 2.0 1.0 20.0
```

## Migration from Positional to Named Parameters

### Before (Positional)
```tcl
set out [torch::tensor_where $cond $x $y]
```
### After (Named)
```tcl
set out [torch::tensor_where -condition $cond -x $x -y $y]
```

## Error Handling
- All parameters are required.
- The `condition` tensor must exist and have dtype `bool`.
- All tensors must be broadcastable to the same shape.
- Clear error messages are provided for missing or invalid parameters.

## Notes
- The command supports both CPU and CUDA tensors.
- The output tensor has the broadcasted shape of the inputs.
- The command is fully backward compatible with the original positional syntax. 