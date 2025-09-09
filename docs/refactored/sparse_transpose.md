# torch::sparse_transpose

Transposes a sparse tensor by swapping two dimensions.

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::sparse_transpose sparse_tensor dim0 dim1
```

### Named Parameter Syntax
```tcl
torch::sparse_transpose -tensor sparse_tensor -dim0 dim0 -dim1 dim1
```

### CamelCase Alias
```tcl
torch::sparseTranspose -tensor sparse_tensor -dim0 dim0 -dim1 dim1
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| sparse_tensor | tensor | The input sparse tensor to transpose |
| dim0 | int | The first dimension to swap |
| dim1 | int | The second dimension to swap |

## Return Value

Returns a new sparse tensor with dimensions `dim0` and `dim1` swapped.

## Examples

### Basic Usage
```tcl
# Create a sparse tensor
set indices [torch::tensor_create -data {{0 1 2} {0 1 2}} -dtype "long"]
set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float"]
set sparse_tensor [torch::sparse_tensor_create -indices $indices -values $values -size {3 3}]

# Transpose using positional syntax
set result1 [torch::sparse_transpose $sparse_tensor 0 1]

# Transpose using named parameter syntax
set result2 [torch::sparse_transpose -tensor $sparse_tensor -dim0 0 -dim1 1]

# Transpose using camelCase alias
set result3 [torch::sparseTranspose -tensor $sparse_tensor -dim0 0 -dim1 1]
```

## Error Handling

The command will raise an error in the following cases:
- If the input tensor is not a valid sparse tensor
- If either dimension index is out of bounds
- If required parameters are missing in named parameter syntax

## Migration Guide

### From Positional to Named Parameter Syntax

Old code:
```tcl
torch::sparse_transpose $tensor 0 1
```

New code:
```tcl
torch::sparse_transpose -tensor $tensor -dim0 0 -dim1 1
# or
torch::sparseTranspose -tensor $tensor -dim0 0 -dim1 1
``` 