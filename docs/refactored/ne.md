# torch::ne / torch::Ne

Element-wise not equal comparison between two tensors.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::ne -input1 tensor1 -input2 tensor2
torch::ne -tensor1 tensor1 -tensor2 tensor2
torch::Ne -input1 tensor1 -input2 tensor2
torch::Ne -tensor1 tensor1 -tensor2 tensor2
```

### Positional Parameters (Legacy)
```tcl
torch::ne tensor1 tensor2
torch::Ne tensor1 tensor2
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `-input1` or `-tensor1` | tensor | Required | First tensor for comparison |
| `-input2` or `-tensor2` | tensor | Required | Second tensor for comparison |

## Description

The `torch::ne` command performs element-wise not equal comparison between two tensors. It returns a new tensor of the same shape as the input tensors, containing boolean values (0 or 1) where:
- 1 indicates the elements are not equal
- 0 indicates the elements are equal

The operation supports broadcasting if the tensors have compatible shapes.

## Return Value

Returns a handle to a new tensor containing the boolean results of the element-wise not equal comparison.

## Examples

### Basic Usage
```tcl
# Create test tensors
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set t2 [torch::tensor_create -data {1.0 3.0 3.0} -dtype float32]

# Compare using positional syntax
set result [torch::ne $t1 $t2]
# Result will be {0 1 0} (only the middle element is different)

# Compare using named parameters
set result [torch::ne -input1 $t1 -input2 $t2]
# Same result: {0 1 0}
```

### Using with Different Shapes (Broadcasting)
```tcl
# Create tensors with different but compatible shapes
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set t2 [torch::tensor_create -data {2.0} -dtype float32]

# Compare (t2 will be broadcast)
set result [torch::ne $t1 $t2]
# Result will be {1 0 1} (only the middle element equals 2.0)
```

### Using camelCase Alias
```tcl
# The command is also available with camelCase alias
set result [torch::Ne -input1 $t1 -input2 $t2]
```

## Error Handling

The command will raise an error if:
- Either tensor handle is invalid
- Required parameters are missing
- Invalid parameter names are provided
- The tensor shapes are incompatible for broadcasting

## See Also

- `torch::eq` - Element-wise equality comparison
- `torch::lt` - Element-wise less than comparison
- `torch::le` - Element-wise less than or equal comparison
- `torch::gt` - Element-wise greater than comparison
- `torch::ge` - Element-wise greater than or equal comparison
