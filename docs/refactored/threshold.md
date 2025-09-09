# torch::threshold

Applies the threshold activation function to a tensor. The threshold function sets values below the threshold to a specified value while keeping values above the threshold unchanged.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::threshold tensor threshold value
```

### Named Parameter Syntax (New)
```tcl
torch::threshold -input tensor -threshold value -value replacement_value
```

### CamelCase Alias
```tcl
torch::Threshold -input tensor -threshold value -value replacement_value
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` | string | Yes | Name of the input tensor |
| `threshold` / `-threshold` | double | Yes | Threshold value. Values below this will be replaced |
| `value` / `-value` | double | Yes | Replacement value for elements below the threshold |

## Return Value

Returns a string handle to the resulting tensor.

## Description

The threshold function applies element-wise thresholding to the input tensor:

- If `input[i] > threshold`, then `output[i] = input[i]`
- If `input[i] <= threshold`, then `output[i] = value`

This is useful for implementing activation functions that have a hard cutoff, such as the hard tanh function or for data preprocessing where you want to clip values below a certain threshold.

## Examples

### Basic Usage

```tcl
# Create input tensor
set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0}]

# Apply threshold with positional syntax
set result [torch::threshold $input 0.0 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.5 0.5 0.5 1.0 2.0

# Apply threshold with named syntax
set result2 [torch::threshold -input $input -threshold 0.0 -value 0.5]
puts [torch::tensor_to_list $result2]
# Output: 0.5 0.5 0.5 1.0 2.0
```

### Negative Threshold

```tcl
set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0}]

# Use negative threshold
set result [torch::threshold $input -1.0 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.5 0.5 0.0 1.0 2.0
```

### High Threshold

```tcl
set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0}]

# Use high threshold
set result [torch::threshold $input 1.5 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.5 0.5 0.5 0.5 2.0
```

### 2D Tensor

```tcl
set input [torch::tensor_create -data {{-2.0 -1.0} {0.0 1.0} {2.0 3.0}}]

set result [torch::threshold $input 0.0 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.5 0.5 0.5 1.0 2.0 3.0
```

### Using CamelCase Alias

```tcl
set input [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0}]

# Use camelCase alias
set result [torch::Threshold -input $input -threshold 0.0 -value 0.5]
puts [torch::tensor_to_list $result]
# Output: 0.5 0.5 0.5 1.0 2.0
```

## Migration Guide

### From Positional to Named Syntax

**Old (Positional):**
```tcl
torch::threshold $tensor 0.0 0.5
```

**New (Named Parameters):**
```tcl
torch::threshold -input $tensor -threshold 0.0 -value 0.5
```

### Benefits of Named Syntax

1. **Clarity**: Parameter names make the code more readable
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Easier to understand and modify
4. **Documentation**: Self-documenting code

## Error Handling

The command will throw an error in the following cases:

- **Missing required parameters**: `Required parameters missing: input tensor required`
- **Invalid tensor name**: `Invalid tensor name`
- **Invalid threshold value**: `Invalid threshold value`
- **Invalid replacement value**: `Invalid value`
- **Unknown parameter**: `Unknown parameter: -invalid. Valid parameters are: -input, -threshold, -value`
- **Missing parameter value**: `Missing value for parameter`

### Error Examples

```tcl
# Missing arguments
torch::threshold
# Error: Required parameters missing: input tensor required

# Invalid tensor
torch::threshold invalid_tensor 0.0 0.5
# Error: Invalid tensor name

# Invalid threshold
set input [torch::tensor_create -data {1.0 2.0 3.0}]
torch::threshold $input invalid 0.5
# Error: Invalid threshold value

# Unknown parameter
torch::threshold -input $input -invalid 0.0 -value 0.5
# Error: Unknown parameter: -invalid. Valid parameters are: -input, -threshold, -value
```

## Notes

- **Backward Compatibility**: The positional syntax is fully supported for backward compatibility
- **Data Types**: Works with all numeric tensor data types
- **In-place Operation**: The original tensor is not modified; a new tensor is returned
- **Performance**: The operation is element-wise and efficient for large tensors
- **Memory**: Creates a new tensor, so ensure you have sufficient memory for large inputs

## Related Commands

- `torch::relu` - Rectified Linear Unit activation
- `torch::leaky_relu` - Leaky ReLU activation
- `torch::hardtanh` - Hard tanh activation
- `torch::clamp` - Clamp values to a range 