# torch::square

Computes the element-wise square of a tensor.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::square tensor
```

### Named Parameter Syntax (New)
```tcl
torch::square -input tensor
torch::square -tensor tensor
```

### CamelCase Alias
```tcl
torch::Square tensor
torch::Square -input tensor
torch::Square -tensor tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor` / `-input` / `-tensor` | string | Yes | Name of the input tensor |

## Returns

Returns a new tensor handle containing the squared values of the input tensor.

## Description

The `torch::square` command computes the element-wise square of each element in the input tensor. This is equivalent to multiplying each element by itself.

Mathematically, for each element `x` in the input tensor, the output contains `x²`.

## Examples

### Basic Usage

```tcl
# Create a tensor
set tensor [torch::tensorCreate -data {1.0 2.0 3.0 4.0} -dtype float32 -device cpu -requiresGrad true]

# Compute squares using positional syntax
set result [torch::square $tensor]
set values [torch::tensor_to_list $result]
puts $values  ;# Output: 1.0 4.0 9.0 16.0
```

### Named Parameter Syntax

```tcl
# Create a tensor
set tensor [torch::tensorCreate -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad true]

# Compute squares using named parameters
set result [torch::square -input $tensor]
set values [torch::tensor_to_list $result]
puts $values  ;# Output: 4.0 1.0 0.0 1.0 4.0
```

### CamelCase Alias

```tcl
# Create a tensor
set tensor [torch::tensorCreate -data {0.5 1.5 2.5} -dtype float32 -device cpu -requiresGrad true]

# Compute squares using camelCase alias
set result [torch::Square -tensor $tensor]
set values [torch::tensor_to_list $result]
puts $values  ;# Output: 0.25 2.25 6.25
```

### Working with Different Data Types

```tcl
# Integer tensor
set int_tensor [torch::tensorCreate -data {1 2 3 4 5} -dtype int64 -device cpu -requiresGrad false]
set int_result [torch::square $int_tensor]
set int_values [torch::tensor_to_list $int_result]
puts $int_values  ;# Output: 1 4 9 16 25

# Float tensor
set float_tensor [torch::tensorCreate -data {1.5 2.5 3.5} -dtype float32 -device cpu -requiresGrad true]
set float_result [torch::square $float_tensor]
set float_values [torch::tensor_to_list $float_result]
puts $float_values  ;# Output: 2.25 6.25 12.25
```

## Error Handling

### Missing Tensor Argument
```tcl
catch {torch::square} result
puts $result  ;# Output: Usage: torch::square tensor | torch::square -input tensor
```

### Invalid Tensor Name
```tcl
catch {torch::square invalid_tensor} result
puts $result  ;# Output: Invalid tensor name
```

### Wrong Number of Arguments
```tcl
set tensor [torch::tensorCreate -data {1.0} -dtype float32 -device cpu -requiresGrad true]
catch {torch::square $tensor extra_arg} result
puts $result  ;# Output: Wrong number of positional arguments. Expected: torch::square tensor
```

### Unknown Named Parameter
```tcl
set tensor [torch::tensorCreate -data {1.0} -dtype float32 -device cpu -requiresGrad true]
catch {torch::square -unknown $tensor} result
puts $result  ;# Output: Unknown parameter: -unknown
```

## Migration Guide

### From Old Positional Syntax
The old positional syntax continues to work without changes:

```tcl
# Old syntax (still works)
set result [torch::square $tensor]

# New syntax (recommended)
set result [torch::square -input $tensor]
```

### Benefits of Named Parameters
- **Clarity**: Parameter names make the code more readable
- **Flexibility**: Easy to add optional parameters in the future
- **Consistency**: Matches modern API design patterns

## Mathematical Properties

- **Squaring preserves sign**: All squared values are non-negative
- **Squaring amplifies differences**: Larger numbers become much larger when squared
- **Zero remains zero**: `0² = 0`
- **One remains one**: `1² = 1` and `(-1)² = 1`

## Performance Notes

- The operation is element-wise and highly optimized
- Works efficiently on both CPU and GPU tensors
- Memory usage is proportional to the input tensor size
- The output tensor has the same shape as the input tensor

## Related Commands

- `torch::sqrt` - Compute square root
- `torch::pow` - Compute power with arbitrary exponent
- `torch::abs` - Compute absolute value
- `torch::mul` - Element-wise multiplication 