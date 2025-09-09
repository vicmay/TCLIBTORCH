# torch::logical_not

Performs element-wise logical NOT operation on a tensor.

## Syntax

### Positional Arguments (Legacy)
```tcl
torch::logical_not tensor
```

### Named Parameters (Recommended)
```tcl
torch::logical_not -input tensor
torch::logical_not -tensor tensor
```

### CamelCase Alias
```tcl
torch::logicalNot -input tensor
torch::logicalNot tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor` | string | Input tensor handle |
| `-input` | string | Input tensor handle (named parameter) |
| `-tensor` | string | Input tensor handle (alternative named parameter) |

## Returns

Returns a new tensor handle containing the element-wise logical NOT of the input tensor.

## Description

The `torch::logical_not` command performs element-wise logical NOT operation on the input tensor. The operation follows these rules:

- For boolean tensors: `true` becomes `false`, `false` becomes `true`
- For numeric tensors: non-zero values become `false`, zero values become `true`
- The output tensor is always of boolean type (`Bool`)

## Mathematical Properties

For any element `x` in the input tensor:
- `logical_not(true) = false`
- `logical_not(false) = true`
- `logical_not(x ≠ 0) = false`
- `logical_not(x = 0) = true`

## Truth Table

| Input | Output |
|-------|--------|
| true  | false  |
| false | true   |
| 1.0   | false  |
| 0.0   | true   |
| -5.3  | false  |

## Examples

### Basic Usage

```tcl
# Load the extension
load ./libtorchtcl.so

# Create boolean tensors
set bool_tensor [torch::tensor_create -data {1 0 1 0} -dtype bool -device cpu]

# Positional syntax
set result1 [torch::logical_not $bool_tensor]

# Named parameter syntax
set result2 [torch::logical_not -input $bool_tensor]
set result3 [torch::logical_not -tensor $bool_tensor]

# CamelCase alias
set result4 [torch::logicalNot -input $bool_tensor]
```

### Working with Numeric Tensors

```tcl
# Create numeric tensor
set numeric_tensor [torch::tensor_create -data {1.5 0.0 -2.3 0.0} -dtype float32 -device cpu]

# Apply logical NOT
set not_result [torch::logical_not -input $numeric_tensor]
# Result: {false true false true} (Bool dtype)
```

### 2D Tensor Operations

```tcl
# Create 2D boolean tensor
set tensor_1d [torch::tensor_create -data {1 0 0 1} -dtype bool -device cpu]
set tensor_2d [torch::tensor_reshape $tensor_1d "2 2"]

# Apply logical NOT
set result [torch::logical_not -input $tensor_2d]
# Original: [[true, false], [false, true]]
# Result:   [[false, true], [true, false]]
```

### Double Negation

```tcl
# Create boolean tensor
set original [torch::tensor_create -data {1 0 1} -dtype bool -device cpu]

# Apply double negation
set negated [torch::logical_not $original]
set double_negated [torch::logical_not $negated]

# double_negated should equal original
```

## Integration with Other Operations

### De Morgan's Laws

```tcl
# De Morgan's Law: NOT(A AND B) = (NOT A) OR (NOT B)
set tensor_a [torch::tensor_create -data {1 0 1 0} -dtype bool -device cpu]
set tensor_b [torch::tensor_create -data {1 1 0 0} -dtype bool -device cpu]

set and_result [torch::logical_and $tensor_a $tensor_b]
set not_and [torch::logical_not $and_result]

set not_a [torch::logical_not $tensor_a]
set not_b [torch::logical_not $tensor_b]
set not_a_or_not_b [torch::logical_or $not_a $not_b]

# not_and should equal not_a_or_not_b
```

## Data Type Support

| Input Type | Output Type | Supported |
|------------|-------------|-----------|
| Bool       | Bool        | ✅        |
| Float32    | Bool        | ✅        |
| Float64    | Bool        | ✅        |
| Int32      | Bool        | ✅        |
| Int64      | Bool        | ✅        |

## Error Handling

The command will raise an error in the following cases:

```tcl
# Missing arguments
catch {torch::logical_not} ;# Error: Wrong number of arguments

# Invalid tensor handle
catch {torch::logical_not "invalid_handle"} ;# Error: Invalid tensor handle

# Unknown named parameter
catch {torch::logical_not -invalid $tensor} ;# Error: Unknown parameter

# Named parameter without value
catch {torch::logical_not -input} ;# Error: Named parameter requires a value
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set result [torch::logical_not $input_tensor]

# NEW (Named Parameters)
set result [torch::logical_not -input $input_tensor]
# or
set result [torch::logical_not -tensor $input_tensor]

# CamelCase (Modern)
set result [torch::logicalNot -input $input_tensor]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter purpose is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Code is more readable and maintainable
4. **Consistency**: Follows modern API design patterns

## Performance Considerations

- The operation is performed element-wise and is highly efficient
- GPU acceleration is automatically used when tensors are on CUDA devices
- The operation preserves the tensor's shape and device placement
- Memory usage is optimized for boolean output tensors

## See Also

- [`torch::logical_and`](logical_and.md) - Element-wise logical AND
- [`torch::logical_or`](logical_or.md) - Element-wise logical OR
- [`torch::logical_xor`](logical_xor.md) - Element-wise logical XOR
- [`torch::eq`](eq.md) - Element-wise equality comparison
- [`torch::ne`](ne.md) - Element-wise not-equal comparison

## Notes

- Double negation (`logical_not(logical_not(x))`) returns the original boolean values
- The operation follows PyTorch's logical NOT semantics
- All three syntax forms (positional, named, camelCase) produce identical results
- The command maintains full backward compatibility with existing code 