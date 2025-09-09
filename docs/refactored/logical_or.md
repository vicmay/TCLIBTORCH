# torch::logical_or

Performs element-wise logical OR operation between two tensors.

## Syntax

### Positional Arguments (Legacy)
```tcl
torch::logical_or tensor1 tensor2
```

### Named Parameters (Recommended)
```tcl
torch::logical_or -input1 tensor1 -input2 tensor2
torch::logical_or -tensor1 tensor1 -tensor2 tensor2
```

### CamelCase Alias
```tcl
torch::logicalOr -input1 tensor1 -input2 tensor2
torch::logicalOr tensor1 tensor2
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `tensor1` | string | First input tensor handle |
| `tensor2` | string | Second input tensor handle |
| `-input1` | string | First input tensor handle (named parameter) |
| `-input2` | string | Second input tensor handle (named parameter) |
| `-tensor1` | string | First input tensor handle (alternative named parameter) |
| `-tensor2` | string | Second input tensor handle (alternative named parameter) |

## Returns

Returns a new tensor handle containing the element-wise logical OR of the two input tensors.

## Description

The `torch::logical_or` command performs element-wise logical OR operation between two tensors. The operation follows these rules:

- For boolean tensors: standard logical OR operation
- For numeric tensors: treats non-zero values as `true`, zero values as `false`
- The output tensor is always of boolean type (`Bool`)
- Broadcasting is supported for tensors with compatible shapes

## Mathematical Properties

For any elements `x` and `y` in the input tensors:
- `logical_or(true, true) = true`
- `logical_or(true, false) = true`
- `logical_or(false, true) = true`
- `logical_or(false, false) = false`
- `logical_or(x ≠ 0, y) = true`
- `logical_or(x = 0, y = 0) = false`

## Truth Table

| Input1 | Input2 | Output |
|--------|--------|--------|
| true   | true   | true   |
| true   | false  | true   |
| false  | true   | true   |
| false  | false  | false  |
| 1.0    | 0.0    | true   |
| 0.0    | 2.5    | true   |
| 0.0    | 0.0    | false  |

## Examples

### Basic Usage

```tcl
# Load the extension
load ./libtorchtcl.so

# Create boolean tensors
set tensor1 [torch::tensor_create -data {1 0 1 0} -dtype bool -device cpu]
set tensor2 [torch::tensor_create -data {0 1 1 0} -dtype bool -device cpu]

# Positional syntax
set result1 [torch::logical_or $tensor1 $tensor2]

# Named parameter syntax
set result2 [torch::logical_or -input1 $tensor1 -input2 $tensor2]
set result3 [torch::logical_or -tensor1 $tensor1 -tensor2 $tensor2]

# CamelCase alias
set result4 [torch::logicalOr -input1 $tensor1 -input2 $tensor2]
```

### Working with Numeric Tensors

```tcl
# Create numeric tensors
set tensor1 [torch::tensor_create -data {1.5 0.0 -2.3 0.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {0.0 2.0 0.0 0.0} -dtype float32 -device cpu]

# Apply logical OR
set or_result [torch::logical_or -input1 $tensor1 -input2 $tensor2]
# Result: {true true true false} (Bool dtype)
```

### Broadcasting Operations

```tcl
# Create tensors with different shapes
set tensor1 [torch::tensor_create -data {1 0 1} -dtype bool -device cpu]
set tensor2 [torch::tensor_create -data {0} -dtype bool -device cpu]

# Broadcasting OR operation
set result [torch::logical_or -input1 $tensor1 -input2 $tensor2]
# tensor1: [true, false, true]
# tensor2: [false] (broadcast to [false, false, false])
# Result:  [true, false, true]
```

### 2D Tensor Operations

```tcl
# Create 2D boolean tensors
set tensor1_1d [torch::tensor_create -data {1 0 0 1} -dtype bool -device cpu]
set tensor1_2d [torch::tensor_reshape $tensor1_1d "2 2"]
set tensor2_1d [torch::tensor_create -data {0 1 1 0} -dtype bool -device cpu]
set tensor2_2d [torch::tensor_reshape $tensor2_1d "2 2"]

# Apply logical OR
set result [torch::logical_or -input1 $tensor1_2d -input2 $tensor2_2d]
# tensor1: [[true, false], [false, true]]
# tensor2: [[false, true], [true, false]]
# Result:  [[true, true], [true, true]]
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

### Logical Properties

```tcl
# Identity: A OR false = A
set tensor_a [torch::tensor_create -data {1 0 1 0} -dtype bool -device cpu]
set tensor_false [torch::tensor_create -data {0 0 0 0} -dtype bool -device cpu]
set identity_result [torch::logical_or $tensor_a $tensor_false]
# identity_result equals tensor_a

# Annihilator: A OR true = true
set tensor_true [torch::tensor_create -data {1 1 1 1} -dtype bool -device cpu]
set annihilator_result [torch::logical_or $tensor_a $tensor_true]
# annihilator_result is all true

# Commutative: A OR B = B OR A
set result_ab [torch::logical_or $tensor_a $tensor_b]
set result_ba [torch::logical_or $tensor_b $tensor_a]
# result_ab equals result_ba
```

## Data Type Support

| Input Type | Output Type | Supported |
|------------|-------------|-----------|
| Bool       | Bool        | ✅        |
| Float32    | Bool        | ✅        |
| Float64    | Bool        | ✅        |
| Int32      | Bool        | ✅        |
| Int64      | Bool        | ✅        |

## Broadcasting Support

The command supports PyTorch's broadcasting rules:

- Tensors with the same shape are combined element-wise
- Smaller tensors are broadcast to match larger tensor dimensions
- Dimensions of size 1 can be broadcast to any size
- Missing dimensions are treated as size 1

```tcl
# Examples of valid broadcasting
set a [torch::tensor_create -data {1 0} -dtype bool -device cpu]           # Shape: [2]
set b [torch::tensor_create -data {1} -dtype bool -device cpu]             # Shape: [1]
set result [torch::logical_or $a $b]  # Broadcasting [1] to [2]
```

## Error Handling

The command will raise an error in the following cases:

```tcl
# Missing arguments
catch {torch::logical_or} ;# Error: Wrong number of arguments

# Single argument
catch {torch::logical_or $tensor1} ;# Error: Missing second tensor

# Invalid tensor handles
catch {torch::logical_or "invalid_handle" $tensor2} ;# Error: Invalid tensor handle
catch {torch::logical_or $tensor1 "invalid_handle"} ;# Error: Invalid tensor handle

# Unknown named parameter
catch {torch::logical_or -invalid $tensor1 -input2 $tensor2} ;# Error: Unknown parameter

# Named parameter without value
catch {torch::logical_or -input1 $tensor1 -input2} ;# Error: Named parameter requires a value

# Missing required parameters
catch {torch::logical_or -input1 $tensor1} ;# Error: Missing input2 parameter
```

## Migration Guide

### From Positional to Named Parameters

```tcl
# OLD (Positional)
set result [torch::logical_or $tensor1 $tensor2]

# NEW (Named Parameters)
set result [torch::logical_or -input1 $tensor1 -input2 $tensor2]
# or
set result [torch::logical_or -tensor1 $tensor1 -tensor2 $tensor2]

# CamelCase (Modern)
set result [torch::logicalOr -input1 $tensor1 -input2 $tensor2]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter roles are explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Code is more readable and maintainable
4. **Consistency**: Follows modern API design patterns

## Performance Considerations

- The operation is performed element-wise and is highly efficient
- GPU acceleration is automatically used when tensors are on CUDA devices
- Broadcasting operations are optimized by PyTorch
- The operation preserves the devices of input tensors
- Memory usage is optimized for boolean output tensors

## See Also

- [`torch::logical_and`](logical_and.md) - Element-wise logical AND
- [`torch::logical_not`](logical_not.md) - Element-wise logical NOT
- [`torch::logical_xor`](logical_xor.md) - Element-wise logical XOR
- [`torch::eq`](eq.md) - Element-wise equality comparison
- [`torch::ne`](ne.md) - Element-wise not-equal comparison

## Notes

- The operation follows PyTorch's logical OR semantics
- Broadcasting follows standard PyTorch broadcasting rules
- All three syntax forms (positional, named, camelCase) produce identical results
- The command maintains full backward compatibility with existing code
- Output tensor dtype is always `Bool` regardless of input types 