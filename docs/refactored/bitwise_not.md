# torch::bitwise_not

Computes the bitwise NOT of tensor elements.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::bitwise_not -input tensor
torch::bitwise_not -tensor tensor
```

### Positional Parameters (Legacy)
```tcl
torch::bitwise_not tensor
```

### CamelCase Alias
```tcl
torch::bitwiseNot -input tensor
torch::bitwiseNot tensor
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` or `-tensor` | string | Input tensor handle | Yes |

## Returns

Returns a handle to a new tensor containing the element-wise bitwise NOT of the input tensor.

## Description

The `torch::bitwise_not` command performs element-wise bitwise NOT operations. Each bit in every element of the input tensor is flipped (0 becomes 1, 1 becomes 0).

**Mathematical Operation:**
- For each element: `result[i] = ~input[i]`
- Flips all bits: `0` becomes `1`, `1` becomes `0`
- Works on integer tensors (int8, int16, int32, int64, uint8)
- For signed integers, uses two's complement representation
- For unsigned integers, simply flips all bits

**Bitwise NOT Operation:**
- Unary operation (takes only one tensor)
- Flips every bit in the binary representation
- `~5` (binary `101`) becomes `...11111010` (which is `-6` in two's complement)
- `~0` (binary `000`) becomes `...11111111` (which is `-1` in two's complement)

## Examples

### Basic Usage with Named Parameters
```tcl
# Create integer tensor
set input [torch::tensor_create {5 10 15} int32]     ; # 101, 1010, 1111 in binary

# Perform bitwise NOT
set result [torch::bitwise_not -input $input]
# result contains: {-6 -11 -16}  (bitwise NOT with two's complement)
```

### Using Alternative Parameter Name
```tcl
set input [torch::tensor_create {0 1 7} int32]
set result [torch::bitwise_not -tensor $input]
# result contains: {-1 -2 -8}  (~0=-1, ~1=-2, ~7=-8)
```

### Legacy Positional Syntax
```tcl
set input [torch::tensor_create {3 6} int32]
set result [torch::bitwise_not $input]
# result contains: {-4 -7}  (~3=-4, ~6=-7)
```

### CamelCase Alias
```tcl
# Using camelCase with named parameters
set result [torch::bitwiseNot -input $input]

# Using camelCase with positional parameters
set result [torch::bitwiseNot $input]
```

### Working with Different Values
```tcl
# Power of 2 values
set input [torch::tensor_create {1 2 4 8} int32]
set result [torch::bitwise_not -input $input]
# result contains: {-2 -3 -5 -9}
```

### Mathematical Examples
```tcl
# Example 1: NOT operation on 0
set val [torch::tensor_create {0} int32]        ; # 0 = 00000000 in binary
set result [torch::bitwise_not -input $val]
# result: -1 (00000000 -> 11111111 = -1 in two's complement)

# Example 2: NOT operation on positive number
set val [torch::tensor_create {5} int32]        ; # 5 = 00000101 in binary
set result [torch::bitwise_not -input $val]
# result: -6 (00000101 -> 11111010 = -6 in two's complement)

# Example 3: Multiple values
set vals [torch::tensor_create {1 3 7 15} int32]
set result [torch::bitwise_not -input $vals]
# result: {-2 -4 -8 -16}  (~1=-2, ~3=-4, ~7=-8, ~15=-16)
```

## Bitwise NOT Truth Table

| Input | Binary (8-bit) | NOT Result (Binary) | Result (Two's Complement) |
|-------|----------------|-------------------|---------------------------|
| 0     | 00000000       | 11111111         | -1                       |
| 1     | 00000001       | 11111110         | -2                       |
| 3     | 00000011       | 11111100         | -4                       |
| 5     | 00000101       | 11111010         | -6                       |
| 7     | 00000111       | 11111000         | -8                       |

## Two's Complement Explanation

For signed integers, the bitwise NOT operation followed by adding 1 gives the negative value:
- `~n = -(n + 1)` in two's complement representation
- This is why `~5 = -6`, `~0 = -1`, etc.

## Data Type Requirements

- **Input tensor**: Must be integer types (int8, int16, int32, int64, uint8)
- **Output tensor**: Same data type as input tensor
- **Not supported**: Float types (float32, float64) are not supported for bitwise operations

## Error Handling

The command will return an error in the following cases:

1. **Missing parameter**: Input tensor must be specified
2. **Invalid tensor handle**: Tensor handle must exist in storage
3. **Incompatible dtype**: Tensor must be integer type
4. **Float tensors**: Bitwise operations don't work on floating-point tensors

### Error Examples
```tcl
# Missing parameter
catch {torch::bitwise_not} result
# Error: Usage: torch::bitwise_not tensor | torch::bitwise_not -input tensor

# Invalid tensor handle
catch {torch::bitwise_not -input "invalid"} result
# Error: Invalid tensor name

# Unknown parameter
catch {torch::bitwise_not -unknown $t1} result
# Error: Unknown parameter: -unknown
```

## Performance Notes

- Bitwise NOT is typically very fast on modern hardware
- Operations are performed element-wise in parallel when possible
- Memory usage is similar to input tensor size
- No broadcasting involved (unary operation)

## Mathematical Properties

- **Double NOT**: `~~n = n` (applying NOT twice returns original value)
- **Relationship to negation**: `~n = -(n + 1)` for two's complement integers
- **Bitwise complement**: Flips every bit in the binary representation
- **Range preservation**: Output values stay within the data type range

## Related Commands

- `torch::bitwise_and` - Element-wise bitwise AND
- `torch::bitwise_or` - Element-wise bitwise OR
- `torch::bitwise_xor` - Element-wise bitwise XOR
- `torch::bitwise_left_shift` - Element-wise bitwise left shift
- `torch::bitwise_right_shift` - Element-wise bitwise right shift
- `torch::logical_not` - Element-wise logical NOT (different from bitwise NOT)

## Migration Guide

### From Positional to Named Parameters

**Old syntax:**
```tcl
set result [torch::bitwise_not $input]
```

**New syntax:**
```tcl
set result [torch::bitwise_not -input $input]
```

**Benefits of named parameters:**
- **Clarity**: Makes it clear what the input parameter represents
- **Consistency**: Matches the pattern of other refactored commands
- **Maintainability**: Code is more readable and self-documenting
- **Future-proofing**: Named parameters are easier to extend if needed

## Comparison with Logical NOT

It's important to distinguish between bitwise NOT and logical NOT:

| Operation | Command | Input | Output | Purpose |
|-----------|---------|-------|--------|---------|
| Bitwise NOT | `torch::bitwise_not` | 5 | -6 | Flips all bits |
| Logical NOT | `torch::logical_not` | 5 | false/0 | Boolean negation |

```tcl
# Bitwise NOT
set val [torch::tensor_create {5} int32]
set bitwise_result [torch::bitwise_not $val]    ; # -6 (flips bits)

# Logical NOT (for comparison)
set logical_result [torch::logical_not $val]    ; # false/0 (boolean negation)
```

## See Also

- [torch::bitwise_and](bitwise_and.md) - Bitwise AND operation
- [torch::bitwise_or](bitwise_or.md) - Bitwise OR operation
- [torch::bitwise_xor](bitwise_xor.md) - Bitwise XOR operation
- [torch::bitwise_left_shift](bitwise_left_shift.md) - Bitwise left shift operation
- [torch::bitwise_right_shift](bitwise_right_shift.md) - Bitwise right shift operation
- [torch::logical_not](logical_not.md) - Logical NOT operation 