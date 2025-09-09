# torch::bitwise_left_shift

Computes the bitwise left shift of tensor elements.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::bitwise_left_shift -input tensor1 -other tensor2
torch::bitwise_left_shift -tensor1 tensor1 -tensor2 tensor2
```

### Positional Parameters (Legacy)
```tcl
torch::bitwise_left_shift tensor1 tensor2
```

### CamelCase Alias
```tcl
torch::bitwiseLeftShift -input tensor1 -other tensor2
torch::bitwiseLeftShift tensor1 tensor2
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` or `-tensor1` | string | Input tensor handle | Yes |
| `-other` or `-tensor2` | string | Shift amount tensor handle | Yes |

## Returns

Returns a handle to a new tensor containing the element-wise bitwise left shift of the input tensor by the shift amounts.

## Description

The `torch::bitwise_left_shift` command performs element-wise bitwise left shift operations. Each element in the input tensor is shifted left by the corresponding number of positions specified in the shift amount tensor.

**Mathematical Operation:**
- For each corresponding pair of elements: `result[i] = input[i] << other[i]`
- Left shift by `n` positions is equivalent to multiplying by `2^n`
- Works on integer tensors (int8, int16, int32, int64, uint8)
- Both tensors must have compatible shapes (broadcastable)

**Bitwise Left Shift:**
- Shifts bits to the left, filling with zeros from the right
- `5 << 1` means `101` becomes `1010` (5 becomes 10)
- `3 << 2` means `011` becomes `1100` (3 becomes 12)

## Examples

### Basic Usage with Named Parameters
```tcl
# Create integer tensors
set input [torch::tensor_create {1 2 4} int32]     ; # 001, 010, 100 in binary
set shifts [torch::tensor_create {1 1 1} int32]    ; # Shift each left by 1

# Perform bitwise left shift
set result [torch::bitwise_left_shift -input $input -other $shifts]
# result contains: {2 4 8}  (010, 100, 1000 in binary)
```

### Using Alternative Parameter Names
```tcl
set input [torch::tensor_create {3 5 7} int32]
set shifts [torch::tensor_create {2 1 3} int32]

set result [torch::bitwise_left_shift -tensor1 $input -tensor2 $shifts]
# result contains: {12 10 56}  (3<<2=12, 5<<1=10, 7<<3=56)
```

### Legacy Positional Syntax
```tcl
set input [torch::tensor_create {1 2} int32]
set shifts [torch::tensor_create {2 3} int32]

set result [torch::bitwise_left_shift $input $shifts]
# result contains: {4 16}  (1<<2=4, 2<<3=16)
```

### CamelCase Alias
```tcl
# Using camelCase with named parameters
set result [torch::bitwiseLeftShift -input $input -other $shifts]

# Using camelCase with positional parameters
set result [torch::bitwiseLeftShift $input $shifts]
```

### Working with Different Tensor Shapes (Broadcasting)
```tcl
# Tensor shapes that can be broadcast
set input [torch::tensor_create {1 2 3 4} int32]   ; # Shape: [4]
set shifts [torch::tensor_create {2} int32]        ; # Shape: [1]

set result [torch::bitwise_left_shift -input $input -other $shifts]
# Broadcasting applied: each element of input is shifted left by 2
# result contains: {4 8 12 16}  (1<<2=4, 2<<2=8, 3<<2=12, 4<<2=16)
```

### Mathematical Examples
```tcl
# Example 1: Single shift
set val [torch::tensor_create {5} int32]      ; # 5 = 101 in binary
set shift [torch::tensor_create {1} int32]    ; # Shift left by 1
set result [torch::bitwise_left_shift -input $val -other $shift]
# result: 10 (101 << 1 = 1010)

# Example 2: Multiple shifts
set vals [torch::tensor_create {1 3 7} int32]    ; # 001, 011, 111 in binary
set shifts [torch::tensor_create {1 2 1} int32]  ; # Different shift amounts
set result [torch::bitwise_left_shift -input $vals -other $shifts]
# result: {2 12 14}  (1<<1=2, 3<<2=12, 7<<1=14)
```

## Bitwise Left Shift Truth Table

| Input | Shift | Result | Binary Representation |
|-------|-------|--------|----------------------|
| 1     | 1     | 2      | 001 << 1 = 010       |
| 2     | 1     | 4      | 010 << 1 = 100       |
| 3     | 2     | 12     | 011 << 2 = 1100      |
| 5     | 1     | 10     | 101 << 1 = 1010      |

## Data Type Requirements

- **Input tensors**: Must be integer types (int8, int16, int32, int64, uint8)
- **Output tensor**: Same data type as input tensors
- **Shape compatibility**: Tensors must be broadcastable
- **Shift amounts**: Should be non-negative integers

## Error Handling

The command will return an error in the following cases:

1. **Missing parameters**: Both input and shift amount tensors must be specified
2. **Invalid tensor handles**: Tensor handles must exist in storage
3. **Incompatible dtypes**: Tensors must be integer types
4. **Shape mismatch**: Tensor shapes must be broadcastable
5. **Invalid shift amounts**: Negative shift amounts may cause undefined behavior

### Error Examples
```tcl
# Missing parameter
catch {torch::bitwise_left_shift -input $t1} result
# Error: Required parameters missing: input and other tensors required

# Invalid tensor handle
catch {torch::bitwise_left_shift -input "invalid" -other $t2} result
# Error: Invalid first tensor name

# Unknown parameter
catch {torch::bitwise_left_shift -input $t1 -unknown $t2} result
# Error: Unknown parameter: -unknown
```

## Related Commands

- `torch::bitwise_right_shift` - Element-wise bitwise right shift
- `torch::bitwise_and` - Element-wise bitwise AND
- `torch::bitwise_or` - Element-wise bitwise OR
- `torch::bitwise_xor` - Element-wise bitwise XOR
- `torch::bitwise_not` - Element-wise bitwise NOT

## Migration Guide

### From Positional to Named Parameters

**Old syntax:**
```tcl
set result [torch::bitwise_left_shift $input $shifts]
```

**New syntax:**
```tcl
set result [torch::bitwise_left_shift -input $input -other $shifts]
```

**Benefits of named parameters:**
- **Clarity**: Makes it clear which tensor is the input and which contains shift amounts
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is more readable and self-documenting
- **Error reduction**: Less likely to accidentally swap input and shift tensors

## Performance Notes

- Bitwise shift operations are typically very fast on modern hardware
- Left shift by `n` is equivalent to multiplication by `2^n` but faster
- Broadcasting may add some computational overhead for different shapes
- Operations are performed element-wise in parallel when possible

## Mathematical Properties

- **Left shift by 0**: No change to the input value
- **Left shift by 1**: Equivalent to multiplying by 2
- **Left shift by n**: Equivalent to multiplying by 2^n
- **Overflow behavior**: Results may overflow for large shift amounts
- **Sign preservation**: For signed integers, the sign bit behavior depends on the specific integer type

## See Also

- [torch::bitwise_right_shift](bitwise_right_shift.md) - Bitwise right shift operation
- [torch::bitwise_and](bitwise_and.md) - Bitwise AND operation
- [torch::bitwise_or](bitwise_or.md) - Bitwise OR operation
- [torch::bitwise_xor](bitwise_xor.md) - Bitwise XOR operation
- [torch::bitwise_not](bitwise_not.md) - Bitwise NOT operation 