# torch::bitwise_and

Computes the bitwise AND of two tensors element-wise.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::bitwise_and -input tensor1 -other tensor2
torch::bitwise_and -tensor1 tensor1 -tensor2 tensor2
```

### Positional Parameters (Legacy)
```tcl
torch::bitwise_and tensor1 tensor2
```

### CamelCase Alias
```tcl
torch::bitwiseAnd -input tensor1 -other tensor2
torch::bitwiseAnd tensor1 tensor2
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` or `-tensor1` | string | First input tensor handle | Yes |
| `-other` or `-tensor2` | string | Second input tensor handle | Yes |

## Returns

Returns a handle to a new tensor containing the element-wise bitwise AND of the two input tensors.

## Description

The `torch::bitwise_and` command performs element-wise bitwise AND operations between two tensors. The operation is applied to corresponding elements of the input tensors, producing a new tensor with the results.

**Mathematical Operation:**
- For each corresponding pair of elements: `result[i] = input[i] & other[i]`
- Works on integer tensors (int8, int16, int32, int64, uint8)
- Both tensors must have compatible shapes (broadcastable)

## Examples

### Basic Usage with Named Parameters
```tcl
# Create integer tensors
set t1 [torch::tensor_create {5 7 3} int32]    ; # 101, 111, 011 in binary
set t2 [torch::tensor_create {3 3 1} int32]    ; # 011, 011, 001 in binary

# Perform bitwise AND
set result [torch::bitwise_and -input $t1 -other $t2]
# result contains: {1 3 1}  (001, 011, 001 in binary)
```

### Using Alternative Parameter Names
```tcl
set t1 [torch::tensor_create {12 8 4} int32]
set t2 [torch::tensor_create {10 6 2} int32]

set result [torch::bitwise_and -tensor1 $t1 -tensor2 $t2]
```

### Legacy Positional Syntax
```tcl
set t1 [torch::tensor_create {15 7} int32]
set t2 [torch::tensor_create {8 3} int32]

set result [torch::bitwise_and $t1 $t2]
```

### CamelCase Alias
```tcl
# Using camelCase with named parameters
set result [torch::bitwiseAnd -input $t1 -other $t2]

# Using camelCase with positional parameters
set result [torch::bitwiseAnd $t1 $t2]
```

### Working with Different Tensor Shapes (Broadcasting)
```tcl
# Tensor shapes that can be broadcast
set t1 [torch::tensor_create {1 2 3 4} int32]     ; # Shape: [4]
set t2 [torch::tensor_create {3} int32]           ; # Shape: [1]

set result [torch::bitwise_and -input $t1 -other $t2]
# Broadcasting applied: each element of t1 is AND-ed with 3
```

## Bitwise AND Truth Table

| A | B | A & B |
|---|---|-------|
| 0 | 0 |   0   |
| 0 | 1 |   0   |
| 1 | 0 |   0   |
| 1 | 1 |   1   |

## Data Type Requirements

- **Input tensors**: Must be integer types (int8, int16, int32, int64, uint8)
- **Output tensor**: Same data type as input tensors
- **Shape compatibility**: Tensors must be broadcastable

## Error Handling

The command will return an error in the following cases:

1. **Missing parameters**: Both input and other tensors must be specified
2. **Invalid tensor handles**: Tensor handles must exist in storage
3. **Incompatible dtypes**: Tensors must be integer types
4. **Shape mismatch**: Tensor shapes must be broadcastable

### Error Examples
```tcl
# Missing parameter
catch {torch::bitwise_and -input $t1} result
# Error: Required parameters missing: input and other tensors required

# Invalid tensor handle
catch {torch::bitwise_and -input "invalid" -other $t2} result
# Error: Invalid first tensor name

# Unknown parameter
catch {torch::bitwise_and -input $t1 -unknown $t2} result
# Error: Unknown parameter: -unknown
```

## Related Commands

- `torch::bitwise_or` - Element-wise bitwise OR
- `torch::bitwise_xor` - Element-wise bitwise XOR
- `torch::bitwise_not` - Element-wise bitwise NOT
- `torch::logical_and` - Element-wise logical AND (for boolean operations)

## Migration Guide

### From Positional to Named Parameters

**Old syntax:**
```tcl
set result [torch::bitwise_and $tensor1 $tensor2]
```

**New syntax:**
```tcl
set result [torch::bitwise_and -input $tensor1 -other $tensor2]
```

**Benefits of named parameters:**
- **Clarity**: Parameter purpose is explicit
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is more readable and self-documenting
- **Error reduction**: Less likely to accidentally swap parameters

## Performance Notes

- Bitwise operations are typically very fast on modern hardware
- Broadcasting may add some computational overhead for different shapes
- Operations are performed element-wise in parallel when possible

## See Also

- [torch::bitwise_or](bitwise_or.md) - Bitwise OR operation
- [torch::bitwise_xor](bitwise_xor.md) - Bitwise XOR operation  
- [torch::bitwise_not](bitwise_not.md) - Bitwise NOT operation
- [torch::logical_and](logical_and.md) - Logical AND operation 