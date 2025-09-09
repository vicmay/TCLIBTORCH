# torch::bitwise_or

Computes the bitwise OR of two tensors element-wise.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::bitwise_or -input tensor1 -other tensor2
torch::bitwise_or -tensor1 tensor1 -tensor2 tensor2
```

### Positional Parameters (Legacy)
```tcl
torch::bitwise_or tensor1 tensor2
```

### CamelCase Alias
```tcl
torch::bitwiseOr -input tensor1 -other tensor2
torch::bitwiseOr tensor1 tensor2
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `-input` or `-tensor1` | string | First input tensor handle | Yes |
| `-other` or `-tensor2` | string | Second input tensor handle | Yes |

## Returns

Returns a handle to a new tensor containing the element-wise bitwise OR of the two input tensors.

## Description

The `torch::bitwise_or` command performs element-wise bitwise OR operations. For each corresponding pair of elements, the operation computes the bitwise OR result where each bit position contains `1` if either or both input bits are `1`, and `0` if both input bits are `0`.

**Mathematical Operation:**
- For each corresponding pair of elements: `result[i] = input1[i] | input2[i]`
- Bitwise OR: `0 | 0 = 0`, `0 | 1 = 1`, `1 | 0 = 1`, `1 | 1 = 1`
- Works on integer tensors (int8, int16, int32, int64, uint8)
- Both tensors must have compatible shapes (broadcastable)

**Bitwise OR Properties:**
- **Commutative**: `A | B = B | A`
- **Associative**: `(A | B) | C = A | (B | C)`
- **Idempotent**: `A | A = A`
- **Identity element**: `A | 0 = A` (OR with 0 returns the original value)
- **Absorption**: `A | 1...1 = 1...1` (OR with all 1s returns all 1s)

## Examples

### Basic Usage with Named Parameters
```tcl
# Create integer tensors
set input1 [torch::tensor_create {5 10 15} int32]    ; # 101, 1010, 1111 in binary
set input2 [torch::tensor_create {3 6 9} int32]     ; # 011, 0110, 1001 in binary

# Perform bitwise OR
set result [torch::bitwise_or -input $input1 -other $input2]
# result contains: {7 14 15}  (111, 1110, 1111 in binary)
```

### Using Alternative Parameter Names
```tcl
set input1 [torch::tensor_create {1 2 4} int32]
set input2 [torch::tensor_create {1 3 5} int32]

set result [torch::bitwise_or -tensor1 $input1 -tensor2 $input2]
# result contains: {1 3 5}  (001|001=001, 010|011=011, 100|101=101)
```

### Legacy Positional Syntax
```tcl
set input1 [torch::tensor_create {8 12} int32]      ; # 1000, 1100 in binary
set input2 [torch::tensor_create {4 10} int32]      ; # 0100, 1010 in binary

set result [torch::bitwise_or $input1 $input2]
# result contains: {12 14}  (1000|0100=1100=12, 1100|1010=1110=14)
```

### CamelCase Alias
```tcl
# Using camelCase with named parameters
set result [torch::bitwiseOr -input $input1 -other $input2]

# Using camelCase with positional parameters
set result [torch::bitwiseOr $input1 $input2]
```

### Working with Different Tensor Shapes (Broadcasting)
```tcl
# Tensor shapes that can be broadcast
set input1 [torch::tensor_create {1 2 3 4} int32]   ; # Shape: [4]
set input2 [torch::tensor_create {1} int32]         ; # Shape: [1]

set result [torch::bitwise_or -input $input1 -other $input2]
# Broadcasting applied: each element of input1 is OR'ed with 1
# result contains: {1 3 3 5}  (1|1=1, 2|1=3, 3|1=3, 4|1=5)
```

### Mathematical Examples
```tcl
# Example 1: OR with zero (identity property)
set vals [torch::tensor_create {5 10 15} int32]
set zeros [torch::tensor_create {0 0 0} int32]
set result [torch::bitwise_or -input $vals -other $zeros]
# result: {5 10 15}  (x | 0 = x)

# Example 2: OR with self (idempotent property)
set vals [torch::tensor_create {7 14 21} int32]
set result [torch::bitwise_or -input $vals -other $vals]
# result: {7 14 21}  (x | x = x)

# Example 3: Commutative property
set a [torch::tensor_create {5 3} int32]
set b [torch::tensor_create {3 5} int32]
set result1 [torch::bitwise_or -input $a -other $b]
set result2 [torch::bitwise_or -input $b -other $a]
# result1 and result2 are identical: {7 7}  (5|3 = 3|5 = 7)
```

### Practical Use Cases
```tcl
# Setting specific bits
set data [torch::tensor_create {8} int32]           ; # 1000 in binary
set mask [torch::tensor_create {3} int32]           ; # 0011 in binary
set result [torch::bitwise_or -input $data -other $mask]
# result: 11 (1000 | 0011 = 1011 = 11) - sets bits 0 and 1

# Combining flags
set flag1 [torch::tensor_create {1} int32]          ; # 0001 in binary
set flag2 [torch::tensor_create {4} int32]          ; # 0100 in binary
set combined [torch::bitwise_or -input $flag1 -other $flag2]
# combined: 5 (0001 | 0100 = 0101 = 5) - combines both flags
```

## Bitwise OR Truth Table

| Input A | Input B | A OR B | Binary Representation |
|---------|---------|--------|-----------------------|
| 0       | 0       | 0      | 000 \| 000 = 000      |
| 1       | 0       | 1      | 001 \| 000 = 001      |
| 0       | 1       | 1      | 000 \| 001 = 001      |
| 1       | 1       | 1      | 001 \| 001 = 001      |
| 5       | 3       | 7      | 101 \| 011 = 111      |
| 12      | 10      | 14     | 1100 \| 1010 = 1110   |

## Data Type Requirements

- **Input tensors**: Must be integer types (int8, int16, int32, int64, uint8)
- **Output tensor**: Same data type as input tensors
- **Shape compatibility**: Tensors must be broadcastable
- **Element-wise operation**: Corresponding elements are processed independently

## Error Handling

The command will return an error in the following cases:

1. **Missing parameters**: Both input tensors must be specified
2. **Invalid tensor handles**: Tensor handles must exist in storage
3. **Incompatible dtypes**: Tensors must be integer types
4. **Shape mismatch**: Tensor shapes must be broadcastable
5. **Float tensors**: Bitwise operations don't work on floating-point tensors

### Error Examples
```tcl
# Missing parameter
catch {torch::bitwise_or -input $t1} result
# Error: Required parameters missing: input and other tensors required

# Invalid tensor handle
catch {torch::bitwise_or -input "invalid" -other $t2} result
# Error: Invalid first tensor name

# Unknown parameter
catch {torch::bitwise_or -input $t1 -unknown $t2} result
# Error: Unknown parameter: -unknown
```

## Performance Notes

- Bitwise OR operations are typically very fast on modern hardware
- Operations are performed element-wise in parallel when possible
- Broadcasting may add some computational overhead for different shapes
- Memory usage scales with the output tensor size
- Integer operations are generally faster than floating-point operations

## Mathematical Properties and Applications

### Boolean Algebra Properties
- **Idempotent Law**: `A | A = A`
- **Commutative Law**: `A | B = B | A`
- **Associative Law**: `(A | B) | C = A | (B | C)`
- **Identity Law**: `A | 0 = A`
- **Domination Law**: `A | 1 = 1` (for single bits)

### Common Applications
1. **Flag Combination**: Combining multiple boolean flags
2. **Bit Setting**: Setting specific bits in a value using a mask
3. **Data Merging**: Combining data from multiple sources
4. **Conditional Logic**: Implementing OR conditions in tensor operations

## Related Commands

- `torch::bitwise_and` - Element-wise bitwise AND
- `torch::bitwise_not` - Element-wise bitwise NOT
- `torch::bitwise_xor` - Element-wise bitwise XOR
- `torch::bitwise_left_shift` - Element-wise bitwise left shift
- `torch::bitwise_right_shift` - Element-wise bitwise right shift
- `torch::logical_or` - Element-wise logical OR (different from bitwise OR)

## Migration Guide

### From Positional to Named Parameters

**Old syntax:**
```tcl
set result [torch::bitwise_or $input1 $input2]
```

**New syntax:**
```tcl
set result [torch::bitwise_or -input $input1 -other $input2]
```

**Benefits of named parameters:**
- **Clarity**: Makes it clear which tensor is the first input and which is the second
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Code is more readable and self-documenting
- **Error reduction**: Less likely to accidentally swap input tensors

## Comparison with Logical OR

It's important to distinguish between bitwise OR and logical OR:

| Operation | Command | Input A | Input B | Output | Purpose |
|-----------|---------|---------|---------|--------|---------|
| Bitwise OR | `torch::bitwise_or` | 5 | 3 | 7 | Operates on individual bits |
| Logical OR | `torch::logical_or` | 5 | 3 | true/1 | Boolean logic operation |

```tcl
# Bitwise OR
set val1 [torch::tensor_create {5} int32]
set val2 [torch::tensor_create {3} int32]
set bitwise_result [torch::bitwise_or $val1 $val2]    ; # 7 (operates on bits)

# Logical OR (for comparison)
set logical_result [torch::logical_or $val1 $val2]    ; # true/1 (boolean logic)
```

## See Also

- [torch::bitwise_and](bitwise_and.md) - Bitwise AND operation
- [torch::bitwise_not](bitwise_not.md) - Bitwise NOT operation
- [torch::bitwise_xor](bitwise_xor.md) - Bitwise XOR operation
- [torch::bitwise_left_shift](bitwise_left_shift.md) - Bitwise left shift operation
- [torch::bitwise_right_shift](bitwise_right_shift.md) - Bitwise right shift operation
- [torch::logical_or](logical_or.md) - Logical OR operation 