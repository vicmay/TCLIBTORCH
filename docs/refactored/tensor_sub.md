# torch::tensor_sub

Subtracts one tensor from another with optional scaling factor.

## Description

The `torch::tensor_sub` command performs element-wise subtraction between two tensors. It supports an optional alpha parameter that scales the second tensor before subtraction, equivalent to `input - alpha * other`.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_sub tensor1 tensor2
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_sub -input tensor1 -other tensor2 [-alpha value]
```

### CamelCase Alias
```tcl
torch::tensorSub -input tensor1 -other tensor2 [-alpha value]
```

## Parameters

| Parameter | Type   | Required | Default | Description                        |
|-----------|--------|----------|---------|------------------------------------|
| input     | string | Yes      | -       | Name of the first tensor (minuend) |
| other     | string | Yes      | -       | Name of the second tensor (subtrahend) |
| alpha     | double | No       | 1.0     | Scaling factor for the second tensor |

## Return Value

Returns a string containing the handle name of the resulting tensor.

## Examples

### Basic Usage
```tcl
# Create tensors
set t1 [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {2.0 4.0 6.0} -dtype float32 -device cpu]

# Using positional syntax
set result1 [torch::tensor_sub $t1 $t2]

# Using named parameter syntax
set result2 [torch::tensor_sub -input $t1 -other $t2]

# Using camelCase alias
set result3 [torch::tensorSub -input $t1 -other $t2]
```

### With Alpha Parameter
```tcl
# Create tensors
set t1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {2.0 4.0 6.0} -dtype float32 -device cpu]

# Subtract with alpha = 2.0 (equivalent to t1 - 2*t2)
set result [torch::tensor_sub -input $t1 -other $t2 -alpha 2.0]

# This computes: [10, 20, 30] - 2*[2, 4, 6] = [10, 20, 30] - [4, 8, 12] = [6, 12, 18]
```

### Edge Cases
```tcl
# Zero tensor subtraction
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
set result [torch::tensor_sub $t1 $t2]  # Result equals t1

# Negative values
set t1 [torch::tensor_create -data {-1.0 -2.0 -3.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_sub $t1 $t2]  # Result: [-2, -4, -6]

# Alpha = 0 (no subtraction)
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
set result [torch::tensor_sub -input $t1 -other $t2 -alpha 0.0]  # Result equals t1
```

## Error Handling

### Invalid Tensor Names
```tcl
set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
catch {torch::tensor_sub invalid_tensor $t2} result
puts $result  # Output: Invalid first tensor name

set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_sub $t1 invalid_tensor} result
puts $result  # Output: Invalid second tensor name
```

### Missing Parameters
```tcl
catch {torch::tensor_sub} result
puts $result  # Output: Required parameters missing: -input and -other

set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_sub $t1} result
puts $result  # Output: Usage: torch::tensor_sub tensor1 tensor2
```

### Unknown Parameter
```tcl
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set t2 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu]
catch {torch::tensor_sub -input $t1 -other $t2 -unknown_param value} result
puts $result  # Output: Unknown parameter: -unknown_param
```

## Migration Guide

### From Old Syntax to New Syntax
**Before (Positional Only):**
```tcl
set result [torch::tensor_sub $t1 $t2]
```
**After (Named Parameters):**
```tcl
set result [torch::tensor_sub -input $t1 -other $t2]
```
**After (CamelCase):**
```tcl
set result [torch::tensorSub -input $t1 -other $t2]
```

### Benefits of New Syntax
- **Clarity**: Parameter names make the code more readable
- **Flexibility**: Alpha parameter allows for scaled subtraction
- **Maintainability**: Easier to understand and modify
- **Consistency**: Follows modern API design patterns
- **Backward Compatibility**: Old syntax still works

## Mathematical Details

The operation performed is:
```
result = input - alpha * other
```

Where:
- `input` is the first tensor (minuend)
- `other` is the second tensor (subtrahend)
- `alpha` is the scaling factor (default: 1.0)

### Examples of Mathematical Operations
```tcl
# Basic subtraction: result = t1 - t2
set result [torch::tensor_sub -input $t1 -other $t2]

# Scaled subtraction: result = t1 - 2*t2
set result [torch::tensor_sub -input $t1 -other $t2 -alpha 2.0]

# No subtraction: result = t1 - 0*t2 = t1
set result [torch::tensor_sub -input $t1 -other $t2 -alpha 0.0]
```

## Technical Notes
- The operation is element-wise, requiring compatible tensor shapes
- The result tensor has the same shape as the input tensors
- The operation supports broadcasting for compatible shapes
- The alpha parameter allows for efficient scaled subtraction without creating intermediate tensors
- This operation does not modify the original tensors

## Related Commands
- `torch::tensor_add` - Add tensors with optional scaling
- `torch::tensor_mul` - Multiply tensors element-wise
- `torch::tensor_div` - Divide tensors element-wise
- `torch::tensor_matmul` - Matrix multiplication
- `torch::tensor_abs` - Absolute value of tensor elements 