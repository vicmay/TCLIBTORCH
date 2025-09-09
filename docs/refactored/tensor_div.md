# torch::tensor_div / torch::tensorDiv

Performs element-wise division of two tensors.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_div tensor1 tensor2
```

### Named Parameters (New Syntax)
```tcl
torch::tensor_div -input tensor1 -other tensor2
torch::tensorDiv -input tensor1 -other tensor2
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor1` / `-input` | string | Name of the first tensor (dividend) | Required |
| `tensor2` / `-other` | string | Name of the second tensor (divisor) | Required |

## Returns

Returns a string handle to the tensor containing the element-wise division result.

## Examples

### Basic Usage

```tcl
# Create tensors for division
set input1 [torch::tensor_create -data {10.0 20.0 30.0 40.0} -dtype float32 -device cpu]
set input2 [torch::tensor_create -data {2.0 4.0 5.0 8.0} -dtype float32 -device cpu]

# Division using positional syntax
set result1 [torch::tensor_div $input1 $input2]

# Division using named syntax
set result2 [torch::tensor_div -input $input1 -other $input2]

# Division using camelCase alias
set result3 [torch::tensorDiv -input $input1 -other $input2]
```

### Mathematical Examples

```tcl
# Simple division
set dividend [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
set divisor [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
set result [torch::tensor_div $dividend $divisor]
# Result: [5.0, 5.0, 6.0]

# Division by one (identity operation)
set tensor [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
set one [torch::tensor_create -data {1.0 1.0 1.0} -dtype float32 -device cpu]
set result [torch::tensor_div $tensor $one]
# Result: [5.0, 10.0, 15.0] (unchanged)

# Division by self (results in ones)
set tensor [torch::tensor_create -data {5.0 10.0 15.0} -dtype float32 -device cpu]
set result [torch::tensor_div $tensor $tensor]
# Result: [1.0, 1.0, 1.0]

# Division with negative values
set neg_tensor [torch::tensor_create -data {-10.0 -20.0 -30.0} -dtype float32 -device cpu]
set pos_tensor [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
set result [torch::tensor_div $neg_tensor $pos_tensor]
# Result: [-5.0, -5.0, -6.0]
```

### Different Data Types

```tcl
# Float32 division
set float32_1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]
set float32_2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float32 -device cpu]
set result [torch::tensor_div $float32_1 $float32_2]

# Float64 division (higher precision)
set float64_1 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float64 -device cpu]
set float64_2 [torch::tensor_create -data {2.0 4.0 5.0} -dtype float64 -device cpu]
set result [torch::tensor_div $float64_1 $float64_2]

# Integer division (truncates to integers)
set int_1 [torch::tensor_create -data {10 20 30} -dtype int32 -device cpu]
set int_2 [torch::tensor_create -data {2 4 5} -dtype int32 -device cpu]
set result [torch::tensor_div $int_1 $int_2]
# Result: [5, 5, 6] (integer division)
```

### Multi-dimensional Tensors

```tcl
# 2D tensor division
set tensor_2d_1 [torch::zeros {2 2} float32 cpu]
set tensor_2d_2 [torch::ones {2 2} float32 cpu]
set result [torch::tensor_div $tensor_2d_1 $tensor_2d_2]
# Result: 2x2 tensor of zeros

# 3D tensor division
set tensor_3d_1 [torch::zeros {2 2 2} float32 cpu]
set tensor_3d_2 [torch::ones {2 2 2} float32 cpu]
set result [torch::tensor_div $tensor_3d_1 $tensor_3d_2]
# Result: 2x2x2 tensor of zeros
```

### Edge Cases

```tcl
# Division by zero (will cause error in PyTorch)
set numerator [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set zero [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
# This will cause an error: torch::tensor_div $numerator $zero

# Zero divided by non-zero
set zero [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
set non_zero [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set result [torch::tensor_div $zero $non_zero]
# Result: [0.0, 0.0, 0.0]

# Large values
set large_1 [torch::tensor_create -data {1000000.0 2000000.0 3000000.0} -dtype float32 -device cpu]
set large_2 [torch::tensor_create -data {1000.0 2000.0 3000.0} -dtype float32 -device cpu]
set result [torch::tensor_div $large_1 $large_2]
# Result: [1000.0, 1000.0, 1000.0]
```

### Parameter Order Flexibility

```tcl
# Named parameters can be specified in any order
set input1 [torch::tensor_create -data {100.0 200.0 300.0} -dtype float32 -device cpu]
set input2 [torch::tensor_create -data {10.0 20.0 30.0} -dtype float32 -device cpu]

# These are equivalent:
set result1 [torch::tensor_div -input $input1 -other $input2]
set result2 [torch::tensor_div -other $input2 -input $input1]
```

## Notes

- The division operation is performed element-wise between the two tensors
- Both tensors must have compatible shapes for broadcasting
- The output tensor has the same shape as the input tensors (after broadcasting)
- For integer tensors, division performs integer division (truncates to integers)
- For floating-point tensors, division performs floating-point division
- Division by zero will cause an error
- Both `torch::tensor_div` and `torch::tensorDiv` are equivalent
- The function preserves the data type and device of the input tensors

## Error Handling

The command will return an error if:
- Either tensor name is invalid or doesn't exist
- Required parameters are missing
- Unknown parameters are provided
- Division by zero is attempted
- Tensors have incompatible shapes for broadcasting

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old code (still works)
set result [torch::tensor_div $tensor1 $tensor2]

# New code (recommended)
set result [torch::tensor_div -input $tensor1 -other $tensor2]
```

## Performance Considerations

- Element-wise division is generally fast and efficient
- The operation is vectorized for optimal performance
- For large tensors, consider using GPU tensors for better performance
- Integer division is typically faster than floating-point division 