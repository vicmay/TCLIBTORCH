# torch::tensor_add / torch::tensorAdd

Adds two tensors element-wise with optional scaling factor.

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::tensor_add tensor1 tensor2 ?alpha?
```

### Named Parameters (New Syntax)
```tcl
torch::tensor_add -input1 tensor1 -input2 tensor2 ?-alpha value?
torch::tensor_add -input tensor1 -other tensor2 ?-alpha value?
torch::tensorAdd -input1 tensor1 -input2 tensor2 ?-alpha value?
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor1` / `-input1` / `-input` | string | Name of the first input tensor | Required |
| `tensor2` / `-input2` / `-other` | string | Name of the second input tensor | Required |
| `-alpha` | double | Scaling factor for the second tensor | 1.0 |

## Returns

Returns a string handle to the tensor containing the result of `tensor1 + alpha * tensor2`.

## Examples

### Basic Usage

```tcl
# Create two tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]

# Add tensors (positional syntax)
set result1 [torch::tensor_add $tensor1 $tensor2]

# Add tensors (named syntax)
set result2 [torch::tensor_add -input1 $tensor1 -input2 $tensor2]

# Add tensors (camelCase alias)
set result3 [torch::tensorAdd -input1 $tensor1 -input2 $tensor2]
```

### With Alpha Parameter

```tcl
# Create two tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]

# Add with alpha = 2.0 (positional)
set result1 [torch::tensor_add $tensor1 $tensor2 2.0]

# Add with alpha = 2.0 (named)
set result2 [torch::tensor_add -input1 $tensor1 -input2 $tensor2 -alpha 2.0]

# Add with alpha = 0.5 (alternative parameter names)
set result3 [torch::tensor_add -input $tensor1 -other $tensor2 -alpha 0.5]
```

### Mathematical Examples

```tcl
# Basic addition: [1, 2, 3] + [4, 5, 6] = [5, 7, 9]
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]

# Scaled addition: [1, 2, 3] + 2.0 * [4, 5, 6] = [1, 2, 3] + [8, 10, 12] = [9, 12, 15]
set result [torch::tensor_add $tensor1 $tensor2 2.0]

# Subtraction using negative alpha: [1, 2, 3] + (-1.0) * [4, 5, 6] = [1, 2, 3] - [4, 5, 6] = [-3, -3, -3]
set result [torch::tensor_add $tensor1 $tensor2 -1.0]

# Zero alpha: [1, 2, 3] + 0.0 * [4, 5, 6] = [1, 2, 3] + [0, 0, 0] = [1, 2, 3]
set result [torch::tensor_add $tensor1 $tensor2 0.0]
```

### Different Data Types

```tcl
# Float32 tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float32 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]

# Float64 tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu]
set tensor2 [torch::tensor_create -data {4.0 5.0 6.0} -dtype float64 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]

# Integer tensors
set tensor1 [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
set tensor2 [torch::tensor_create -data {4 5 6} -dtype int32 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]
```

### Multi-dimensional Tensors

```tcl
# 2D tensors
set tensor1 [torch::zeros {2 2} float32 cpu]
set tensor2 [torch::ones {2 2} float32 cpu]
set result [torch::tensor_add $tensor1 $tensor2]

# 3D tensors
set tensor1 [torch::zeros {2 2 2} float32 cpu]
set tensor2 [torch::ones {2 2 2} float32 cpu]
set result [torch::tensor_add $tensor1 $tensor2]
```

### Edge Cases

```tcl
# Zero tensors
set tensor1 [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {0.0 0.0 0.0} -dtype float32 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]

# Large values
set tensor1 [torch::tensor_create -data {1000000.0 2000000.0 3000000.0} -dtype float32 -device cpu]
set tensor2 [torch::tensor_create -data {4000000.0 5000000.0 6000000.0} -dtype float32 -device cpu]
set result [torch::tensor_add $tensor1 $tensor2]
```

## Notes

- The addition is performed element-wise
- The result tensor has the same shape as the input tensors
- The alpha parameter scales the second tensor before addition
- Both `torch::tensor_add` and `torch::tensorAdd` are equivalent
- The function supports broadcasting if the tensors have compatible shapes
- The output tensor inherits the data type and device from the input tensors

## Error Handling

The command will return an error if:
- Either input tensor name is invalid or doesn't exist
- Required parameters are missing
- An invalid alpha value is provided
- Unknown parameters are provided
- The tensors have incompatible shapes for broadcasting

## Migration from Old Syntax

If you have existing code using the positional syntax, it will continue to work unchanged:

```tcl
# Old code (still works)
set result [torch::tensor_add $tensor1 $tensor2 2.0]

# New code (recommended)
set result [torch::tensor_add -input1 $tensor1 -input2 $tensor2 -alpha 2.0]
``` 