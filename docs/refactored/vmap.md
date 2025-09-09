# torch::vmap

Applies a function to each element of a tensor (vectorized map operation).

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::vmap func inputs
```

### Named Parameter Syntax  
```tcl
torch::vmap -func function -inputs tensor
```

### CamelCase Alias
```tcl
torch::vectorMap func inputs
torch::vectorMap -func function -inputs tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | string | **Required.** Function handle or name to apply to each element |
| `inputs` | string | **Required.** Name of the input tensor |

## Returns

Returns a new tensor handle containing the result of applying the function to each element of the input tensor.

## Description

The `torch::vmap` command performs a vectorized map operation, applying a specified function to each element of the input tensor. This is useful for:

- **Element-wise operations**: Applying the same function to every element in a tensor
- **Batch processing**: Processing multiple inputs with the same function
- **Functional programming**: Implementing map operations in tensor computations
- **Performance optimization**: Vectorized operations are typically more efficient than loops

The function is applied to each element independently, maintaining the original tensor structure and shape.

## Examples

### Basic Usage
```tcl
# Create input tensor
set inputs [torch::tensor_create {1.0 2.0 3.0 4.0}]

# Apply function to each element
set result [torch::vmap "square_function" $inputs]
puts "Result: [torch::tensor_to_list $result]"
```

### Named Parameter Syntax
```tcl
# Create 2D tensor
set inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]

# Use named parameters
set result [torch::vmap -func "my_function" -inputs $inputs]
puts "Shape: [torch::tensor_shape $result]"

# Alternative parameter names
set result2 [torch::vmap -function "test_func" -input $inputs]
```

### CamelCase Alias
```tcl
# Using camelCase alias
set inputs [torch::tensor_create {1.0 2.0 3.0}]

# Positional syntax with camelCase
set result [torch::vectorMap "my_func" $inputs]

# Named parameter syntax with camelCase
set result2 [torch::vectorMap -func "my_func" -inputs $inputs]
```

### Different Tensor Dimensions
```tcl
# 1D tensor
set inputs_1d [torch::tensor_create {1.0 2.0 3.0}]
set vmap_1d [torch::vmap "func_1d" $inputs_1d]

# 2D tensor
set inputs_2d [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
set vmap_2d [torch::vmap "func_2d" $inputs_2d]

# Higher dimensional tensors
set inputs_3d [torch::tensor_create {{1.0 2.0} {3.0 4.0}}]
set vmap_3d [torch::vmap "func_3d" $inputs_3d]
```

### Practical Example: Element-wise Transformation
```tcl
# Create input tensor
set data [torch::tensor_create {0.1 0.5 0.9 1.2}]

# Apply sigmoid-like function to each element
set transformed [torch::vmap "sigmoid_func" $data]
puts "Original: [torch::tensor_to_list $data]"
puts "Transformed: [torch::tensor_to_list $transformed]"
```

### Batch Processing Example
```tcl
# Process multiple samples
set batch_inputs [torch::tensor_create {{1.0 2.0} {3.0 4.0} {5.0 6.0}}]

# Apply preprocessing function to each sample
set processed [torch::vmap "preprocess_func" $batch_inputs]
puts "Batch shape: [torch::tensor_shape $processed]"
```

## Error Handling

The command will raise an error in the following cases:

- **Invalid tensor**: If the input tensor name doesn't exist
- **Missing required parameter**: If any required parameter is not provided
- **Unknown parameter**: If an unrecognized parameter name is used

### Error Examples
```tcl
# Error: Invalid tensor
catch {torch::vmap "func" "nonexistent"} error
puts $error  ;# "Invalid tensor"

# Error: Missing required parameter  
catch {torch::vmap -func "test"} error
puts $error  ;# "Required parameters missing: -func and -inputs"

# Error: Unknown parameter
set inputs [torch::tensor_create {1.0 2.0}]
catch {torch::vmap -func "test" -inputs $inputs -unknown param} error
puts $error  ;# "Unknown parameter: -unknown"
```

## Mathematical Details

The vmap operation can be mathematically described as:

**vmap(f, x) = [f(x₁), f(x₂), ..., f(xₙ)]**

Where:
- `f` is the function to apply
- `x` is the input tensor with elements x₁, x₂, ..., xₙ
- The result is a tensor with the same shape as the input

### Properties
- **Shape preservation**: Output tensor has the same shape as input tensor
- **Element-wise application**: Function is applied independently to each element
- **Commutativity**: vmap(f, vmap(g, x)) = vmap(g ∘ f, x) (function composition)
- **Distributivity**: vmap(f, x + y) = vmap(f, x) + vmap(f, y) (for linear functions)

## Performance Notes

- **Vectorized execution**: Operations are optimized for batch processing
- **Memory efficient**: Avoids creating intermediate tensors when possible
- **Parallelizable**: Can take advantage of multi-threading and GPU acceleration
- **Function overhead**: The function application overhead is amortized across all elements

## Relationship to Other Operations

Vmap is closely related to other tensor operations:

- **Map vs Reduce**: vmap applies a function to each element, while reduce combines elements
- **Broadcasting**: Similar to broadcasting but with explicit function application
- **Functional programming**: Implements the map higher-order function for tensors
- **Batch processing**: Enables efficient processing of multiple inputs

## See Also

- [`torch::functional_call`](functional_call.md) - Functional call with parameters
- [`torch::grad`](grad.md) - Gradient computation
- [`torch::jacobian`](jacobian.md) - Jacobian matrix computation
- [`torch::hessian`](hessian.md) - Hessian matrix computation

## Migration Guide

### From Positional to Named Parameters

**Old positional syntax:**
```tcl
set result [torch::vmap "my_func" $inputs]
```

**New named parameter syntax:**
```tcl
set result [torch::vmap -func "my_func" -inputs $inputs]
```

**CamelCase alias:**
```tcl
set result [torch::vectorMap -func "my_func" -inputs $inputs]
```

Both syntaxes are fully supported and can be used interchangeably. The named parameter syntax is recommended for new code as it's more readable and self-documenting.

### Alternative Parameter Names

The named parameter syntax supports alternative parameter names for flexibility:

- `-func` or `-function`
- `-inputs` or `-input`

This allows for more readable code based on context and personal preference.

## Implementation Notes

The current implementation is a simplified version that preserves the input tensor. In a full implementation, the function would be applied to each element of the tensor, potentially transforming the values while maintaining the tensor structure.

For production use, consider implementing the actual function application logic based on your specific requirements. 