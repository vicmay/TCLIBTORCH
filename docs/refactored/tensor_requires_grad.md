# torch::tensor_requires_grad

Returns whether a tensor requires gradients for automatic differentiation.

## Description

The `torch::tensor_requires_grad` command checks if the specified tensor has gradients enabled for automatic differentiation. This is essential for deep learning workflows where you need to track gradients for backpropagation and optimization.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_requires_grad tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_requires_grad -input tensor
```

### CamelCase Alias
```tcl
torch::tensorRequiresGrad -input tensor
```

## Parameters

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| input     | string | Yes      | Name of the input tensor           |

## Return Value

Returns `1` if the tensor requires gradients, `0` if it doesn't.

## Examples

### Basic Usage
```tcl
# Create tensors with different gradient settings
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
set b [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]

# Using positional syntax
set grad_a [torch::tensor_requires_grad $a]
set grad_b [torch::tensor_requires_grad $b]

# Using named parameter syntax
set grad_a_named [torch::tensor_requires_grad -input $a]
set grad_b_named [torch::tensor_requires_grad -input $b]

# Using camelCase alias
set grad_a_camel [torch::tensorRequiresGrad -input $a]
set grad_b_camel [torch::tensorRequiresGrad -input $b]

puts "Tensor a requires gradients: $grad_a"
puts "Tensor b requires gradients: $grad_b"
```

### Gradient Management
```tcl
# Check gradient status before operations
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad true]
set requires_grad [torch::tensor_requires_grad $tensor]

if {$requires_grad} {
    puts "Tensor will track gradients for backpropagation"
} else {
    puts "Tensor will not track gradients"
}
```

### Training vs Inference Mode
```tcl
# Function to validate training tensor
proc validate_training_tensor {tensor_name} {
    set requires_grad [torch::tensor_requires_grad $tensor_name]
    if {$requires_grad} {
        return 1
    } else {
        error "Tensor must require gradients for training, got: $requires_grad"
    }
}

# Function to validate inference tensor
proc validate_inference_tensor {tensor_name} {
    set requires_grad [torch::tensor_requires_grad $tensor_name]
    if {!$requires_grad} {
        return 1
    } else {
        error "Tensor should not require gradients for inference, got: $requires_grad"
    }
}

# Usage
set training_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
set inference_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]

validate_training_tensor $training_tensor
validate_inference_tensor $inference_tensor
```

### Memory Optimization
```tcl
# Check gradient status to optimize memory usage
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
set requires_grad [torch::tensor_requires_grad $tensor]

if {$requires_grad} {
    puts "Tensor uses extra memory for gradient storage"
    # Consider disabling gradients for inference to save memory
} else {
    puts "Tensor uses minimal memory (no gradient storage)"
}
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_requires_grad invalid_tensor} result
# Returns: "Invalid tensor name"
```

### Missing Input Parameter
```tcl
catch {torch::tensor_requires_grad} result
# Returns: "Input tensor is required"
```

### Too Many Arguments
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
catch {torch::tensor_requires_grad $a extra} result
# Returns: "Invalid number of arguments"
```

### Unknown Parameter
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]
catch {torch::tensor_requires_grad -input $a -unknown_param value} result
# Returns: "Unknown parameter: -unknown_param"
```

## Supported Data Types

The `tensor_requires_grad` command works with all tensor data types, but only floating point and complex tensors can actually require gradients:

### Supported for Gradients
- **float32** - 32-bit floating point
- **float64** - 64-bit floating point
- **complex64** - 64-bit complex
- **complex128** - 128-bit complex

### Not Supported for Gradients
- **int32** - 32-bit integer
- **int64** - 64-bit integer
- **bool** - Boolean
- **uint8** - 8-bit unsigned integer

### Notes
- Integer and boolean tensors will always return `0` for `requires_grad`
- Attempting to create integer/boolean tensors with `requiresGrad true` will cause an error
- Only floating point and complex tensors can participate in automatic differentiation

## Edge Cases

### Empty Tensor
```tcl
set a [torch::tensor_create -data {} -dtype float32 -device cpu -requiresGrad true]
set result [torch::tensor_requires_grad $a]
# Returns: 1 (if requiresGrad was true)
```

### Single Element Tensor
```tcl
set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu -requiresGrad false]
set result [torch::tensor_requires_grad $a]
# Returns: 0 (if requiresGrad was false)
```

### Large Tensor
```tcl
set data [list]
for {set i 0} {$i < 10000} {incr i} {
    lappend data [expr {$i * 1.0}]
}
set a [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad true]
set result [torch::tensor_requires_grad $a]
# Returns: 1 (if requiresGrad was true)
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only)**:
```tcl
# Old way - still supported
set requires_grad [torch::tensor_requires_grad $tensor]
```

**New (Named Parameters)**:
```tcl
# New way - more explicit
set requires_grad [torch::tensor_requires_grad -input $tensor]
```

**CamelCase Alternative**:
```tcl
# Modern camelCase syntax
set requires_grad [torch::tensorRequiresGrad -input $tensor]
```

### Benefits of New Syntax
- **Explicit parameter names**: No confusion about parameter order
- **Better error messages**: Clear indication of missing parameters
- **Future extensibility**: Easy to add new parameters
- **Consistency**: Matches other refactored commands

## Related Commands

- [torch::tensor_dtype](tensor_dtype.md) - Get tensor data type
- [torch::tensor_device](tensor_device.md) - Get tensor device
- [torch::tensor_grad](tensor_grad.md) - Get tensor gradient
- [torch::tensor_backward](tensor_backward.md) - Compute gradients
- [torch::tensor_to](tensor_to.md) - Move tensor to different device/dtype

## Performance Considerations

### Memory Usage
- **With gradients**: Tensors use additional memory to store gradient information
- **Without gradients**: Tensors use minimal memory (no gradient storage)
- **Memory optimization**: Disable gradients for inference to save memory

### Computation Overhead
- **With gradients**: Operations track computational graph for backpropagation
- **Without gradients**: Operations are faster (no graph tracking)
- **Training vs inference**: Use gradients only when needed

### Best Practices
```tcl
# Training mode - enable gradients
set training_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad true]

# Inference mode - disable gradients for efficiency
set inference_tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]

# Check gradient status
set training_status [torch::tensor_requires_grad $training_tensor]
set inference_status [torch::tensor_requires_grad $inference_tensor]

puts "Training tensor requires gradients: $training_status"
puts "Inference tensor requires gradients: $inference_status"
```

### Memory Management
- Use `requiresGrad false` for tensors that don't need gradients
- Disable gradients for inference to reduce memory usage
- Monitor gradient status in large models to optimize memory

## Notes

- The `requires_grad` property is fundamental for automatic differentiation
- Only floating point and complex tensors can require gradients
- Integer and boolean tensors will always return `0` for `requires_grad`
- Use `torch::tensor_to` to change the `requires_grad` property of existing tensors
- Gradient status affects both memory usage and computation speed
- Always check gradient status when optimizing for memory or performance
- The gradient status is essential for debugging training vs inference issues

---

**Migration Note**: 
- **Old:** `torch::tensor_requires_grad $t` (still supported)
- **New:** `torch::tensor_requires_grad -input $t` or `torch::tensorRequiresGrad -input $t`
- Both syntaxes are fully supported and produce identical results. 