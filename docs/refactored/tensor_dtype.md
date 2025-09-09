# torch::tensor_dtype

Returns the data type of a tensor.

## Description

The `torch::tensor_dtype` command retrieves the data type (dtype) of the specified tensor. This is useful for checking the precision and type of tensor data, which is important for operations that require specific data types or for debugging tensor operations.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_dtype tensor
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_dtype -input tensor
```

### CamelCase Alias
```tcl
torch::tensorDtype -input tensor
```

## Parameters

| Parameter | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| input     | string | Yes      | Name of the input tensor           |

## Return Value

Returns a string containing the data type of the tensor (e.g., "Float", "Double", "Int", "Long", "Bool").

## Examples

### Basic Usage
```tcl
# Create tensors with different data types
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set b [torch::tensor_create -data {1 2 3} -dtype int32 -device cpu]
set c [torch::tensor_create -data {1 0 1} -dtype bool -device cpu]

# Using positional syntax
set dtype_a [torch::tensor_dtype $a]
set dtype_b [torch::tensor_dtype $b]
set dtype_c [torch::tensor_dtype $c]

# Using named parameter syntax
set dtype_a_named [torch::tensor_dtype -input $a]
set dtype_b_named [torch::tensor_dtype -input $b]

# Using camelCase alias
set dtype_a_camel [torch::tensorDtype -input $a]
set dtype_b_camel [torch::tensorDtype -input $b]
```

### Checking Data Types for Operations
```tcl
# Create a tensor
set tensor [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu]

# Check data type before operations
set dtype [torch::tensor_dtype $tensor]
if {$dtype == "Float" || $dtype == "Double"} {
    puts "Tensor is floating point - safe for division operations"
} else {
    puts "Tensor is not floating point - may need type conversion"
}
```

### Data Type Validation
```tcl
# Function to validate tensor data type
proc validate_float_tensor {tensor_name} {
    set dtype [torch::tensor_dtype $tensor_name]
    if {[string match "*Float*" $dtype] || [string match "*Double*" $dtype]} {
        return 1
    } else {
        error "Tensor must be floating point, got: $dtype"
    }
}

# Usage
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
validate_float_tensor $tensor
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_dtype invalid_tensor} result
# Returns: "Invalid tensor name"
```

### Missing Input Parameter
```tcl
catch {torch::tensor_dtype} result
# Returns: "Input tensor is required"
```

### Too Many Arguments
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_dtype $a extra} result
# Returns: "Invalid number of arguments"
```

### Unknown Parameter
```tcl
set a [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
catch {torch::tensor_dtype -input $a -unknown_param value} result
# Returns: "Unknown parameter: -unknown_param"
```

## Supported Data Types

The `tensor_dtype` command returns the following data type strings:

### Floating Point Types
- **"Float"** - 32-bit floating point (float32)
- **"Double"** - 64-bit floating point (float64)

### Integer Types
- **"Int"** - 32-bit integer (int32)
- **"Long"** - 64-bit integer (int64)

### Boolean Type
- **"Bool"** - Boolean values (true/false)

### Notes
- The exact string returned may vary slightly depending on the PyTorch version
- Complex data types are not currently supported by the tensor creation commands
- The data type is independent of the device (CPU/CUDA)

## Edge Cases

### Empty Tensor
```tcl
set a [torch::tensor_create -data {} -dtype float32 -device cpu]
set dtype [torch::tensor_dtype $a]
# Returns: "Float" (or equivalent)
```

### Single Element Tensor
```tcl
set a [torch::tensor_create -data {5.0} -dtype float32 -device cpu]
set dtype [torch::tensor_dtype $a]
# Returns: "Float" (or equivalent)
```

### Large Tensor
```tcl
set data [list]
for {set i 0} {$i < 10000} {incr i} {
    lappend data [expr {$i * 1.0}]
}
set a [torch::tensor_create -data $data -dtype float32 -device cpu]
set dtype [torch::tensor_dtype $a]
# Returns: "Float" (or equivalent)
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only)**:
```tcl
# Old way - still supported
set dtype [torch::tensor_dtype $tensor]
```

**New (Named Parameters)**:
```tcl
# New way - more explicit
set dtype [torch::tensor_dtype -input $tensor]
```

**CamelCase Alternative**:
```tcl
# Modern camelCase syntax
set dtype [torch::tensorDtype -input $tensor]
```

### Benefits of New Syntax
- **Explicit parameter names**: No confusion about parameter order
- **Better error messages**: Clear indication of missing parameters
- **Future extensibility**: Easy to add new parameters
- **Consistency**: Matches other refactored commands

## Related Commands

- [torch::tensor_device](tensor_device.md) - Get tensor device
- [torch::tensor_requires_grad](tensor_requires_grad.md) - Check if tensor requires gradients
- [torch::tensor_shape](tensor_shape.md) - Get tensor shape
- [torch::tensor_numel](tensor_numel.md) - Get number of elements in tensor
- [torch::tensor_to](tensor_to.md) - Move tensor to different device/dtype

## Notes

- The data type is a fundamental property of the tensor and cannot be changed without creating a new tensor
- Use `torch::tensor_to` to convert a tensor to a different data type
- The data type affects memory usage and numerical precision
- Floating point tensors are required for gradient computation
- Integer tensors are more memory efficient but have limited mathematical operations
- Boolean tensors are useful for logical operations and masking

---

**Migration Note**: 
- **Old:** `torch::tensor_dtype $t` (still supported)
- **New:** `torch::tensor_dtype -input $t` or `torch::tensorDtype -input $t`
- Both syntaxes are fully supported and produce identical results. 