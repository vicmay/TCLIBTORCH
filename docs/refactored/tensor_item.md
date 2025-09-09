# torch::tensor_item / torch::tensorItem

Extract a scalar value from a single-element tensor.

## Description

The `tensor_item` command extracts the scalar value from a tensor that contains exactly one element. This is useful for converting tensor results back to scalar values for further processing or output.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::tensor_item tensor_name
```

### Named Parameter Syntax (New)
```tcl
torch::tensor_item -tensor tensor_name
torch::tensor_item -input tensor_name
```

### CamelCase Alias
```tcl
torch::tensorItem tensor_name
torch::tensorItem -tensor tensor_name
torch::tensorItem -input tensor_name
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor_name` | string | Yes | Name of the single-element tensor |
| `-tensor` | string | Yes* | Alternative parameter name for tensor |
| `-input` | string | Yes* | Alternative parameter name for tensor |

*Required when using named parameter syntax

## Return Value

Returns the scalar value as a string. The format depends on the tensor's data type:
- **Float32/Float64**: Returns decimal representation (e.g., "5.000000", "3.141590")
- **Int32/Int64**: Returns integer representation (e.g., "42", "123456789")
- **Bool**: Returns "1" for true, "0" for false
- **Other types**: Converted to double precision

## Examples

### Basic Usage

```tcl
# Create a simple scalar tensor
set tensor [torch::tensor_create -data 5.0 -dtype float32]

# Extract value using positional syntax
set value [torch::tensor_item $tensor]
puts "Value: $value"  # Output: Value: 5.000000

# Extract value using named syntax
set value [torch::tensor_item -tensor $tensor]
puts "Value: $value"  # Output: Value: 5.000000

# Extract value using camelCase alias
set value [torch::tensorItem $tensor]
puts "Value: $value"  # Output: Value: 5.000000
```

### Different Data Types

```tcl
# Integer tensor
set int_tensor [torch::tensor_create -data 42 -dtype int32]
set value [torch::tensor_item $int_tensor]
puts "Integer value: $value"  # Output: Integer value: 42

# Double precision tensor
set double_tensor [torch::tensor_create -data 3.14159 -dtype float64]
set value [torch::tensor_item $double_tensor]
puts "Double value: $value"  # Output: Double value: 3.141590

# Large integer
set large_tensor [torch::tensor_create -data 123456789 -dtype int64]
set value [torch::tensor_item $large_tensor]
puts "Large value: $value"  # Output: Large value: 123456789
```

### Negative Values and Edge Cases

```tcl
# Negative float
set neg_tensor [torch::tensor_create -data -2.5 -dtype float32]
set value [torch::tensor_item $neg_tensor]
puts "Negative value: $value"  # Output: Negative value: -2.500000

# Zero
set zero_tensor [torch::tensor_create -data 0.0 -dtype float32]
set value [torch::tensor_item $zero_tensor]
puts "Zero value: $value"  # Output: Zero value: 0.000000

# Very small number
set small_tensor [torch::tensor_create -data 0.000001 -dtype float32]
set value [torch::tensor_item $small_tensor]
puts "Small value: $value"  # Output: Small value: 0.000001
```

### After Tensor Operations

```tcl
# Create tensor and sum it
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
set summed [torch::tensor_sum -input $tensor]
set value [torch::tensor_item $summed]
puts "Sum value: $value"  # Output: Sum value: 6.000000

# Create tensor and reshape to single element
set tensor [torch::tensor_create -data {5.0} -dtype float32]
set reshaped [torch::tensor_reshape -input $tensor -shape {1}]
set value [torch::tensor_item $reshaped]
puts "Reshaped value: $value"  # Output: Reshaped value: 5.000000
```

### Different Devices

```tcl
# CPU tensor
set cpu_tensor [torch::tensor_create -data 5.0 -dtype float32 -device cpu]
set value [torch::tensor_item $cpu_tensor]
puts "CPU value: $value"  # Output: CPU value: 5.000000

# CUDA tensor (if available)
if {[torch::cuda_is_available]} {
    set cuda_tensor [torch::tensor_create -data 5.0 -dtype float32 -device cuda]
    set value [torch::tensor_item $cuda_tensor]
    puts "CUDA value: $value"  # Output: CUDA value: 5.000000
}
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::tensor_item invalid_tensor} result
puts $result  # Output: Invalid tensor name: invalid_tensor
```

### Missing Parameters
```tcl
# Missing tensor name
catch {torch::tensor_item} result
puts $result  # Output: Required parameter missing: tensor

# Missing parameter value
catch {torch::tensor_item -tensor} result
puts $result  # Output: Missing value for parameter
```

### Unknown Parameters
```tcl
set tensor [torch::tensor_create -data 5.0 -dtype float32]
catch {torch::tensor_item -unknown $tensor} result
puts $result  # Output: Unknown parameter: -unknown
```

### Too Many Arguments
```tcl
set tensor [torch::tensor_create -data 5.0 -dtype float32]
catch {torch::tensor_item $tensor extra_arg} result
puts $result  # Output: Usage: torch::tensor_item tensor
```

### Tensor with Multiple Elements
```tcl
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32]
catch {torch::tensor_item $tensor} result
puts $result  # Output: Tensor must have exactly one element
```

### Empty Tensor
```tcl
set tensor [torch::tensor_create -data {} -dtype float32]
catch {torch::tensor_item $tensor} result
puts $result  # Output: Tensor must have exactly one element
```

## Migration Guide

### From Old Syntax to New Syntax

**Old (Positional Only):**
```tcl
set value [torch::tensor_item $tensor]
```

**New (Named Parameters):**
```tcl
set value [torch::tensor_item -tensor $tensor]
# or
set value [torch::tensor_item -input $tensor]
```

**New (CamelCase Alias):**
```tcl
set value [torch::tensorItem $tensor]
set value [torch::tensorItem -tensor $tensor]
```

### Backward Compatibility

The old positional syntax continues to work:
```tcl
# This still works
set value [torch::tensor_item $tensor]

# This also works
set value [torch::tensor_item -tensor $tensor]
```

## Notes

1. **Single Element Requirement**: The tensor must contain exactly one element. Tensors with zero or multiple elements will cause an error.

2. **Data Type Handling**: The command automatically handles different data types and converts them to appropriate string representations.

3. **Precision**: Float values are displayed with 6 decimal places by default.

4. **Device Independence**: Works with tensors on any device (CPU/CUDA).

5. **Memory Efficiency**: This operation is very lightweight and doesn't copy tensor data.

6. **Common Use Cases**: 
   - Extracting loss values from training
   - Getting scalar results from reduction operations
   - Converting tensor outputs to scalar values for logging or display

## Related Commands

- `torch::tensor_create` - Create tensors
- `torch::tensor_sum` - Sum tensor elements (often results in single element)
- `torch::tensor_mean` - Mean of tensor elements (often results in single element)
- `torch::tensor_shape` - Get tensor shape
- `torch::tensor_numel` - Get total number of elements
- `torch::tensor_dtype` - Get tensor data type 