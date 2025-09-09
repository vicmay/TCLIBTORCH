# torch::frac / torch::Frac

Compute the fractional part of each element in a tensor.

## Description

The `frac` command computes the fractional part of each element in the input tensor. For any real number `x`, the fractional part is defined as `frac(x) = x - trunc(x)`, where `trunc(x)` is the integer part of `x` obtained by truncating towards zero.

This operation preserves the sign of the input values and is useful in various mathematical computations, signal processing, and numerical algorithms.

## Mathematical Foundation

### Definition
For any real number `x`:
- `frac(x) = x - trunc(x)`
- `trunc(x)` is the integer part obtained by truncating towards zero

### Properties
- **Range**: For any `x`, `-1 < frac(x) < 1`
- **Sign preservation**: `frac(x)` has the same sign as `x` (except for exact integers)
- **Periodicity**: `frac(x + n) = frac(x)` for any integer `n`
- **Identity**: `x = trunc(x) + frac(x)`

### Examples
- `frac(2.3) = 0.3`
- `frac(-2.3) = -0.3`
- `frac(5.0) = 0.0`
- `frac(-5.0) = 0.0`
- `frac(0.7) = 0.7`
- `frac(-0.7) = -0.7`

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::frac input_tensor
```

### Named Parameter Syntax (New)
```tcl
torch::frac -input input_tensor
torch::frac -tensor input_tensor
```

### CamelCase Alias
```tcl
torch::Frac input_tensor
torch::Frac -input input_tensor
torch::Frac -tensor input_tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_tensor` | string | Yes | Name of the input tensor |
| `-input` | string | Yes* | Alternative parameter name for input tensor |
| `-tensor` | string | Yes* | Alternative parameter name for input tensor |

*Required when using named parameter syntax

## Return Value

Returns a new tensor handle containing the fractional parts of all input elements. The output tensor has the same shape, dtype, and device as the input tensor.

## Supported Data Types

- **Float32** ✅ - Single precision floating point
- **Float64** ✅ - Double precision floating point  
- **Int32** ❌ - Not supported (use float conversion first)
- **Int64** ❌ - Not supported (use float conversion first)
- **Bool** ❌ - Not supported (use float conversion first)

**Note**: Integer and boolean tensors must be converted to floating-point types before applying the frac operation.

## Examples

### Basic Usage

```tcl
# Create a tensor with mixed positive and negative values
set input [torch::tensor_create -data {2.3 -1.7 3.5 -0.5 0.8} -dtype float32]

# Compute fractional parts using positional syntax
set result [torch::frac $input]
torch::tensor_print $result
# Output: [ 0.3000, -0.7000,  0.5000, -0.5000,  0.8000]

# Compute fractional parts using named syntax
set result [torch::frac -input $input]
torch::tensor_print $result
# Output: [ 0.3000, -0.7000,  0.5000, -0.5000,  0.8000]

# Compute fractional parts using camelCase alias
set result [torch::Frac -input $input]
torch::tensor_print $result
# Output: [ 0.3000, -0.7000,  0.5000, -0.5000,  0.8000]
```

### Multi-dimensional Tensors

```tcl
# Create a 2D tensor
set matrix [torch::tensor_create -data {{1.2 -2.3} {3.7 -4.1}} -dtype float32]

# Apply frac operation
set frac_matrix [torch::frac $matrix]
torch::tensor_print $frac_matrix
# Output: [[ 0.2000, -0.3000],
#          [ 0.7000, -0.1000]]

# Verify shape is preserved
set original_shape [torch::tensor_shape $matrix]
set result_shape [torch::tensor_shape $frac_matrix]
puts "Original shape: $original_shape"  # Output: 2 2
puts "Result shape: $result_shape"      # Output: 2 2
```

### Edge Cases

```tcl
# Exact integers
set integers [torch::tensor_create -data {1.0 2.0 -3.0 -4.0} -dtype float32]
set int_frac [torch::frac $integers]
torch::tensor_print $int_frac
# Output: [ 0.0000,  0.0000,  0.0000,  0.0000]

# Very small values
set small_values [torch::tensor_create -data {0.001 -0.001 0.0001} -dtype float32]
set small_frac [torch::frac $small_values]
torch::tensor_print $small_frac
# Output: [ 0.0010, -0.0010,  0.0001]

# Large values
set large_values [torch::tensor_create -data {1000.3 -2000.7} -dtype float32]
set large_frac [torch::frac $large_values]
torch::tensor_print $large_frac
# Output: [ 0.3000, -0.7000]
```

### Mathematical Verification

```tcl
# Verify the identity: x = trunc(x) + frac(x)
set input [torch::tensor_create -data {2.3 -1.7 3.5} -dtype float32]
set frac_part [torch::frac $input]
set trunc_part [torch::trunc $input]
set reconstructed [torch::add $frac_part $trunc_part]

torch::tensor_print $input
# Output: [ 2.3000, -1.7000,  3.5000]
torch::tensor_print $reconstructed
# Output: [ 2.3000, -1.7000,  3.5000]
```

### Different Data Types

```tcl
# Float32 (default)
set f32_tensor [torch::tensor_create -data {1.5 2.7} -dtype float32]
set f32_result [torch::frac $f32_tensor]

# Float64 for higher precision
set f64_tensor [torch::tensor_create -data {1.5 2.7} -dtype float64]
set f64_result [torch::frac $f64_tensor]

# Both produce similar results but with different precision
torch::tensor_print $f32_result
# Output: [ 0.5000,  0.7000]
torch::tensor_print $f64_result  
# Output: [ 0.5000,  0.7000]
```

## Error Handling

The command provides comprehensive error handling for various invalid inputs:

### Missing Arguments
```tcl
# Error: Missing required argument
set code [catch {torch::frac} msg]
puts $msg
# Output: Usage: torch::frac input_tensor
#            or: torch::frac -input TENSOR
```

### Invalid Tensor Names
```tcl
# Error: Tensor doesn't exist
set code [catch {torch::frac nonexistent_tensor} msg]
puts $msg
# Output: Invalid tensor name
```

### Unknown Parameters
```tcl
set input [torch::tensor_create -data {1.5} -dtype float32]

# Error: Invalid parameter name
set code [catch {torch::frac -invalid_param $input} msg]
puts $msg
# Output: Unknown parameter: -invalid_param
```

### Missing Parameter Values
```tcl
# Error: Parameter without value
set code [catch {torch::frac -input} msg]
puts $msg
# Output: Missing value for parameter
```

### Too Many Arguments
```tcl
set input [torch::tensor_create -data {1.5} -dtype float32]

# Error: Extra arguments
set code [catch {torch::frac $input extra_arg} msg]
puts $msg
# Output: Usage: torch::frac input_tensor
```

### Data Type Limitations
```tcl
# Error: Integer tensors not supported
set int_tensor [torch::tensor_create -data {1 2 3} -dtype int32]
set code [catch {torch::frac $int_tensor} msg]
puts $msg
# Output: "frac_cpu" not implemented for 'Int'

# Solution: Convert to float first
set float_tensor [torch::tensor_to $int_tensor -dtype float32]
set result [torch::frac $float_tensor]  # This works
```

## Performance Considerations

### Computational Complexity
- **Time Complexity**: O(n) where n is the number of elements
- **Space Complexity**: O(n) for the output tensor
- **Memory Efficiency**: Input tensor is not modified (immutable operation)

### Optimization Tips
```tcl
# For large tensors, consider processing in chunks
proc process_large_tensor {tensor_name chunk_size} {
    set total_elements [torch::tensor_numel $tensor_name]
    set results {}
    
    for {set i 0} {$i < $total_elements} {set i [expr {$i + $chunk_size}]} {
        set end [expr {min($i + $chunk_size, $total_elements)}]
        set chunk [torch::tensor_slice $tensor_name 0 $i $end]
        set chunk_result [torch::frac $chunk]
        lappend results $chunk_result
    }
    
    return [torch::tensor_cat $results 0]
}
```

### Device Considerations
```tcl
# CPU tensors (default)
set cpu_tensor [torch::tensor_create -data {1.5 2.7} -dtype float32 -device cpu]
set cpu_result [torch::frac $cpu_tensor]

# GPU tensors (if available)
if {[torch::cuda_is_available]} {
    set gpu_tensor [torch::tensor_to $cpu_tensor -device cuda]
    set gpu_result [torch::frac $gpu_tensor]  # Computed on GPU
}
```

## Integration Examples

### With Other Mathematical Operations
```tcl
# Chain operations
set input [torch::tensor_create -data {1.2 2.3 3.4} -dtype float32]
set frac_result [torch::frac $input]
set abs_frac [torch::abs $frac_result]
set squared_frac [torch::mul $frac_result $frac_result]

# Complex mathematical expressions
set complex_result [torch::add [torch::frac $input] [torch::sin $input]]
```

### Signal Processing Example
```tcl
# Extract fractional part for phase analysis
proc normalize_phase {phase_tensor} {
    # Normalize phase to [-π, π) range
    set scaled [torch::div $phase_tensor [torch::tensor_create -data 6.28318530718 -dtype float32]]
    set frac_part [torch::frac $scaled]
    return [torch::mul $frac_part [torch::tensor_create -data 6.28318530718 -dtype float32]]
}
```

### Numerical Algorithms
```tcl
# Fixed-point iteration with fractional feedback
proc iterate_with_frac {x iterations} {
    set current $x
    for {set i 0} {$i < $iterations} {incr i} {
        set next [torch::add $current [torch::frac $current]]
        set current [torch::mul $next [torch::tensor_create -data 0.5 -dtype float32]]
    }
    return $current
}
```

## Migration Guide

### From Legacy Syntax to Modern Syntax

**Legacy Positional Syntax:**
```tcl
# Old way (still supported)
set result [torch::frac $input_tensor]
```

**Modern Named Parameter Syntax:**
```tcl
# New way (recommended)
set result [torch::frac -input $input_tensor]

# Alternative parameter name
set result [torch::frac -tensor $input_tensor]

# CamelCase alias
set result [torch::Frac -input $input_tensor]
```

### Benefits of Modern Syntax

1. **Self-documenting**: Parameter names make code more readable
2. **Flexible**: Parameter order doesn't matter
3. **Extensible**: Easy to add new parameters in future versions
4. **Consistent**: Matches modern API patterns across the library

### Compatibility Notes

- **Backward Compatibility**: All existing code using positional syntax continues to work
- **Mixed Usage**: You can use both syntaxes in the same project
- **Parameter Names**: `-input` and `-tensor` are equivalent and interchangeable

## Related Commands

- **torch::trunc** - Extract integer part (truncate towards zero)
- **torch::floor** - Round down to nearest integer
- **torch::ceil** - Round up to nearest integer
- **torch::round** - Round to nearest integer
- **torch::modf** - Split into integer and fractional parts (planned)

## Mathematical Relationships

```tcl
# Fundamental identity
# x = trunc(x) + frac(x)
set x [torch::tensor_create -data {2.7} -dtype float32]
set trunc_x [torch::trunc $x]     # Result: 2.0
set frac_x [torch::frac $x]       # Result: 0.7
set sum [torch::add $trunc_x $frac_x]  # Result: 2.7 (equals x)

# Relationship with floor (for positive numbers)
# For x >= 0: frac(x) = x - floor(x)
set pos_x [torch::tensor_create -data {2.7} -dtype float32]
set floor_x [torch::floor $pos_x]
set diff [torch::sub $pos_x $floor_x]  # Same as frac(x) for positive x

# For negative numbers, floor and trunc differ
set neg_x [torch::tensor_create -data {-2.7} -dtype float32]
set frac_neg [torch::frac $neg_x]      # Result: -0.7
set floor_neg [torch::floor $neg_x]    # Result: -3.0
set trunc_neg [torch::trunc $neg_x]    # Result: -2.0
```

## See Also

- [torch::trunc](trunc.md) - Integer part extraction
- [torch::floor](floor.md) - Floor function
- [torch::ceil](ceil.md) - Ceiling function
- [torch::round](round.md) - Rounding function
- [Tensor Operations Guide](../tensor_operations.md) - Complete tensor operations reference 