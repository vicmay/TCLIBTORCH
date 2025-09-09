# torch::deg2rad

Convert degrees to radians element-wise.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::deg2rad input
```

### Named Parameter Syntax (Recommended)
```tcl
torch::deg2rad -input input
```

### CamelCase Alias
```tcl
torch::deg2Rad input
torch::deg2Rad -input input
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | tensor | Yes | Input tensor containing degrees |

**Note**: The `input` parameter can be specified either positionally or as a named parameter.

## Returns

Returns a new tensor with the same shape and dtype as the input tensor, where each element has been converted from degrees to radians using the formula: `radians = degrees × π / 180`

## Mathematical Background

The degree-to-radian conversion is a fundamental operation in trigonometry and mathematics:

- **Degrees**: Angular measurement where a full circle is 360°
- **Radians**: Angular measurement where a full circle is 2π radians
- **Conversion Formula**: `radians = degrees × π / 180`

Common conversions:
- 0° = 0 radians
- 30° = π/6 ≈ 0.5236 radians  
- 45° = π/4 ≈ 0.7854 radians
- 90° = π/2 ≈ 1.5708 radians
- 180° = π ≈ 3.1416 radians
- 270° = 3π/2 ≈ 4.7124 radians
- 360° = 2π ≈ 6.2832 radians

## Examples

### Basic Usage

#### Positional Syntax
```tcl
# Convert common angles from degrees to radians
set degrees [torch::tensor_create -data {0.0 30.0 45.0 90.0 180.0 270.0 360.0} -shape {7} -dtype float32]
set radians [torch::deg2rad $degrees]
puts [torch::tensor_data $radians]
# Output: 0 0.523599 0.785398 1.5708 3.14159 4.71239 6.28319
```

#### Named Parameter Syntax
```tcl
# Same conversion using named parameters
set degrees [torch::tensor_create -data {0.0 30.0 45.0 90.0 180.0 270.0 360.0} -shape {7} -dtype float32]
set radians [torch::deg2rad -input $degrees]
puts [torch::tensor_data $radians]
# Output: 0 0.523599 0.785398 1.5708 3.14159 4.71239 6.28319
```

#### CamelCase Alias
```tcl
# Using camelCase alias
set degrees [torch::tensor_create -data {45.0 90.0} -shape {2} -dtype float32]
set radians [torch::deg2Rad $degrees]
puts [torch::tensor_data $radians]
# Output: 0.785398 1.5708
```

### Multi-dimensional Tensors

```tcl
# Convert a 2D matrix of angles
set degree_matrix [torch::tensor_create -data {0.0 45.0 90.0 135.0 180.0 225.0} -shape {2 3} -dtype float32]
set radian_matrix [torch::deg2rad -input $degree_matrix]
puts "Shape: [torch::tensor_shape $radian_matrix]"
# Output: Shape: 2 3
puts "Data: [torch::tensor_data $radian_matrix]"
# Output: Data: 0 0.785398 1.5708 2.35619 3.14159 3.92699
```

### Different Data Types

```tcl
# Float32 tensors
set degrees_f32 [torch::tensor_create -data {30.0 60.0} -shape {2} -dtype float32]
set radians_f32 [torch::deg2rad $degrees_f32]
puts "Float32 dtype: [torch::tensor_dtype $radians_f32]"
# Output: Float32 dtype: Float

# Float64 tensors
set degrees_f64 [torch::tensor_create -data {30.0 60.0} -shape {2} -dtype float64]
set radians_f64 [torch::deg2rad -input $degrees_f64]
puts "Float64 dtype: [torch::tensor_dtype $radians_f64]"
# Output: Float64 dtype: Double
```

### Edge Cases

#### Negative Angles
```tcl
# Negative degree values
set negative_degrees [torch::tensor_create -data {-90.0 -45.0 -180.0} -shape {3} -dtype float32]
set negative_radians [torch::deg2rad $negative_degrees]
puts [torch::tensor_data $negative_radians]
# Output: -1.5708 -0.785398 -3.14159
```

#### Large Angles
```tcl
# Angles greater than 360 degrees
set large_degrees [torch::tensor_create -data {450.0 720.0} -shape {2} -dtype float32]
set large_radians [torch::deg2rad -input $large_degrees]
puts [torch::tensor_data $large_radians]
# Output: 7.85398 12.5664
```

#### Scalar Tensors
```tcl
# Single value (scalar) tensor
set scalar_degrees [torch::tensor_create -data {90.0} -shape {} -dtype float32]
set scalar_radians [torch::deg2rad $scalar_degrees]
puts [torch::tensor_data $scalar_radians]
# Output: 1.5708
```

## Error Handling

The command performs comprehensive validation and provides clear error messages:

### Invalid Tensor Name
```tcl
catch {torch::deg2rad invalid_tensor} msg
puts $msg
# Output: Invalid tensor name
```

### Missing Parameters
```tcl
catch {torch::deg2rad} msg
puts $msg
# Output: Usage: torch::deg2rad tensor | torch::deg2rad -input tensor
```

### Invalid Parameter Names
```tcl
set tensor [torch::tensor_create -data {45.0} -shape {1} -dtype float32]
catch {torch::deg2rad -invalid $tensor} msg
puts $msg
# Output: Unknown parameter: -invalid
```

### Missing Parameter Values
```tcl
catch {torch::deg2rad -input} msg
puts $msg
# Output: Missing value for parameter
```

## Performance Notes

- **In-place Operation**: This is not an in-place operation; a new tensor is created
- **Memory Usage**: Memory usage scales linearly with tensor size
- **Numerical Precision**: Results maintain the precision of the input tensor's data type
- **Broadcasting**: Not applicable (unary operation)

## Technical Notes

### Data Type Preservation
The output tensor maintains the same data type as the input tensor:
- `float32` input → `float32` output (reported as "Float")
- `float64` input → `float64` output (reported as "Double")

### Shape Preservation
The output tensor has exactly the same shape as the input tensor, regardless of dimensionality.

### Numerical Stability
The conversion uses the mathematical constant π with high precision, ensuring accurate results across the full range of floating-point values.

## Migration Guide

### From Positional to Named Parameters

**Old Style (Positional)**:
```tcl
set result [torch::deg2rad $input_tensor]
```

**New Style (Named Parameters)**:
```tcl
set result [torch::deg2rad -input $input_tensor]
```

**Benefits of Named Parameters**:
- **Clarity**: Parameter names make code self-documenting
- **Flexibility**: Parameters can be specified in any order
- **Maintainability**: Easier to modify and extend
- **Consistency**: Matches modern API design patterns

## Related Commands

- [`torch::rad2deg`](rad2deg.md) - Convert radians to degrees
- [`torch::sin`](sin.md) - Sine function (expects radians)
- [`torch::cos`](cos.md) - Cosine function (expects radians)
- [`torch::tan`](tan.md) - Tangent function (expects radians)

## See Also

- [PyTorch deg2rad Documentation](https://pytorch.org/docs/stable/generated/torch.deg2rad.html)
- [Mathematical Functions](../mathematical_functions.md)
- [Tensor Operations](../tensor_operations.md) 