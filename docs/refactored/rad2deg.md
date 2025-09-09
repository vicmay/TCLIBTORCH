# torch::rad2deg

Converts angles from radians to degrees.

## Syntax

```tcl
# Positional syntax (backward compatible)
torch::rad2deg tensor

# Named parameter syntax
torch::rad2deg -input tensor

# camelCase alias
torch::radToDeg -input tensor
```

## Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| tensor / -input | tensor | Input tensor containing angles in radians |

## Return Value

Returns a new tensor with the angles converted from radians to degrees.

## Description

The `rad2deg` command converts angles from radians to degrees element-wise. The conversion formula used is:
```
degrees = radians * (180/π)
```

The command preserves the input tensor's shape, device, and data type. The operation is element-wise and supports broadcasting.

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create a tensor with angles in radians
set angles [torch::tensor_create {0.0 1.57079633 3.14159265} float32]

# Convert to degrees
set degrees [torch::rad2deg $angles]
# Result: tensor([0.0, 90.0, 180.0])
```

### Using Named Parameters
```tcl
# Create a tensor with angles in radians
set angles [torch::tensor_create {0.0 1.57079633 3.14159265} float32]

# Convert to degrees using named parameters
set degrees [torch::rad2deg -input $angles]
# Result: tensor([0.0, 90.0, 180.0])
```

### Using camelCase Alias
```tcl
# Create a tensor with angles in radians
set angles [torch::tensor_create {0.0 1.57079633 3.14159265} float32]

# Convert to degrees using camelCase alias
set degrees [torch::radToDeg -input $angles]
# Result: tensor([0.0, 90.0, 180.0])
```

### Working with Negative Angles
```tcl
# Create a tensor with negative angles in radians
set angles [torch::tensor_create {-3.14159265 -1.57079633} float32]

# Convert to degrees
set degrees [torch::rad2deg $angles]
# Result: tensor([-180.0, -90.0])
```

## Error Handling

The command will raise an error in the following cases:
- If no input tensor is provided
- If an invalid parameter name is used
- If a parameter value is missing
- If the input tensor handle is invalid

## See Also

- [torch::deg2rad](deg2rad.md) - Convert angles from degrees to radians
- [torch::sin](sin.md) - Compute sine of angles in radians
- [torch::cos](cos.md) - Compute cosine of angles in radians
- [torch::tan](tan.md) - Compute tangent of angles in radians 