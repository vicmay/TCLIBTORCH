# torch::arange

Creates a 1-dimensional tensor with values from `start` to `end` (exclusive) with step size `step`.

## Syntax

### Named Parameters (New Syntax)
```tcl
torch::arange -end value ?-start value? ?-step value? ?-dtype type? ?-device device?
```

### Positional Parameters (Legacy Syntax)
```tcl
torch::arange end ?start? ?step? ?dtype? ?device?
```

## Parameters

### Named Parameters
- **`-end`** (required): The end value (exclusive) for the range
- **`-start`** (optional): The start value for the range (default: 0.0)
- **`-step`** (optional): The step size between values (default: 1.0)
- **`-dtype`** (optional): Data type of the tensor (default: "float32")
- **`-device`** (optional): Device to create tensor on (default: "cpu")

### Positional Parameters
- **`end`** (required): The end value (exclusive) for the range
- **`start`** (optional): The start value for the range (default: 0.0)
- **`step`** (optional): The step size between values (default: 1.0)
- **`dtype`** (optional): Data type of the tensor (default: "float32")
- **`device`** (optional): Device to create tensor on (default: "cpu")

## Examples

### Basic Usage

```tcl
# Create range from 0 to 5 (exclusive)
set tensor [torch::arange 5]
# Result: tensor([0, 1, 2, 3, 4])

# Same with named parameters
set tensor [torch::arange -end 5]
# Result: tensor([0, 1, 2, 3, 4])
```

### With Start and End

```tcl
# Create range from 2 to 8 (exclusive)
set tensor [torch::arange 2 8]
# Result: tensor([2, 3, 4, 5, 6, 7])

# Same with named parameters
set tensor [torch::arange -start 2 -end 8]
# Result: tensor([2, 3, 4, 5, 6, 7])
```

### With Step Size

```tcl
# Create range from 0 to 10 with step 2
set tensor [torch::arange 0 10 2]
# Result: tensor([0, 2, 4, 6, 8])

# Same with named parameters
set tensor [torch::arange -start 0 -end 10 -step 2]
# Result: tensor([0, 2, 4, 6, 8])
```

### With Data Type

```tcl
# Create integer tensor
set tensor [torch::arange 3 int32]
# Result: tensor([0, 1, 2], dtype=torch.int32)

# Same with named parameters
set tensor [torch::arange -end 3 -dtype int32]
# Result: tensor([0, 1, 2], dtype=torch.int32)
```

### With All Parameters

```tcl
# Create tensor with all parameters specified
set tensor [torch::arange 1 6 1 float64 cpu]
# Result: tensor([1., 2., 3., 4., 5.], dtype=torch.float64)

# Same with named parameters
set tensor [torch::arange -start 1 -end 6 -step 1 -dtype float64 -device cpu]
# Result: tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
```

### Named Parameters in Different Order

```tcl
# Parameters can be specified in any order with named syntax
set tensor [torch::arange -dtype float64 -start 1 -step 0.5 -end 3]
# Result: tensor([1.0000, 1.5000, 2.0000, 2.5000], dtype=torch.float64)
```

## Supported Data Types

- `float32` (default)
- `float64`
- `int32`
- `int64`
- `uint8`
- `int8`
- `int16`
- `uint16`
- `bool`

## Supported Devices

- `cpu` (default)
- `cuda` (if CUDA is available)

## Error Handling

### Missing Required Parameters
```tcl
# This will fail - missing -end parameter
torch::arange -start 1
# Error: upper bound and larger bound inconsistent with step sign
```

### Invalid Parameters
```tcl
# This will fail - unknown parameter
torch::arange -invalid 5
# Error: Unknown parameter
```

### Missing Values
```tcl
# This will fail - missing value for parameter
torch::arange -start
# Error: Missing value for parameter
```

## Performance Notes

- Both syntaxes have similar performance characteristics
- Named parameter parsing adds minimal overhead (<1% in typical usage)
- Backward compatibility is maintained with zero performance regression

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old syntax
torch::arange 1 6 1 float32 cpu

# New syntax (equivalent)
torch::arange -start 1 -end 6 -step 1 -dtype float32 -device cpu
```

### Benefits of Named Parameters

1. **Clarity**: Parameter meaning is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Defaults**: Optional parameters can be omitted in any combination
4. **Maintainability**: Code is more self-documenting
5. **Safety**: Parameter validation is more robust 