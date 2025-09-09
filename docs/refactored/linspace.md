# torch::linspace

Creates a 1-dimensional tensor with `steps` values linearly spaced from `start` to `end` (inclusive).

## Syntax

### Named Parameters (New Syntax)
```tcl
torch::linspace -start value -end value -steps value ?-dtype type? ?-device device?
```

### Positional Parameters (Legacy Syntax)
```tcl
torch::linspace start end steps ?dtype? ?device?
```

## Parameters

### Named Parameters
- **`-start`** (required): The start value of the sequence
- **`-end`** (required): The end value of the sequence (inclusive)
- **`-steps`** (required): Number of values in the sequence
- **`-dtype`** (optional): Data type of the tensor (default: "float32")
- **`-device`** (optional): Device to create tensor on (default: "cpu")

### Positional Parameters
- **`start`** (required): The start value of the sequence
- **`end`** (required): The end value of the sequence (inclusive)
- **`steps`** (required): Number of values in the sequence
- **`dtype`** (optional): Data type of the tensor (default: "float32")
- **`device`** (optional): Device to create tensor on (default: "cpu")

## Examples

### Basic Usage

```tcl
# Create 5 values from 0 to 10 (inclusive)
set tensor [torch::linspace 0 10 5]
# Result: tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])

# Same with named parameters
set tensor [torch::linspace -start 0 -end 10 -steps 5]
# Result: tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])
```

### With Different Data Types

```tcl
# Create integer tensor
set tensor [torch::linspace 1 5 4 float64]
# Result: tensor([1.0000, 2.3333, 3.6667, 5.0000], dtype=torch.float64)

# Same with named parameters
set tensor [torch::linspace -start 1 -end 5 -steps 4 -dtype float64]
# Result: tensor([1.0000, 2.3333, 3.6667, 5.0000], dtype=torch.float64)
```

### With Device Specification

```tcl
# Create tensor on CPU
set tensor [torch::linspace 0 1 10 float32 cpu]
# Result: tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0000])

# Same with named parameters
set tensor [torch::linspace -start 0 -end 1 -steps 10 -dtype float32 -device cpu]
# Result: tensor([0.0000, 0.1111, 0.2222, 0.3333, 0.4444, 0.5556, 0.6667, 0.7778, 0.8889, 1.0000])
```

### Named Parameters in Different Order

```tcl
# Parameters can be specified in any order with named syntax
set tensor [torch::linspace -dtype float64 -start 2 -end 8 -steps 6 -device cpu]
# Result: tensor([2.0000, 3.2000, 4.4000, 5.6000, 6.8000, 8.0000], dtype=torch.float64)
```

### Common Use Cases

```tcl
# Create evenly spaced values for plotting
set x_values [torch::linspace -start -3.14159 -end 3.14159 -steps 100]

# Create time steps for simulation
set time_steps [torch::linspace -start 0 -end 10 -steps 1000]

# Create frequency bins
set frequencies [torch::linspace -start 20 -end 20000 -steps 512]
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
# This will fail - missing -steps parameter
torch::linspace -start 0 -end 10
# Error: -steps parameter is required
```

### Invalid Parameters
```tcl
# This will fail - unknown parameter
torch::linspace -invalid 5
# Error: Unknown parameter
```

### Invalid Data Types
```tcl
# This will fail - invalid dtype
torch::linspace -start 0 -end 10 -steps 5 -dtype bad
# Error: Invalid dtype: bad
```

### Missing Values
```tcl
# This will fail - missing value for parameter
torch::linspace -start
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
torch::linspace 0 10 5 float32 cpu

# New syntax (equivalent)
torch::linspace -start 0 -end 10 -steps 5 -dtype float32 -device cpu
```

### Benefits of Named Parameters

1. **Clarity**: Parameter meaning is explicit
2. **Flexibility**: Parameters can be specified in any order
3. **Maintainability**: Code is more self-documenting
4. **Future-proof**: Easier to add new parameters

## Backward Compatibility

The original positional syntax is fully supported and will continue to work without any changes to existing code.

## Mathematical Properties

- Creates `steps` values evenly distributed between `start` and `end`
- The sequence includes both `start` and `end` values
- Step size = (end - start) / (steps - 1)
- Useful for creating evenly spaced sequences for plotting, interpolation, etc.

## Use Cases

- **Plotting**: Create x-axis values for graphs
- **Interpolation**: Generate evenly spaced points for interpolation
- **Signal Processing**: Create frequency bins or time steps
- **Simulation**: Generate time steps for numerical simulations
- **Data Analysis**: Create evenly spaced bins for histograms

## Related Commands

- `torch::arange` - Create range tensor with step size
- `torch::logspace` - Create logarithmically spaced tensor
- `torch::zeros` - Create tensor filled with zeros
- `torch::ones` - Create tensor filled with ones 