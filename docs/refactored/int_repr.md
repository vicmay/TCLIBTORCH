# torch::int_repr

Gets the integer representation of a quantized tensor. This function extracts the underlying integer values from a quantized tensor, which is useful for debugging quantization schemes, understanding quantization behavior, and implementing custom quantization operations.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::int_repr -input quantized_tensor
torch::int_repr -tensor quantized_tensor
```

### Positional Parameters (Legacy)
```tcl
torch::int_repr quantized_tensor
```

### CamelCase Alias
```tcl
torch::intRepr -input quantized_tensor
torch::intRepr -tensor quantized_tensor
```

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `-input` or `-tensor` | tensor | Yes | - | Quantized tensor to get integer representation from |

## Returns

Returns a new tensor containing the integer representation of the quantized tensor. The returned tensor has the same shape as the input but contains the underlying integer values used in the quantization.

## Description

The `int_repr` function extracts the integer values from a quantized tensor without applying the scale and zero-point transformation. This is particularly useful for:

- **Debugging quantization**: Understanding how floating-point values are mapped to integers
- **Custom quantization operations**: Implementing specialized quantized operations
- **Quantization analysis**: Analyzing the distribution of quantized values
- **Storage optimization**: Accessing the compact integer representation

For a quantized tensor created with scale `s` and zero-point `z`, the relationship between the original floating-point value `x`, the quantized integer `q`, and the dequantized value `x'` is:

```
q = round(x / s) + z
x' = (q - z) * s
```

The `int_repr` function returns the `q` values directly.

## Examples

### Basic Usage

```tcl
# Create a floating-point tensor
set tensor [torch::tensor -data {1.0 2.0 3.0 4.0} -shape {2 2}]

# Quantize it
set quantized [torch::quantize_per_tensor $tensor 0.1 10 quint8]

# Get integer representation (named parameters)
set int_repr [torch::int_repr -input $quantized]
puts [torch::tensor_shape $int_repr]
;# Output: {2 2}

# Using positional syntax (legacy)
set int_repr2 [torch::int_repr $quantized]
puts [torch::tensor_shape $int_repr2]
;# Output: {2 2}

# Using camelCase alias
set int_repr3 [torch::intRepr -input $quantized]
puts [torch::tensor_shape $int_repr3]
;# Output: {2 2}
```

### Quantization Workflow Analysis

```tcl
# Create test data
set original [torch::tensor -data {0.5 1.0 1.5 2.0} -shape {2 2}]

# Quantize with specific parameters
set scale 0.1
set zero_point 10
set quantized [torch::quantize_per_tensor $original $scale $zero_point quint8]

# Get integer representation
set int_values [torch::int_repr -input $quantized]
puts "Integer representation: [torch::tensor_data $int_values]"

# Dequantize for comparison
set dequantized [torch::dequantize $quantized]
puts "Original: [torch::tensor_data $original]"
puts "Dequantized: [torch::tensor_data $dequantized]"
```

### Different Quantization Schemes

```tcl
# Test different quantization parameters
set tensor [torch::ones -shape {3 3}]

# Low precision quantization
set quant1 [torch::quantize_per_tensor $tensor 0.01 100 quint8]
set int_repr1 [torch::int_repr -input $quant1]

# High precision quantization
set quant2 [torch::quantize_per_tensor $tensor 0.001 200 quint8]
set int_repr2 [torch::int_repr -input $quant2]

puts "Low precision int values: [torch::tensor_data $int_repr1]"
puts "High precision int values: [torch::tensor_data $int_repr2]"
```

### Multi-dimensional Tensors

```tcl
# 1D tensor
set tensor_1d [torch::randn -shape {10}]
set quant_1d [torch::quantize_per_tensor $tensor_1d 0.1 10 quint8]
set int_repr_1d [torch::int_repr -input $quant_1d]
puts "1D shape: [torch::tensor_shape $int_repr_1d]"

# 3D tensor
set tensor_3d [torch::randn -shape {2 3 4}]
set quant_3d [torch::quantize_per_tensor $tensor_3d 0.05 5 quint8]
set int_repr_3d [torch::int_repr -input $quant_3d]
puts "3D shape: [torch::tensor_shape $int_repr_3d]"

# 4D tensor (typical for neural networks)
set tensor_4d [torch::randn -shape {2 3 4 5}]
set quant_4d [torch::quantize_per_tensor $tensor_4d 0.02 20 quint8]
set int_repr_4d [torch::int_repr -input $quant_4d]
puts "4D shape: [torch::tensor_shape $int_repr_4d]"
```

### Quantization Debugging

```tcl
# Create a tensor with known values
set debug_tensor [torch::tensor -data {-1.0 0.0 1.0 2.0} -shape {2 2}]

# Quantize with specific parameters for easy calculation
set scale 0.5
set zero_point 128
set quantized [torch::quantize_per_tensor $debug_tensor $scale $zero_point quint8]

# Get integer representation
set int_repr [torch::int_repr -input $quantized]
puts "Original values: [torch::tensor_data $debug_tensor]"
puts "Integer representation: [torch::tensor_data $int_repr]"
puts "Expected calculation:"
puts "  -1.0 -> round(-1.0/0.5) + 128 = -2 + 128 = 126"
puts "   0.0 -> round(0.0/0.5) + 128 = 0 + 128 = 128"
puts "   1.0 -> round(1.0/0.5) + 128 = 2 + 128 = 130"
puts "   2.0 -> round(2.0/0.5) + 128 = 4 + 128 = 132"
```

### Quantization Analysis

```tcl
# Analyze quantization distribution
set large_tensor [torch::randn -shape {100 100}]
set quantized [torch::quantize_per_tensor $large_tensor 0.1 10 quint8]
set int_repr [torch::int_repr -input $quantized]

# Get statistics about the integer representation
set min_val [torch::tensor_min $int_repr]
set max_val [torch::tensor_max $int_repr]
set mean_val [torch::tensor_mean $int_repr]

puts "Integer representation statistics:"
puts "  Min: [torch::tensor_data $min_val]"
puts "  Max: [torch::tensor_data $max_val]"
puts "  Mean: [torch::tensor_data $mean_val]"
```

### Custom Quantization Operations

```tcl
# Example: Custom quantized operation using integer representation
set tensor1 [torch::ones -shape {2 2}]
set tensor2 [torch::ones -shape {2 2}]

# Quantize both tensors
set quant1 [torch::quantize_per_tensor $tensor1 0.1 10 quint8]
set quant2 [torch::quantize_per_tensor $tensor2 0.1 10 quint8]

# Get integer representations
set int1 [torch::int_repr -input $quant1]
set int2 [torch::int_repr -input $quant2]

puts "Quantized tensor 1 integers: [torch::tensor_data $int1]"
puts "Quantized tensor 2 integers: [torch::tensor_data $int2]"

# Note: For actual quantized arithmetic, use torch::quantized_add, etc.
```

## Parameter Aliases

The function supports multiple parameter names for flexibility:

- `-input` and `-tensor`: Both specify the quantized tensor

## Error Handling

The function provides clear error messages for invalid inputs:

```tcl
# Missing input tensor
catch {torch::int_repr} msg
puts $msg
;# Output: Required parameters missing: input quantized tensor required

# Invalid tensor handle
catch {torch::int_repr invalid_tensor} msg
puts $msg
;# Output: Invalid quantized tensor: invalid_tensor

# Unknown parameter
set tensor [torch::ones -shape {2 3}]
set quantized [torch::quantize_per_tensor $tensor 0.1 10 quint8]
catch {torch::int_repr -input $quantized -unknown_param value} msg
puts $msg
;# Output: Unknown parameter: -unknown_param

# Missing parameter value
catch {torch::int_repr -input} msg
puts $msg
;# Output: Named parameters must come in pairs
```

## Technical Details

### Input Requirements
- **Tensor Type**: Input must be a quantized tensor (created with `torch::quantize_per_tensor` or similar)
- **Shape**: Any shape is supported (1D, 2D, 3D, 4D, etc.)
- **Quantization**: Tensor must have been properly quantized with scale and zero-point

### Output Characteristics
- **Shape**: Same as input tensor
- **Data Type**: Integer type corresponding to the quantization scheme (typically `uint8` for `quint8`)
- **Values**: Raw integer values used in the quantized representation

### Performance Considerations
- **Memory**: O(N) where N is the number of elements
- **Computation**: O(1) - just extracts existing integer values
- **Efficiency**: Very fast operation as it only accesses existing data

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
torch::int_repr $quantized_tensor

# New named parameter syntax
torch::int_repr -input $quantized_tensor
torch::int_repr -tensor $quantized_tensor
```

### Advantages of Named Parameters
- **Clarity**: Makes the code more self-documenting
- **Consistency**: Matches the pattern used by other refactored commands
- **Flexibility**: Can be extended with additional parameters if needed
- **Error prevention**: Less likely to pass wrong arguments

## Related Functions

- `torch::quantize_per_tensor`: Creates quantized tensors
- `torch::quantize_per_channel`: Creates per-channel quantized tensors
- `torch::dequantize`: Converts quantized tensors back to floating-point
- `torch::q_scale`: Gets the scale of a quantized tensor
- `torch::q_zero_point`: Gets the zero-point of a quantized tensor

## Use Cases

### Quantization Debugging
```tcl
# Debug quantization behavior
set original [torch::tensor -data {1.1 2.2 3.3} -shape {3}]
set quantized [torch::quantize_per_tensor $original 0.1 10 quint8]
set int_repr [torch::int_repr -input $quantized]

puts "Original: [torch::tensor_data $original]"
puts "Quantized integers: [torch::tensor_data $int_repr]"
puts "Dequantized: [torch::tensor_data [torch::dequantize $quantized]]"
```

### Model Optimization Analysis
```tcl
# Analyze quantization impact on model weights
set weights [torch::randn -shape {64 32}]
set quantized_weights [torch::quantize_per_tensor $weights 0.01 128 quint8]
set weight_integers [torch::int_repr -input $quantized_weights]

# Analyze integer distribution
set unique_values [torch::tensor_unique $weight_integers]
puts "Number of unique quantized values: [torch::tensor_numel $unique_values]"
```

### Custom Quantization Schemes
```tcl
# Implement custom quantization analysis
set data [torch::randn -shape {1000}]
set quantized [torch::quantize_per_tensor $data 0.05 50 quint8]
set integers [torch::int_repr -input $quantized]

# Analyze quantization efficiency
set min_int [torch::tensor_min $integers]
set max_int [torch::tensor_max $integers]
puts "Integer range used: [torch::tensor_data $min_int] to [torch::tensor_data $max_int]"
puts "Range efficiency: [expr {([torch::tensor_data $max_int] - [torch::tensor_data $min_int]) / 255.0 * 100}]%"
```

### Storage and Serialization
```tcl
# Extract integer representation for compact storage
set model_tensor [torch::randn -shape {128 128}]
set quantized [torch::quantize_per_tensor $model_tensor 0.02 100 quint8]
set integers [torch::int_repr -input $quantized]

# The integers can be stored more compactly than floating-point values
puts "Original tensor size: [expr {[torch::tensor_numel $model_tensor] * 4}] bytes (float32)"
puts "Quantized integer size: [expr {[torch::tensor_numel $integers] * 1}] bytes (uint8)"
puts "Compression ratio: [expr {[torch::tensor_numel $model_tensor] * 4.0 / [torch::tensor_numel $integers]}]:1"
```

## Mathematical Background

### Quantization Formula
For a quantized tensor with scale `s` and zero-point `z`:
- **Quantization**: `q = round(x / s) + z`
- **Dequantization**: `x' = (q - z) * s`

The `int_repr` function returns the `q` values directly.

### Quantization Range
For `quint8` (8-bit unsigned integer):
- **Range**: 0 to 255
- **Zero-point**: Typically in range [0, 255]
- **Scale**: Determines the precision and range of representable values

### Precision Analysis
- **Precision**: Determined by the scale factor
- **Range**: Determined by the combination of scale and zero-point
- **Efficiency**: How well the available integer range is utilized

## Applications

### Model Quantization
- **Weight analysis**: Understanding how model weights are quantized
- **Activation analysis**: Analyzing quantized activations in neural networks
- **Calibration**: Determining optimal quantization parameters

### Embedded Systems
- **Memory optimization**: Accessing compact integer representations
- **Custom operations**: Implementing specialized quantized operations
- **Hardware acceleration**: Preparing data for quantized hardware

### Research and Development
- **Quantization research**: Analyzing different quantization schemes
- **Algorithm development**: Developing new quantization techniques
- **Performance analysis**: Understanding quantization impact on model accuracy

## See Also

- [torch::quantize_per_tensor](quantize_per_tensor.md) - Quantize tensors
- [torch::dequantize](dequantize.md) - Dequantize tensors
- [torch::q_scale](q_scale.md) - Get quantization scale
- [torch::q_zero_point](q_zero_point.md) - Get quantization zero-point
- [Quantization Guide](../guides/quantization.md) - Complete quantization workflow 