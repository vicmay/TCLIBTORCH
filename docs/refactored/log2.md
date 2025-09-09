# torch::log2

## Overview
Computes the base-2 logarithm of input tensor elements. This function supports both traditional positional syntax and modern named parameter syntax.

## Syntax

### Positional Syntax (Traditional)
```tcl
torch::log2 tensor
```

### Named Parameter Syntax (Recommended)
```tcl
torch::log2 -input tensor
torch::log2 -tensor tensor
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` / `tensor` | string | Yes | Name of input tensor |

## Returns
- **Type**: string
- **Description**: Handle to the result tensor containing base-2 logarithms

## Mathematical Definition
For input tensor **x**, computes:
```
log₂(x) = log(x) / log(2)
```

Where:
- For x > 0: Returns the base-2 logarithm
- For x = 1: Returns exactly 0
- For x = 2^n: Returns exactly n

## Key Properties
1. **Binary logarithm**: Specifically optimized for powers of 2
2. **Computer science applications**: Essential for complexity analysis
3. **Bit manipulation**: Related to binary representation lengths
4. **Mathematical relationship**: log₂(x) = ln(x) / ln(2)

## Examples

### Basic Usage with Powers of 2
```tcl
# Create tensor with powers of 2
set input [torch::tensor_create -data {1.0 2.0 4.0 8.0 16.0} -dtype float32 -device cpu]

# Positional syntax
set result1 [torch::log2 $input]
# Result: [0.0, 1.0, 2.0, 3.0, 4.0]

# Named parameter syntax
set result2 [torch::log2 -input $input]
# Result: [0.0, 1.0, 2.0, 3.0, 4.0]
```

### Computer Science Applications
```tcl
# Array sizes for complexity analysis
set sizes [torch::tensor_create -data {1024.0 2048.0 4096.0 8192.0} -dtype float32 -device cpu]
set log_sizes [torch::log2 -input $sizes]
# Result: [10.0, 11.0, 12.0, 13.0] - representing log₂(n) complexity

# Binary tree depth calculation
set nodes [torch::tensor_create -data {1.0 2.0 4.0 8.0 16.0 32.0} -dtype float32 -device cpu]
set depth [torch::log2 -input $nodes]
# Result: [0.0, 1.0, 2.0, 3.0, 4.0, 5.0] - tree depths
```

### Non-Power-of-2 Values
```tcl
# General values
set input [torch::tensor_create -data {3.0 5.0 10.0 100.0} -dtype float64 -device cpu]
set result [torch::log2 -input $input]
# Result: [1.585, 2.322, 3.322, 6.644] (approximate)
```

### Fractional Powers
```tcl
# Fractional powers of 2
set fractions [torch::tensor_create -data {0.5 0.25 0.125} -dtype float32 -device cpu]
set result [torch::log2 -input $fractions]
# Result: [-1.0, -2.0, -3.0] - negative logarithms
```

## Data Type Support
- **float32**: Standard precision
- **float64**: Double precision  
- **double**: Extended precision

```tcl
# Float64 for high precision
set input [torch::tensor_create -data {2.0 4.0 8.0} -dtype float64 -device cpu]
set result [torch::log2 -input $input]

# Double precision
set input [torch::tensor_create -data {1.0 2.0 4.0} -dtype double -device cpu]
set result [torch::log2 -input $input]
```

## Computer Science Applications

### Algorithm Complexity
```tcl
# Analyzing O(log n) complexity
set n_values [torch::tensor_create -data {1.0 2.0 4.0 8.0 16.0 32.0 64.0} -dtype float32 -device cpu]
set operations [torch::log2 -input $n_values]
# Shows how operations scale logarithmically with input size
```

### Binary Representation
```tcl
# Number of bits needed to represent values
set values [torch::tensor_create -data {1.0 2.0 3.0 4.0 7.0 8.0 15.0 16.0} -dtype float32 -device cpu]
set bits_needed [torch::log2 -input $values]
# Ceiling of result gives minimum bits needed
```

### Information Theory
```tcl
# Calculating information content (bits)
set probabilities [torch::tensor_create -data {0.5 0.25 0.125 0.0625} -dtype float64 -device cpu]
set information [torch::log2 -input $probabilities]
# Negative of result gives information content in bits
```

## Relationship with Other Functions

### Inverse Relationship with exp2
```tcl
set x [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu]
set log2_x [torch::log2 -input $x]
set reconstructed [torch::exp2 -input $log2_x]
# reconstructed ≈ x (within numerical precision)
```

### Conversion from Natural Logarithm
```tcl
set x [torch::tensor_create -data {2.0 4.0 8.0} -dtype float64 -device cpu]

# Using log2 directly
set result1 [torch::log2 -input $x]

# Using natural log and conversion
set ln_x [torch::tensor_log $x]
set ln_2 [torch::tensor_create -data {0.6931471805599453} -dtype float64 -device cpu]
set result2 [torch::tensor_div $ln_x $ln_2]
# result1 ≈ result2
```

## Error Handling

The command validates input parameters and provides clear error messages:

### Missing Arguments
```tcl
torch::log2
# Error: Usage: torch::log2 tensor
#    or: torch::log2 -input TENSOR
```

### Invalid Tensor
```tcl
torch::log2 invalid_tensor
# Error: Invalid tensor name
```

### Missing Parameter Values
```tcl
torch::log2 -input
# Error: Missing value for parameter
```

### Unknown Parameters
```tcl
torch::log2 -unknown_param tensor1
# Error: Unknown parameter: -unknown_param
```

## Performance Considerations

### Numerical Stability
- For powers of 2: Results are exact
- For other values: Standard floating-point precision applies
- Very small values: May approach precision limits

### Optimization
- Direct hardware implementation for common cases
- Optimized for power-of-2 inputs
- Efficient for large tensors

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old style
set result [torch::log2 $tensor]

# New style (recommended)
set result [torch::log2 -input $tensor]
```

### Benefits of Named Syntax
1. **Clarity**: Parameter purpose is explicit
2. **Maintainability**: Code is self-documenting
3. **Flexibility**: Can add optional parameters easily
4. **Consistency**: Matches other modern torch commands

## See Also
- [`torch::log`](log.md) - Natural logarithm
- [`torch::log10`](log10.md) - Base-10 logarithm
- [`torch::log1p`](log1p.md) - log(1+x) with better precision
- [`torch::exp2`](exp2.md) - Base-2 exponential (inverse)
- [`torch::pow`](pow.md) - General power function

## Mathematical Notes

### Binary Logarithm Properties
- log₂(1) = 0
- log₂(2) = 1  
- log₂(2ⁿ) = n
- log₂(x·y) = log₂(x) + log₂(y)
- log₂(x/y) = log₂(x) - log₂(y)
- log₂(xⁿ) = n·log₂(x)

### Domain and Range
- **Domain**: x > 0 (positive real numbers)
- **Range**: All real numbers
- **Special cases**: 
  - log₂(1) = 0
  - log₂(2) = 1
  - log₂(0.5) = -1

### Relationship to Information Theory
In information theory, log₂ represents information content in bits:
- 1 bit = log₂(2) = 1 unit of information
- n bits can represent 2ⁿ distinct states
- Information content of probability p is -log₂(p) bits 