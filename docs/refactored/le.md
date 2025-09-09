# torch::le

Element-wise "less than or equal to" comparison between tensors.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::le -input1 tensor1 -input2 tensor2
torch::le -tensor1 tensor1 -tensor2 tensor2
torch::Le -input1 tensor1 -input2 tensor2
torch::Le -tensor1 tensor1 -tensor2 tensor2
```

### Positional Parameters (Legacy)
```tcl
torch::le tensor1 tensor2
torch::Le tensor1 tensor2
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `-input1` / `-tensor1` | string | Yes | Name of the first tensor |
| `-input2` / `-tensor2` | string | Yes | Name of the second tensor |

## Returns

Returns a boolean tensor containing the element-wise comparison results where each element is `true` if the corresponding element in the first tensor is less than or equal to the corresponding element in the second tensor, `false` otherwise.

## Description

The `torch::le` command performs element-wise "less than or equal to" comparison between two tensors. The operation follows PyTorch's broadcasting rules, allowing tensors of different but compatible shapes to be compared.

The comparison returns a boolean tensor where:
- `true` (1) indicates the element in tensor1 ≤ element in tensor2
- `false` (0) indicates the element in tensor1 > element in tensor2

This operation is fundamental for:
- **Masking and filtering**: Creating boolean masks for data selection
- **Conditional operations**: Implementing threshold-based logic
- **Model evaluation**: Comparing predictions with ground truth
- **Data validation**: Checking value ranges and constraints

## Broadcasting Rules

The tensors must be broadcast-compatible. Two tensors are broadcast-compatible if:
- They have the same number of dimensions, or
- One tensor has fewer dimensions and can be "expanded" to match the other
- For each dimension, the sizes are either equal, or one of them is 1

## Examples

### Basic Usage
```tcl
# Create test tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
set tensor2 [torch::tensor_create -data {1.5 2.0 2.5} -shape {3}]

# Element-wise comparison using named syntax (recommended)
set result [torch::le -input1 $tensor1 -input2 $tensor2]
# Result: [true, true, false] (1<=1.5, 2<=2.0, 3<=2.5)

# Using camelCase alias
set result [torch::Le -tensor1 $tensor1 -tensor2 $tensor2]

# Using legacy positional syntax
set result [torch::le $tensor1 $tensor2]
```

### Equal Values
```tcl
# Test with equal values
set tensor1 [torch::tensor_create -data {2.0 3.0 4.0} -shape {3}]
set tensor2 [torch::tensor_create -data {2.0 3.0 4.0} -shape {3}]

# All elements are equal, so all results are true
set result [torch::le -input1 $tensor1 -input2 $tensor2]
# Result: [true, true, true]
```

### Different Data Types
```tcl
# Integer tensors
set int_tensor1 [torch::tensor_create -data {1 3 5} -shape {3} -dtype int32]
set int_tensor2 [torch::tensor_create -data {2 3 4} -shape {3} -dtype int32]

set result [torch::le -input1 $int_tensor1 -input2 $int_tensor2]
# Result: [true, true, false] (1<=2, 3<=3, 5<=4)
```

### 2D Tensor Comparison
```tcl
# Create 2D tensors
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0 4.0} -shape {2 2}]
set tensor2 [torch::tensor_create -data {1.5 1.5 3.0 5.0} -shape {2 2}]

set result [torch::le -input1 $tensor1 -input2 $tensor2]
# Result: [[true, false], [true, true]]
# Comparisons: 1.0<=1.5, 2.0<=1.5, 3.0<=3.0, 4.0<=5.0
```

### Broadcasting Examples
```tcl
# Scalar vs tensor broadcasting
set tensor1 [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
set scalar [torch::tensor_create -data {2.0} -shape {1}]

set result [torch::le -input1 $tensor1 -input2 $scalar]
# Result: [true, true, false] (1<=2, 2<=2, 3<=2)

# Different shape broadcasting
set tensor1 [torch::tensor_create -data {1.0 2.0} -shape {2 1}]
set tensor2 [torch::tensor_create -data {1.5 2.5} -shape {1 2}]

set result [torch::le -input1 $tensor1 -input2 $tensor2]
# Result: [[true, true], [false, true]]
```

### Threshold Detection
```tcl
# Find values below or equal to threshold
set data [torch::tensor_create -data {0.1 0.5 0.8 1.2 1.5} -shape {5}]
set threshold [torch::tensor_create -data {1.0} -shape {1}]

set mask [torch::le -input1 $data -input2 $threshold]
# mask: [true, true, true, false, false]

# Use mask for filtering (if supported by your workflow)
puts "Values <= 1.0 detected"
```

### Model Evaluation
```tcl
# Compare predictions with thresholds for classification
set predictions [torch::tensor_create -data {0.3 0.7 0.9 0.2} -shape {4}]
set threshold [torch::tensor_create -data {0.5} -shape {1}]

set low_confidence [torch::le -input1 $predictions -input2 $threshold]
# Result: [true, false, false, true]
puts "Low confidence predictions identified"
```

### Range Validation
```tcl
# Check if values are within acceptable range
set values [torch::tensor_create -data {-1.0 0.5 2.0 3.5} -shape {4}]
set upper_bound [torch::tensor_create -data {3.0} -shape {1}]

set within_range [torch::le -input1 $values -input2 $upper_bound]
# Result: [true, true, true, false]
puts "Range validation complete"
```

## Migration Guide

### From Legacy Positional Syntax
```tcl
# Old syntax (still supported)
torch::le $tensor1 $tensor2

# New syntax (recommended)
torch::le -input1 $tensor1 -input2 $tensor2

# Or using tensor aliases
torch::le -tensor1 $tensor1 -tensor2 $tensor2

# Or using camelCase
torch::Le -input1 $tensor1 -input2 $tensor2
```

### Benefits of Named Parameters
- **Clarity**: Makes it explicit which tensor is which operand
- **Maintainability**: Code is self-documenting
- **Consistency**: Matches modern TCL conventions
- **Flexibility**: Parameter order doesn't matter
- **Error Prevention**: Reduces risk of swapping operands

## Error Handling

The command will throw an error in the following cases:

```tcl
# Invalid tensor names
catch {torch::le -input1 "nonexistent" -input2 $tensor2} error
puts "Error: $error"

# Missing required parameters
catch {torch::le -input1 $tensor1} error
puts "Error: $error"

# Unknown parameters
catch {torch::le -invalid_param $tensor1 -input2 $tensor2} error
puts "Error: $error"

# Incompatible tensor shapes (for broadcasting)
set tensor1 [torch::tensor_create -data {1.0 2.0} -shape {2}]
set tensor2 [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
catch {torch::le -input1 $tensor1 -input2 $tensor2} error
puts "Error: $error"
```

## Performance Considerations

### Memory Usage
- Creates a new boolean tensor for the result
- Memory usage scales with the size of the larger tensor after broadcasting
- Boolean tensors use 1 byte per element

### Broadcasting Efficiency
- Broadcasting is performed efficiently without copying data
- Large tensors with small scalars are very efficient
- Avoid unnecessary shape mismatches

### Best Practices
```tcl
# Good: Reuse comparison results
set mask [torch::le -input1 $data -input2 $threshold]
# Use mask multiple times...

# Good: Use appropriate data types
set int_result [torch::le -input1 $int_tensor1 -input2 $int_tensor2]

# Consider: Pre-allocate result tensors for repeated operations
# (if your workflow supports it)
```

## Integration with Other Operations

### Combining with Logical Operations
```tcl
# Find values in range [min, max]
set data [torch::tensor_create -data {0.5 1.5 2.5 3.5} -shape {4}]
set min_val [torch::tensor_create -data {1.0} -shape {1}]
set max_val [torch::tensor_create -data {3.0} -shape {1}]

set ge_min [torch::ge -input1 $data -input2 $min_val]
set le_max [torch::le -input1 $data -input2 $max_val]
set in_range [torch::logical_and -input1 $ge_min -input2 $le_max]
```

### Conditional Operations
```tcl
# Use with where/conditional operations (if available)
set condition [torch::le -input1 $tensor1 -input2 $tensor2]
# Apply conditional logic based on mask
```

## Mathematical Properties

### Reflexivity
```tcl
# For any tensor A: A <= A is always true
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -shape {3}]
set result [torch::le -input1 $tensor -input2 $tensor]
# Result: [true, true, true]
```

### Transitivity
If A <= B and B <= C, then A <= C (element-wise)

### Relationship with Other Comparisons
- `a <= b` is equivalent to `!(a > b)`
- `a <= b` is equivalent to `(a < b) || (a == b)`

## See Also

- [torch::lt](lt.md) - Less than comparison
- [torch::ge](ge.md) - Greater than or equal comparison
- [torch::gt](gt.md) - Greater than comparison
- [torch::eq](eq.md) - Equality comparison
- [torch::ne](ne.md) - Not equal comparison
- [torch::logical_and](logical_and.md) - Logical AND operation
- [torch::where](where.md) - Conditional selection based on boolean mask

## Notes

- The comparison follows IEEE 754 standards for floating-point values
- NaN values have special behavior in comparisons
- Broadcasting follows PyTorch's broadcasting semantics
- The result tensor shape follows broadcasting rules
- Boolean tensors can be used as masks in many operations 