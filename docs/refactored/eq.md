# torch::eq

Element-wise equality comparison between tensors. Returns a boolean tensor where each element indicates whether the corresponding elements in the input tensors are equal.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::eq tensor1 tensor2
```

### Named Parameter Syntax
```tcl
torch::eq -input1 tensor1 -input2 tensor2
torch::eq -tensor1 tensor1 -tensor2 tensor2
```

### CamelCase Alias
```tcl
torch::Eq tensor1 tensor2
torch::Eq -input1 tensor1 -input2 tensor2
```

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tensor1` / `-input1` / `-tensor1` | Tensor | Yes | First input tensor for comparison |
| `tensor2` / `-input2` / `-tensor2` | Tensor | Yes | Second input tensor for comparison |

## Returns

Returns a tensor handle containing a boolean tensor where each element is `true` if the corresponding elements in the input tensors are equal, `false` otherwise.

## Mathematical Foundation

The equality operation computes element-wise equality:

```
output[i] = input1[i] == input2[i]
```

For tensors with different shapes, PyTorch broadcasting rules apply:
- Tensors are aligned from the rightmost dimension
- Dimensions of size 1 can be broadcast to any size
- Missing dimensions are assumed to be of size 1

### Broadcasting Examples

1. **Scalar vs Vector**: `[3] == [1]` → `[3]` (scalar broadcast to vector)
2. **Different Shapes**: `[2,1] == [3]` → `[2,3]` (both dimensions broadcast)
3. **Same Shape**: `[2,3] == [2,3]` → `[2,3]` (no broadcasting needed)

## Examples

### Basic Usage

```tcl
# Create test tensors
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -shape {3}]
set t2 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -shape {3}]

# Positional syntax
set result1 [torch::eq $t1 $t2]
# Returns: tensor with [true, true, true]

# Named parameter syntax
set result2 [torch::eq -input1 $t1 -input2 $t2]
# Returns: same result as above

# CamelCase alias
set result3 [torch::Eq $t1 $t2]
# Returns: same result as above
```

### Different Values

```tcl
set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -shape {3}]
set t2 [torch::tensor_create -data {1.0 2.5 3.0} -dtype float32 -shape {3}]

set result [torch::eq $t1 $t2]
# Returns: tensor with [true, false, true]
```

### Broadcasting Examples

```tcl
# Scalar broadcast to vector
set scalar [torch::tensor_create -data {2.0} -dtype float32 -shape {1}]
set vector [torch::tensor_create -data {2.0 2.0 2.0} -dtype float32 -shape {3}]

set result [torch::eq $scalar $vector]
# Returns: tensor with shape [3] containing [true, true, true]

# Matrix vs vector broadcasting
set matrix [torch::tensor_create -data {1.0 2.0} -dtype float32 -shape {2 1}]
set vector [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -shape {3}]

set result [torch::eq $matrix $vector]
# Returns: tensor with shape [2, 3]
```

### Integration with Other Operations

```tcl
# Use equality result for counting
set data [torch::tensor_create -data {1.0 2.0 1.0 3.0 1.0} -dtype float32 -shape {5}]
set target [torch::tensor_create -data {1.0} -dtype float32 -shape {1}]

set mask [torch::eq $data $target]
set count [torch::tensor_sum $mask]
# count contains the number of elements equal to 1.0
```

### Data Type Support

```tcl
# Float32 tensors
set f1 [torch::tensor_create -data {1.5 2.5} -dtype float32 -shape {2}]
set f2 [torch::tensor_create -data {1.5 2.5} -dtype float32 -shape {2}]
set result_f [torch::eq $f1 $f2]

# Integer tensors
set i1 [torch::tensor_create -data {10 20} -dtype int64 -shape {2}]
set i2 [torch::tensor_create -data {10 20} -dtype int64 -shape {2}]
set result_i [torch::eq $i1 $i2]

# All return boolean tensors regardless of input type
```

## Use Cases

### 1. Condition Checking
```tcl
# Check which elements meet a condition
set data [torch::tensor_create -data {1.0 2.0 3.0 2.0 1.0} -dtype float32 -shape {5}]
set threshold [torch::tensor_create -data {2.0} -dtype float32 -shape {1}]

set is_equal [torch::eq $data $threshold]
# Result: boolean mask indicating which elements equal 2.0
```

### 2. Model Evaluation
```tcl
# Compare predictions with ground truth
set predictions [torch::tensor_create -data {1 0 1 1 0} -dtype int64 -shape {5}]
set ground_truth [torch::tensor_create -data {1 0 1 0 0} -dtype int64 -shape {5}]

set correct [torch::eq $predictions $ground_truth]
set accuracy [torch::tensor_mean [torch::tensor_to_dtype $correct float32]]
```

### 3. Data Filtering
```tcl
# Find specific values in data
set data [torch::tensor_create -data {1.0 2.0 1.0 3.0 1.0} -dtype float32 -shape {5}]
set target_value [torch::tensor_create -data {1.0} -dtype float32 -shape {1}]

set mask [torch::eq $data $target_value]
# Use mask for indexing or further processing
```

### 4. Validation and Testing
```tcl
# Check if two computational results are identical
set result1 [torch::tensor_add $tensor_a $tensor_b]
set result2 [torch::tensor_add $tensor_b $tensor_a]  # Should be same due to commutativity

set are_equal [torch::eq $result1 $result2]
set all_equal [torch::tensor_all $are_equal]
# Verify mathematical properties
```

## Mathematical Properties

### Reflexivity
```tcl
# Any tensor equals itself
set tensor [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -shape {3}]
set self_eq [torch::eq $tensor $tensor]
# Result: all elements are true
```

### Symmetry
```tcl
# eq(a,b) == eq(b,a)
set eq_ab [torch::eq $tensor_a $tensor_b]
set eq_ba [torch::eq $tensor_b $tensor_a]
# eq_ab and eq_ba are identical
```

### Special Values
```tcl
# Zero equality
set zeros1 [torch::tensor_create -data {0.0 0.0} -dtype float32 -shape {2}]
set zeros2 [torch::tensor_create -data {0.0 0.0} -dtype float32 -shape {2}]
set result [torch::eq $zeros1 $zeros2]  # [true, true]

# Negative values
set neg1 [torch::tensor_create -data {-1.0 -2.0} -dtype float32 -shape {2}]
set neg2 [torch::tensor_create -data {-1.0 -2.0} -dtype float32 -shape {2}]
set result [torch::eq $neg1 $neg2]  # [true, true]
```

## Error Handling

The command provides detailed error messages for common issues:

```tcl
# Missing arguments
catch {torch::eq} error
# Error: Usage: torch::eq tensor1 tensor2 | torch::eq -input1 tensor1 -input2 tensor2

# Invalid tensor name
catch {torch::eq "invalid_tensor" $valid_tensor} error
# Error: Invalid tensor name for input1

# Missing named parameter value
catch {torch::eq -input1 $tensor1 -input2} error
# Error: Missing value for parameter

# Unknown parameter
catch {torch::eq -input1 $tensor1 -unknown_param $tensor2} error
# Error: Unknown parameter: -unknown_param. Valid parameters are: -input1/-tensor1, -input2/-tensor2
```

## Performance Considerations

1. **Memory Efficiency**: The operation is performed in-place where possible
2. **Broadcasting**: Large tensor broadcasts may require significant memory
3. **Data Types**: Boolean output tensors are memory-efficient
4. **GPU Support**: Automatically uses GPU tensors when available

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old positional syntax
set result [torch::eq $tensor1 $tensor2]

# New named parameter syntax
set result [torch::eq -input1 $tensor1 -input2 $tensor2]

# Or with alternative parameter names
set result [torch::eq -tensor1 $tensor1 -tensor2 $tensor2]
```

### Benefits of Named Parameters

1. **Self-documenting**: Parameter names make code more readable
2. **Order independence**: Parameters can be specified in any order
3. **Future extensibility**: Easy to add new optional parameters
4. **Error prevention**: Reduces mistakes from parameter ordering

## See Also

- [torch::ne](ne.md) - Element-wise not-equal comparison
- [torch::gt](gt.md) - Element-wise greater-than comparison
- [torch::lt](lt.md) - Element-wise less-than comparison
- [torch::ge](ge.md) - Element-wise greater-than-or-equal comparison
- [torch::le](le.md) - Element-wise less-than-or-equal comparison
- [torch::tensor_all](tensor_all.md) - Check if all elements are true
- [torch::tensor_any](tensor_any.md) - Check if any elements are true

## Version Information

- **Added**: LibTorch TCL Extension 1.0
- **Modified**: Version 2.0 (Added named parameter support and camelCase aliases)
- **Backward Compatibility**: Fully maintained with positional syntax 