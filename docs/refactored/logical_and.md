# torch::logical_and

Performs element-wise logical AND operation between two tensors.

## Syntax

### Positional Arguments (Backward Compatible)
```tcl
torch::logical_and input1 input2
```

### Named Parameters
```tcl
torch::logical_and -input1 tensor1 -input2 tensor2
torch::logical_and -tensor1 tensor1 -tensor2 tensor2
```

### CamelCase Alias
```tcl
torch::logicalAnd -input1 tensor1 -input2 tensor2
torch::logicalAnd input1 input2
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| input1/tensor1 | Tensor | First input tensor | Yes |
| input2/tensor2 | Tensor | Second input tensor | Yes |

## Returns

Returns a tensor of boolean type containing the element-wise logical AND result.

## Description

The logical AND operation returns `true` only when both input values are non-zero (truthy), and `false` otherwise. For boolean tensors, this performs the standard AND truth table. For numeric tensors, zero values are considered `false`, and all non-zero values are considered `true`.

### Truth Table
| A | B | A AND B |
|---|---|---------|
| true | true | true |
| true | false | false |
| false | true | false |
| false | false | false |

### Broadcasting
This operation supports PyTorch's broadcasting rules, allowing tensors of different shapes to be combined as long as they are broadcast-compatible.

## Examples

### Basic Boolean AND
```tcl
# Create boolean tensors
set tensor1 [torch::tensor_create -data {true false true false} -dtype bool -device cpu -requiresGrad false]
set tensor2 [torch::tensor_create -data {true true false false} -dtype bool -device cpu -requiresGrad false]

# Positional syntax
set result1 [torch::logical_and $tensor1 $tensor2]
# Result: [true false false false]

# Named syntax
set result2 [torch::logical_and -input1 $tensor1 -input2 $tensor2]
# Result: [true false false false]

# CamelCase alias
set result3 [torch::logicalAnd -tensor1 $tensor1 -tensor2 $tensor2]
# Result: [true false false false]
```

### Numeric Tensors (Non-zero = true, Zero = false)
```tcl
set num1 [torch::tensor_create -data {1.0 0.0 2.0 -1.0} -dtype float32 -device cpu -requiresGrad false]
set num2 [torch::tensor_create -data {1.0 1.0 0.0 0.0} -dtype float32 -device cpu -requiresGrad false]

set result [torch::logical_and -input1 $num1 -input2 $num2]
# Result: [true false false false]
# 1.0 & 1.0 = true, 0.0 & 1.0 = false, 2.0 & 0.0 = false, -1.0 & 0.0 = false
```

### Broadcasting Example
```tcl
set scalar [torch::tensor_create -data {true} -dtype bool -device cpu -requiresGrad false]
set vector [torch::tensor_create -data {true false true false} -dtype bool -device cpu -requiresGrad false]

set result [torch::logical_and -input1 $scalar -input2 $vector]
# Result: [true false true false]
# Broadcasting: true & [true false true false] = [true false true false]
```

### Complex Boolean Expressions
```tcl
set a [torch::tensor_create -data {true false} -dtype bool -device cpu -requiresGrad false]
set b [torch::tensor_create -data {true true} -dtype bool -device cpu -requiresGrad false]
set c [torch::tensor_create -data {false true} -dtype bool -device cpu -requiresGrad false]

# Chained operations: (a AND b) AND c
set result1 [torch::logical_and $a $b]
set final [torch::logical_and $result1 $c]
# Result: [false false]
```

## Data Type Support

- **Boolean tensors**: Direct logical AND operation
- **Integer tensors**: Zero = false, non-zero = true
- **Float tensors**: Zero = false, non-zero = true
- **Mixed types**: Automatically handled by PyTorch

## Error Handling

```tcl
# Missing parameters
try {
    torch::logical_and
} trap {TCL ERROR} msg {
    puts "Error: $msg"  # "Required parameters missing"
}

# Invalid parameter names
try {
    torch::logical_and -invalid $tensor1 -input2 $tensor2
} trap {TCL ERROR} msg {
    puts "Error: $msg"  # "Unknown parameter: -invalid"
}

# Wrong number of positional arguments
try {
    torch::logical_and $tensor1
} trap {TCL ERROR} msg {
    puts "Error: $msg"  # "Wrong number of positional arguments"
}
```

## Performance Notes

- Logical operations are typically very fast
- Broadcasting may create temporary tensors for shape adjustment
- Boolean tensors are memory-efficient (1 bit per element in theory)
- Operations are performed element-wise and can be parallelized

## See Also

- [torch::logical_or](logical_or.md) - Element-wise logical OR
- [torch::logical_not](logical_not.md) - Element-wise logical NOT  
- [torch::logical_xor](logical_xor.md) - Element-wise logical XOR
- [torch::eq](eq.md) - Element-wise equality comparison

## Migration Guide

### From Positional to Named Parameters

```tcl
# Old (still supported)
set result [torch::logical_and $tensor1 $tensor2]

# New (recommended)
set result [torch::logical_and -input1 $tensor1 -input2 $tensor2]

# Alternative parameter names
set result [torch::logical_and -tensor1 $tensor1 -tensor2 $tensor2]

# CamelCase style
set result [torch::logicalAnd -input1 $tensor1 -input2 $tensor2]
```

The old positional syntax remains fully supported for backward compatibility. 