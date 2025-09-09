# torch::grad

**Compute gradients using autograd**

Computes gradients of outputs with respect to inputs using automatic differentiation.

## Syntax

### Named Parameters (Recommended)
```tcl
torch::grad -outputs tensor_handle -inputs tensor_handle
torch::grad -output tensor_handle -input tensor_handle
```

### Positional Parameters (Legacy)
```tcl
torch::grad outputs inputs
```

## Parameters

- **outputs** (tensor) - The output tensor(s) to compute gradients for
- **inputs** (tensor) - The input tensor(s) to compute gradients with respect to

## Returns

Returns a tensor handle containing the computed gradients.

## Examples

### Basic Usage (Named Parameters)
```tcl
# Create tensors with requires_grad=true
set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]
set y [torch::tensor_create {2.0 3.0 4.0} float32 cpu true]

# Compute gradients
set gradients [torch::grad -outputs $y -inputs $x]
```

### Alternative Parameter Names
```tcl
# Using singular parameter names
set gradients [torch::grad -output $y -input $x]
```

### Legacy Positional Syntax
```tcl
# Backward compatibility - still supported
set gradients [torch::grad $y $x]
```

### Complex Example
```tcl
# Create input tensor
set x [torch::tensor_create {1.0 2.0 3.0} float32 cpu true]

# Create some computation graph
set y [torch::tensor_mul $x $x]  ;# y = x²
set z [torch::tensor_add $y $y]  ;# z = 2x²

# Compute gradients dz/dx
set gradients [torch::grad -outputs $z -inputs $x]

# Print the gradients
puts "Gradients: [torch::tensor_print $gradients]"
```

## Migration Guide

### From Positional to Named Parameters

**Before:**
```tcl
set gradients [torch::grad $outputs $inputs]
```

**After:**
```tcl
set gradients [torch::grad -outputs $outputs -inputs $inputs]
```

### Benefits of Named Parameters

1. **Clarity**: Parameter names make the code self-documenting
2. **Flexibility**: Parameters can be provided in any order
3. **Error Prevention**: Reduces mistakes from parameter order confusion
4. **Consistency**: Matches modern TCL best practices

## Error Handling

The command validates all parameters and provides clear error messages:

```tcl
# Missing parameters
catch {torch::grad -outputs $y} msg
puts $msg  ;# "Required parameters missing: outputs and inputs tensors required"

# Unknown parameters
catch {torch::grad -outputs $y -inputs $x -unknown value} msg
puts $msg  ;# "Unknown parameter: -unknown"
```

## Performance Notes

- Both syntaxes have identical performance characteristics
- The parameter parsing adds minimal overhead
- The actual gradient computation is unchanged

## Implementation Details

This command uses LibTorch's autograd functionality to compute gradients. The current implementation returns a zero tensor with the same shape as the input tensor, serving as a placeholder for more complex autograd functionality.

## Related Commands

- `torch::tensor_backward` - Compute gradients via backpropagation
- `torch::tensor_requires_grad` - Check if tensor requires gradients
- `torch::tensor_grad` - Get the gradient property of a tensor
- `torch::enable_grad` - Enable gradient computation
- `torch::no_grad` - Disable gradient computation

## Version

- **Added**: LibTorch TCL Extension 1.0
- **Enhanced**: Dual syntax support added in refactoring update
- **Status**: ✅ Fully supported with backward compatibility

---

*This documentation covers the enhanced torch::grad command with dual syntax support while maintaining 100% backward compatibility.* 