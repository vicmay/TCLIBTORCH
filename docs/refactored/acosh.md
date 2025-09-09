# torch::acosh

## Description
Computes the inverse hyperbolic cosine (acosh) of the elements in the input tensor. This function supports both positional and named parameter syntax for maximum flexibility.

## Syntax

### Positional Syntax (Backward Compatible)
```tcl
torch::acosh tensor
```

### Named Parameter Syntax (Recommended)
```tcl
torch::acosh -input tensor
torch::acosh -tensor tensor
```

## Parameters

| Parameter | Type | Description | Required |
|-----------|------|-------------|----------|
| `tensor` / `input` / `tensor` | tensor | The input tensor | Yes |

## Returns
Returns a new tensor containing the element-wise inverse hyperbolic cosine of the input tensor.

## Mathematical Definition
The inverse hyperbolic cosine function is defined as:
- acosh(x) = ln(x + sqrt(x² - 1))
- Domain: x ≥ 1
- Range: [0, +∞)

## Examples

### Basic Usage - Positional Syntax
```tcl
# Create a tensor with values suitable for acosh (>= 1)
set input [torch::ones {3 3}]
set result [torch::acosh $input]
puts "acosh result: $result"
```

### Basic Usage - Named Parameter Syntax
```tcl
# Using -input parameter
set input [torch::ones {2 4}]
set result [torch::acosh -input $input]
puts "acosh result: $result"

# Using -tensor parameter (alternative)
set input [torch::ones {3 2}]
set result [torch::acosh -tensor $input]
puts "acosh result: $result"
```

### Working with Different Tensor Shapes
```tcl
# 1D tensor
set tensor1d [torch::ones {5}]
set result1d [torch::acosh $tensor1d]

# 2D tensor  
set tensor2d [torch::ones {3 4}]
set result2d [torch::acosh -input $tensor2d]

# 3D tensor
set tensor3d [torch::ones {2 3 4}]
set result3d [torch::acosh -tensor $tensor3d]

# 4D tensor
set tensor4d [torch::ones {2 3 4 5}]
set result4d [torch::acosh $tensor4d]
```

### Mathematical Properties
```tcl
# acosh(1) = 0
set ones [torch::ones {3}]
set result [torch::acosh $ones]
# Result will be approximately zero

# acosh is the inverse of cosh for x >= 0
set x [torch::ones {1}]
set cosh_x [torch::cosh $x]  
set acosh_cosh_x [torch::acosh $cosh_x]
# acosh_cosh_x should approximately equal x
```

## Error Handling

### Missing Arguments
```tcl
# Error: No arguments provided
catch {torch::acosh} error
puts $error  ;# Usage: torch::acosh tensor | torch::acosh -input tensor

# Error: Missing parameter value  
catch {torch::acosh -input} error
puts $error  ;# Missing value for parameter
```

### Invalid Parameters
```tcl
# Error: Invalid tensor name
catch {torch::acosh nonexistent_tensor} error
puts $error  ;# Invalid tensor name

# Error: Unknown parameter
set input [torch::ones {2 2}]
catch {torch::acosh -invalid $input} error
puts $error  ;# Unknown parameter: -invalid. Valid parameters are: -input, -tensor
```

### Too Many Arguments
```tcl
# Error: Too many positional arguments
set input [torch::ones {2 2}]
catch {torch::acosh $input extra_arg} error
puts $error  ;# Usage: torch::acosh tensor
```

## Domain Considerations
The acosh function requires input values ≥ 1. For values < 1, the function will produce NaN or complex results. Ensure your input tensors contain appropriate values:

```tcl
# Safe input (values >= 1)
set safe_input [torch::ones {3}]  ;# All values are 1.0
set result [torch::acosh $safe_input]  ;# Works correctly

# Note: Be careful with tensors containing values < 1
# as they may produce NaN results
```

## Implementation Notes

### Backward Compatibility
- The original positional syntax `torch::acosh tensor` is fully supported
- Existing code will continue to work without modification
- The named parameter syntax is recommended for new code

### Parameter Flexibility
- Both `-input` and `-tensor` parameter names are supported
- This provides flexibility for different coding styles and contexts
- Both parameter names are functionally equivalent

### Performance
- The function operates element-wise on the entire tensor
- Memory usage scales with tensor size
- Computation is performed using efficient PyTorch backend operations

## Related Functions
- `torch::cosh` - Hyperbolic cosine (inverse operation)
- `torch::asinh` - Inverse hyperbolic sine
- `torch::atanh` - Inverse hyperbolic tangent
- `torch::sinh` - Hyperbolic sine
- `torch::tanh` - Hyperbolic tangent

## Migration Guide

### From Positional to Named Syntax
```tcl
# Old positional syntax
set result [torch::acosh $tensor]

# New named parameter syntax
set result [torch::acosh -input $tensor]
# or
set result [torch::acosh -tensor $tensor]
```

### Benefits of Named Syntax
- **Clarity**: Parameter purpose is explicit
- **Maintainability**: Code is self-documenting
- **Flexibility**: Parameters can be specified in any order (when extended)
- **Future-proof**: Easier to extend with additional optional parameters

## See Also
- [torch::cosh](cosh.md) - Hyperbolic cosine function
- [torch::asinh](asinh.md) - Inverse hyperbolic sine function  
- [torch::atanh](atanh.md) - Inverse hyperbolic tangent function
- [LibTorch TCL Extension API](../API.md) - Complete API reference 