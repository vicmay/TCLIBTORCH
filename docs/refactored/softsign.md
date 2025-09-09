# torch::softsign

## Description

The `torch::softsign` command applies the Softsign activation function to a tensor. The Softsign function is defined as:

```
softsign(x) = x / (1 + |x|)
```

This activation function has several interesting properties:
- Bounded output: The function maps inputs to the range (-1, 1)
- Smooth and continuous everywhere, including at x = 0
- Preserves the sign of the input
- Approaches ±1 asymptotically for large magnitude inputs
- Has a gentler saturation compared to tanh

## Syntax

### Positional Syntax (Legacy)
```tcl
torch::softsign tensor
```

### Named Parameter Syntax
```tcl
torch::softsign -input tensor
# or
torch::softsign -tensor tensor
```

### CamelCase Alias
The command also provides a camelCase alias:
```tcl
torch::softSign tensor
# or
torch::softSign -input tensor
```

## Arguments

- `tensor` (positional) or `-input`/`-tensor` (named): The input tensor to apply the Softsign function to.

## Return Value

Returns a new tensor of the same shape as the input tensor, containing the result of applying the Softsign function element-wise.

## Examples

### Basic Usage
```tcl
# Create a simple tensor
set t [torch::tensor [list 1.0 -1.0 2.0 -2.0]]

# Apply softsign using positional syntax
set result1 [torch::softsign $t]
;# Result: tensor([0.5000, -0.5000, 0.6667, -0.6667])

# Apply softsign using named parameter syntax
set result2 [torch::softsign -input $t]
;# Same result as above

# Using camelCase alias
set result3 [torch::softSign $t]
;# Same result as above
```

### Working with 2D Tensors
```tcl
# Create a 2D tensor
set t [torch::tensor [list [list 1.0 -1.0] [list 2.0 -2.0]]]

# Apply softsign
set result [torch::softsign $t]
;# Result: tensor([[0.5000, -0.5000],
;#                 [0.6667, -0.6667]])
```

### Large Values Example
```tcl
# Softsign with large values demonstrates asymptotic behavior
set t [torch::tensor [list 100.0 -100.0]]
set result [torch::softsign $t]
;# Result: tensor([0.9901, -0.9901])
```

## Mathematical Properties

1. **Boundedness**: The output is always in the range (-1, 1)
   ```tcl
   set t [torch::tensor [list 1000.0 -1000.0 10000.0 -10000.0]]
   set result [torch::softsign $t]
   ;# All values will be close to but not equal to ±1
   ```

2. **Sign Preservation**: The function preserves the sign of the input
   ```tcl
   set t [torch::tensor [list 1.0 -1.0 2.0 -2.0 0.0]]
   set result [torch::softsign $t]
   ;# Positive inputs remain positive, negative remain negative, zero remains zero
   ```

3. **Continuity at Zero**: The function is smooth and continuous at x = 0
   ```tcl
   set t [torch::tensor [list -0.001 0.0 0.001]]
   set result [torch::softsign $t]
   ;# Shows smooth transition through zero
   ```

## Error Handling

The command will return an error in the following cases:

1. Missing tensor argument:
   ```tcl
   torch::softsign
   ;# Error: wrong # args: should be "torch::softsign tensor | -input tensor"
   ```

2. Invalid tensor name:
   ```tcl
   torch::softsign invalid_tensor
   ;# Error: Invalid tensor name
   ```

3. Invalid named parameter:
   ```tcl
   torch::softsign -invalid $t
   ;# Error: Unknown parameter: -invalid
   ```

4. Missing value for named parameter:
   ```tcl
   torch::softsign -input
   ;# Error: Missing value for parameter
   ```

## See Also

- `torch::softplus` - Softplus activation function
- `torch::tanh` - Hyperbolic tangent activation function
- `torch::sigmoid` - Sigmoid activation function 