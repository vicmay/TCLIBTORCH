# `torch::glu`

> Gated Linear Unit (GLU) activation function.
>
> The GLU activation function splits the input tensor along the last dimension into two equal parts and applies:
>
> \[ \text{glu}(x) = \text{first\_half}(x) \odot \sigma(\text{second\_half}(x)) \]
>
> where \( \sigma \) is the sigmoid function and \( \odot \) denotes element-wise multiplication.
>
> GLU is commonly used in language modeling and transformer architectures where gating mechanisms help control information flow.

---

## 🆕 Named-parameter Syntax (Recommended)

```tcl
# Usage: torch::glu -input tensorHandle
set result [torch::glu -input $tensor]
```

### Parameters
| Name  | Type   | Required | Description                  |
|-------|--------|----------|------------------------------|
| `-input` | tensor handle | ✅ Yes | Input tensor to apply GLU (last dimension must be even) |

### Alternative Parameter Names
- `-tensor` - Alternative name for the input tensor parameter

### Example
```tcl
set x [torch::randn {3 8}]  ;# Last dimension must be even
set y [torch::glu -input $x]  ;# Output shape: {3 4}
```

---

## ♻️ Positional Syntax (Backward-compatible)

```tcl
# Usage: torch::glu tensorHandle
set result [torch::glu $tensor]
```

This form is preserved for full backward compatibility with existing scripts.

### Example
```tcl
set x [torch::ones {2 6}]
set y [torch::glu $x]  ;# Output shape: {2 3}
```

---

## 🔀 camelCase Alias

`torch::glu` is already in camelCase format, therefore no additional alias is needed.

---

## 📐 Mathematical Properties

### Input Requirements
- The last dimension of the input tensor **must be even**
- GLU splits this dimension in half for the gating mechanism

### Output Shape
- All dimensions remain the same except the last dimension is halved
- Input shape: `{d1, d2, ..., dn}` → Output shape: `{d1, d2, ..., dn/2}`

### Examples
```tcl
# 2D tensor
set x [torch::randn {3 8}]
set y [torch::glu $x]  ;# Shape: {3 4}

# 3D tensor  
set x [torch::randn {2 5 6}]
set y [torch::glu $x]  ;# Shape: {2 5 3}

# 1D tensor
set x [torch::randn {10}]
set y [torch::glu $x]  ;# Shape: {5}
```

---

## ✅ Notes

* Both syntaxes are fully equivalent and yield identical results.
* The output tensor has the same dtype and device as the input.
* Gradients are propagated automatically if `requires_grad` is set on the input tensor.
* GLU is particularly effective in language models and transformer architectures.
* The gating mechanism allows the model to selectively pass information through the network.

---

## ⚠️ Error Handling

| Error Condition | Message |
|-----------------|---------|
| Missing arguments | `wrong # args` |
| Missing `-input` value (named syntax) | `Missing value for option` |
| Unknown parameter | `Unknown parameter: -foo` |
| Invalid tensor handle | `Invalid tensor name` |
| Odd last dimension | Runtime error (PyTorch requirement) |

### Common Errors

#### Odd Last Dimension
```tcl
set x [torch::randn {3 5}]  ;# Last dimension is odd (5)
catch {torch::glu $x} error
;# Error: GLU requires even-sized last dimension
```

**Solution**: Ensure the last dimension of your input tensor is even.

---

## 🧮 Usage Examples

### Basic GLU Application
```tcl
# Create input tensor with even last dimension
set input [torch::randn {4 8}]

# Apply GLU - both syntaxes equivalent
set output1 [torch::glu $input]
set output2 [torch::glu -input $input]

puts "Input shape: [torch::tensor_shape $input]"    ;# {4 8}
puts "Output shape: [torch::tensor_shape $output1]"  ;# {4 4}
```

### GLU in Neural Network Layer
```tcl
# Feedforward layer with GLU activation
proc glu_feedforward {input hidden_size} {
    # Linear transformation to 2 * hidden_size
    set expanded [torch::linear $input -inFeatures [lindex [torch::tensor_shape $input] end] -outFeatures [expr {$hidden_size * 2}]]
    
    # Apply GLU to gate the information
    set gated [torch::glu $expanded]
    
    return $gated
}

# Usage
set x [torch::randn {32 512}]  ;# Batch of 32, features 512
set output [glu_feedforward $x 256]  ;# Output: {32 256}
```

### GLU with Different Data Types
```tcl
# GLU with float32
set x32 [torch::randn {3 6} -dtype float32]
set y32 [torch::glu -input $x32]

# GLU with float64 (double)
set x64 [torch::randn {3 6} -dtype float64]
set y64 [torch::glu -input $x64]
```

### Batch Processing with GLU
```tcl
# Process multiple sequences
set batch_size 16
set seq_len 128
set feature_dim 1024

set sequences [torch::randn [list $batch_size $seq_len [expr {$feature_dim * 2}]]]
set glu_output [torch::glu -input $sequences]

puts "Input: [torch::tensor_shape $sequences]"      ;# {16 128 2048}  
puts "Output: [torch::tensor_shape $glu_output]"    ;# {16 128 1024}
```

---

## 🔬 Mathematical Verification

### Manual GLU Implementation
```tcl
# Verify GLU behavior manually
set input [torch::randn {2 8}]

# Split manually
set first_half [torch::tensor_slice $input -dim 1 -start 0 -end 4]
set second_half [torch::tensor_slice $input -dim 1 -start 4 -end 8]

# Apply sigmoid to second half
set gated [torch::sigmoid $second_half]

# Element-wise multiplication
set manual_result [torch::tensor_mul $first_half $gated]

# Compare with GLU
set glu_result [torch::glu $input]

# Shapes should match
puts "Manual: [torch::tensor_shape $manual_result]"
puts "GLU: [torch::tensor_shape $glu_result]"
```

---

## 🧪 Tests

Comprehensive tests covering both syntaxes, error handling, mathematical properties, and edge cases reside in `tests/refactored/glu_test.tcl`.

### Test Categories
- **Positional syntax**: Backward compatibility tests
- **Named parameter syntax**: Modern API tests  
- **Error handling**: Invalid inputs and parameters
- **Mathematical correctness**: Dimension halving and consistency
- **Edge cases**: Minimum tensor sizes and data types

---

## 🔄 Migration Guide

Old scripts using positional syntax continue to work:

```tcl
# Legacy positional
set y [torch::glu $x]
```

For new code, prefer the more explicit named-parameter style:

```tcl
# Modern named parameters
set y [torch::glu -input $x]
```

Both forms are equivalent and yield identical results.

---

## 🔗 Related Commands

- [`torch::linear`](linear.md) - Linear transformation often used before GLU
- [`torch::sigmoid`](sigmoid.md) - Sigmoid activation used internally by GLU
- [`torch::gelu`](gelu.md) - Alternative activation function
- [`torch::relu`](relu.md) - Another common activation function
- [`torch::tensor_mul`](tensor_mul.md) - Element-wise multiplication used in GLU 