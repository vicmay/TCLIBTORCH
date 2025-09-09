# torch::relu6

Applies the ReLU6 (Rectified Linear Unit 6) function element-wise to the input tensor, clipping values to the range [0, 6].

## Syntax

### Positional Parameters (Backward Compatible)
```tcl
torch::relu6 tensor
```

### Named Parameters (New Syntax)
```tcl
torch::relu6 -input tensor
```

## Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `tensor` / `-input` | string | Name of the input tensor | Required |

## Returns

Returns a string handle to the tensor containing the result of applying the ReLU6 function: `relu6(x) = max(0, min(6, x))`.

## Mathematical Description

The ReLU6 function is defined as:
```
relu6(x) = max(0, min(6, x))
```

This can be written piecewise as:
- For x < 0: relu6(x) = 0
- For 0 ≤ x ≤ 6: relu6(x) = x (identity)
- For x > 6: relu6(x) = 6

ReLU6 is a bounded version of the standard ReLU activation function that clips the output to the range [0, 6]. This clipping helps with numerical stability and is particularly useful in quantized neural networks and mobile/embedded applications.

## Examples

### Basic Usage (Positional Syntax)
```tcl
# Create a tensor with various values
set input [torch::tensor_create -data {-3.0 -1.0 0.0 2.0 5.0 8.0 12.0} -dtype float32 -device cpu]

# Apply ReLU6
set result [torch::relu6 $input]
# Result will be {0.0, 0.0, 0.0, 2.0, 5.0, 6.0, 6.0}

# Check the results
puts "Input shape: [torch::tensor_shape $input]"
puts "Output shape: [torch::tensor_shape $result]"
```

### Named Parameters (New Syntax)
```tcl
# Create input tensor
set input [torch::full {3 3} 8.0]

# Apply ReLU6 with named parameters
set result [torch::relu6 -input $input]

# All values above 6 will be clipped to 6
puts "Shape: [torch::tensor_shape $result]"
puts "Data type: [torch::tensor_dtype $result]"
```

### Range Verification Examples
```tcl
# Test negative values (should become 0)
set negative_input [torch::full {1} -5.0]
set result [torch::relu6 $negative_input]
set value [torch::tensor_item $result]
puts "relu6(-5.0) = $value"  # Should be 0.0

# Test values in range [0, 6] (should remain unchanged)
set in_range_input [torch::full {1} 3.5]
set result [torch::relu6 $in_range_input]
set value [torch::tensor_item $result]
puts "relu6(3.5) = $value"  # Should be 3.5

# Test values above 6 (should become 6)
set above_range_input [torch::full {1} 10.0]
set result [torch::relu6 $above_range_input]
set value [torch::tensor_item $result]
puts "relu6(10.0) = $value"  # Should be 6.0

# Test boundary values
set boundary_low [torch::full {1} 0.0]
set boundary_high [torch::full {1} 6.0]
set result_low [torch::relu6 $boundary_low]
set result_high [torch::relu6 $boundary_high]
puts "relu6(0.0) = [torch::tensor_item $result_low]"    # Should be 0.0
puts "relu6(6.0) = [torch::tensor_item $result_high]"   # Should be 6.0
```

### Neural Network Usage
```tcl
# ReLU6 is commonly used in mobile and quantized neural networks
set conv_output [torch::randn {32 64 28 28}]
set activated [torch::relu6 $conv_output]

# For quantized models where bounded activations are needed
set features [torch::randn {128 256}]
set bounded_features [torch::relu6 $features]

# Comparison with unbounded ReLU for stability
set layer_output [torch::randn {64 128}]
set unbounded [torch::relu $layer_output]    # Standard ReLU (unbounded)
set bounded [torch::relu6 $layer_output]     # ReLU6 (bounded to [0,6])
```

### Mathematical Properties Demonstration
```tcl
# Demonstrate clipping behavior across different ranges
set test_values [torch::tensor_create -data {-10.0 -1.0 0.0 1.0 3.0 6.0 7.0 15.0} -dtype float32 -device cpu]
set clipped_values [torch::relu6 $test_values]

puts "Original: {-10.0, -1.0, 0.0, 1.0, 3.0, 6.0, 7.0, 15.0}"
puts "ReLU6:    {0.0, 0.0, 0.0, 1.0, 3.0, 6.0, 6.0, 6.0}"

# Verify equivalence to manual clipping
set manual_clipped [torch::clamp $test_values 0.0 6.0]
# manual_clipped should be identical to clipped_values
```

## Error Handling

### Invalid Tensor Name
```tcl
catch {torch::relu6 invalid_tensor} error
puts $error  # "Invalid tensor name"
```

### Missing Arguments
```tcl
catch {torch::relu6} error
puts $error  # Usage information
```

### Unknown Parameters
```tcl
set input [torch::ones {2}]
catch {torch::relu6 -invalid_param $input} error
puts $error  # "Unknown parameter: -invalid_param"
```

### Too Many Arguments
```tcl
set input [torch::ones {2}]
catch {torch::relu6 $input extra_argument} error
puts $error  # Usage information
```

## Data Type Support

ReLU6 preserves the input tensor's data type:

```tcl
# Float32 input
set input_f32 [torch::full {2} 8.0]
set result_f32 [torch::relu6 $input_f32]
puts [torch::tensor_dtype $result_f32]  # Float

# Float64 input
set input_f64 [torch::full {2} 8.0 float64]
set result_f64 [torch::relu6 $input_f64]
puts [torch::tensor_dtype $result_f64]  # Double

# Integer input (will be converted to float)
set input_int [torch::full {2} 8 int32]
set result_int [torch::relu6 $input_int]
puts [torch::tensor_dtype $result_int]  # Int (but values are clipped)
```

## Performance Characteristics

- **Computational Efficiency**: Fast element-wise clamp operation
- **Memory Usage**: Same as input tensor
- **Numerical Stability**: Bounded output prevents overflow issues
- **Quantization Friendly**: Fixed range [0, 6] is ideal for quantization

## Migration Guide

### From Positional to Named Parameters

**Before (Positional):**
```tcl
set result [torch::relu6 $input_tensor]
```

**After (Named Parameters):**
```tcl
set result [torch::relu6 -input $input_tensor]
```

### Migrating from Standard ReLU

**Standard ReLU (unbounded):**
```tcl
set result [torch::relu $input_tensor]
```

**ReLU6 (bounded to [0, 6]):**
```tcl
set result [torch::relu6 $input_tensor]
```

### Manual Clipping to ReLU6

**Manual clipping:**
```tcl
set temp [torch::relu $input_tensor]
set result [torch::clamp $temp 0.0 6.0]
```

**Using ReLU6 directly:**
```tcl
set result [torch::relu6 $input_tensor]
```

## Comparison with Related Functions

| Function | Range | Use Case |
|----------|-------|----------|
| `relu` | [0, +∞) | Standard activation, can have very large values |
| `relu6` | [0, 6] | Mobile/quantized networks, numerical stability |
| `leaky_relu` | (-∞, +∞) | Addresses dying ReLU problem |
| `hardtanh` | [min, max] | Configurable clipping range |
| `clamp` | [min, max] | General-purpose value clipping |

## Common Use Cases

1. **Mobile Neural Networks**: Bounded activations for efficient quantization
2. **Quantized Models**: Fixed range simplifies quantization schemes
3. **Embedded Systems**: Reduced dynamic range for hardware optimization
4. **Numerical Stability**: Prevents activation values from becoming too large
5. **Memory-Constrained Environments**: Bounded range allows for better memory planning
6. **MobileNet Architecture**: Originally introduced and widely used in MobileNets

## Advanced Examples

### Quantization-Aware Training Simulation
```tcl
# Simulate quantization effects
set activations [torch::randn {64 128}]
set relu6_output [torch::relu6 $activations]

# The bounded range [0, 6] makes quantization more predictable
set quantization_scale [expr {6.0 / 255.0}]  # 8-bit quantization
puts "Quantization scale: $quantization_scale"
```

### Comparison with Different Clipping Ranges
```tcl
# Compare different activation functions
set input [torch::randn {32 64}]

set relu_result [torch::relu $input]        # Unbounded
set relu6_result [torch::relu6 $input]      # Bounded to [0, 6]
set hardtanh_result [torch::hardtanh $input -1.0 1.0]  # Bounded to [-1, 1]

puts "ReLU shape: [torch::tensor_shape $relu_result]"
puts "ReLU6 shape: [torch::tensor_shape $relu6_result]"
puts "HardTanh shape: [torch::tensor_shape $hardtanh_result]"
```

### Mobile Network Activation Pattern
```tcl
# Typical mobile network layer with ReLU6
proc mobile_conv_block {input channels} {
    # Depthwise convolution (simulated)
    set conv_out [torch::conv2d $input $channels]
    
    # Batch normalization (simulated)
    set bn_out [torch::batch_norm2d $conv_out]
    
    # ReLU6 activation
    set activated [torch::relu6 $bn_out]
    
    return $activated
}

set input_image [torch::randn {1 32 224 224}]
set output [mobile_conv_block $input_image 64]
puts "Mobile block output shape: [torch::tensor_shape $output]"
```

### Range Analysis
```tcl
# Analyze activation ranges before and after ReLU6
proc analyze_activations {tensor name} {
    set min_val [torch::min $tensor]
    set max_val [torch::max $tensor]
    set mean_val [torch::mean $tensor]
    
    puts "$name statistics:"
    puts "  Min: [torch::tensor_item $min_val]"
    puts "  Max: [torch::tensor_item $max_val]"
    puts "  Mean: [torch::tensor_item $mean_val]"
}

set random_input [torch::randn {1000}]
analyze_activations $random_input "Before ReLU6"

set relu6_output [torch::relu6 $random_input]
analyze_activations $relu6_output "After ReLU6"
```

## Implementation Notes

- ReLU6 was introduced to address numerical stability issues in mobile and quantized neural networks
- The choice of 6 as the upper bound is somewhat arbitrary but has proven effective in practice
- The bounded range makes the function more amenable to fixed-point arithmetic
- ReLU6 maintains the sparsity benefits of ReLU while preventing extremely large activations
- The function is piecewise linear, making it computationally efficient

## Choosing When to Use ReLU6

### Use ReLU6 when:
- **Deploying to mobile devices** where quantization is important
- **Numerical stability** is a concern with very large activations
- **Memory bandwidth** is limited and bounded ranges help optimization
- **Quantization-aware training** is being performed
- **Following proven architectures** like MobileNet, EfficientNet variants

### Use standard ReLU when:
- **Maximum expressiveness** is needed
- **Large dynamic ranges** are beneficial for the task
- **Desktop/server deployment** where quantization is not a priority
- **Research settings** where architectural flexibility is important

## Hardware Considerations

ReLU6 is particularly well-suited for:
- **Mobile GPUs** with limited precision
- **Neural Processing Units (NPUs)** designed for quantized inference
- **FPGA implementations** where bounded ranges simplify hardware design
- **Edge AI accelerators** optimized for efficient quantized computation

## See Also

- [`torch::relu`](relu.md) - Standard unbounded ReLU
- [`torch::leaky_relu`](leaky_relu.md) - ReLU with small negative slope
- [`torch::hardtanh`](hardtanh.md) - Configurable clipping activation
- [`torch::clamp`](clamp.md) - General tensor value clipping
- [`torch::sigmoid`](sigmoid.md) - Bounded activation in [0, 1] range 