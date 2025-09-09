#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test randn-1.1 {Basic positional syntax - 2D tensor} {
    set result [torch::randn {2 3}]
    llength [split $result]
} {1}

test randn-1.2 {Basic positional syntax - 1D tensor} {
    set result [torch::randn {5}]
    llength [split $result]
} {1}

test randn-1.3 {Basic positional syntax - scalar tensor} {
    set result [torch::randn {}]
    llength [split $result]
} {1}

# Test 2: Positional syntax with device
test randn-2.1 {Positional syntax with CPU device} {
    set result [torch::randn {2 2} cpu]
    llength [split $result]
} {1}

test randn-2.2 {Positional syntax with device and dtype} {
    set result [torch::randn {3 3} cpu float32]
    llength [split $result]
} {1}

# Test 3: Named parameter syntax
test randn-3.1 {Named parameter syntax - basic} {
    set result [torch::randn -shape {2 3}]
    llength [split $result]
} {1}

test randn-3.2 {Named parameter syntax with device} {
    set result [torch::randn -shape {2 2} -device cpu]
    llength [split $result]
} {1}

test randn-3.3 {Named parameter syntax with dtype} {
    set result [torch::randn -shape {2 2} -dtype float32]
    llength [split $result]
} {1}

test randn-3.4 {Named parameter syntax - all parameters} {
    set result [torch::randn -shape {3 2} -device cpu -dtype float32]
    llength [split $result]
} {1}

test randn-3.5 {Named parameter syntax - parameter order shouldn't matter} {
    set result [torch::randn -dtype float32 -shape {2 2} -device cpu]
    llength [split $result]
} {1}

# Test 4: Different data types
test randn-4.1 {Test with float32 dtype} {
    set result [torch::randn {2 2} cpu float32]
    llength [split $result]
} {1}

test randn-4.2 {Test with float64 dtype} {
    set result [torch::randn {2 2} cpu float64]
    llength [split $result]
} {1}

test randn-4.3 {Named syntax with float64} {
    set result [torch::randn -shape {2 2} -dtype float64]
    llength [split $result]
} {1}

# Test 5: Different tensor shapes
test randn-5.1 {1D tensor of size 10} {
    set result [torch::randn {10}]
    llength [split $result]
} {1}

test randn-5.2 {3D tensor} {
    set result [torch::randn {2 3 4}]
    llength [split $result]
} {1}

test randn-5.3 {4D tensor} {
    set result [torch::randn {2 2 2 2}]
    llength [split $result]
} {1}

test randn-5.4 {Large tensor} {
    set result [torch::randn {100}]
    llength [split $result]
} {1}

# Test 6: Statistical properties validation (normal distribution)
test randn-6.1 {Values should have reasonable range (mostly within [-3, 3])} {
    set tensor [torch::randn {1000}]
    set max_val [torch::tensor_max $tensor]
    set min_val [torch::tensor_min $tensor]
    
    # Get the actual values
    set max_item [torch::tensor_item $max_val]
    set min_item [torch::tensor_item $min_val]
    
    # For normal distribution, most values should be within [-3, 3] (99.7% rule)
    # But we'll be lenient for test stability
    expr {$min_item > -6.0 && $max_item < 6.0}
} {1}

test randn-6.2 {Mean should be close to zero for large sample} {
    set tensor [torch::randn {10000}]
    set mean_tensor [torch::tensor_mean $tensor]
    set mean_val [torch::tensor_item $mean_tensor]
    
    # Mean should be close to 0, allow some variance
    expr {abs($mean_val) < 0.1}
} {1}

# Test 7: Tensor properties validation
test randn-7.1 {Check tensor shape} {
    set tensor [torch::randn {3 4}]
    torch::tensor_shape $tensor
} {3 4}

test randn-7.2 {Check tensor dtype} {
    set tensor [torch::randn {2 2} cpu float32]
    torch::tensor_dtype $tensor
} {Float32}

test randn-7.3 {Check tensor device} {
    set tensor [torch::randn {2 2} cpu]
    set device_info [torch::tensor_device $tensor]
    string match "*device=cpu*" $device_info
} {1}

test randn-7.4 {Check tensor numel} {
    set tensor [torch::randn {3 4}]
    torch::tensor_numel $tensor
} {12}

# Test 8: Error handling - invalid arguments
test randn-8.1 {Error on no arguments} {
    catch {torch::randn} error_msg
    string match "*shape*" $error_msg
} {1}

test randn-8.2 {Invalid device still creates tensor (current behavior)} {
    set result [torch::randn {2 2} invalid_device]
    string match "tensor*" $result
} {1}

test randn-8.3 {Invalid dtype throws error} {
    catch {torch::randn {2 2} cpu invalid_dtype} error_msg
    string match "*Unknown scalar type: invalid_dtype*" $error_msg
} {1}

test randn-8.4 {Error on missing shape parameter in named syntax} {
    catch {torch::randn -device cpu} error_msg
    string match "*shape*" $error_msg
} {1}

test randn-8.5 {Error on unknown parameter} {
    catch {torch::randn -shape {2 2} -invalid_param value} error_msg
    string match "*Unknown parameter*" $error_msg
} {1}

test randn-8.6 {Error on missing value for parameter} {
    catch {torch::randn -shape} error_msg
    string match "*Missing value*" $error_msg
} {1}

# Test 9: Edge cases
test randn-9.1 {Scalar tensor (empty shape)} {
    set result [torch::randn {}]
    torch::tensor_shape $result
} {}

test randn-9.2 {Single element tensor} {
    set tensor [torch::randn {1}]
    torch::tensor_numel $tensor
} {1}

test randn-9.3 {Very small tensor} {
    set tensor [torch::randn {1 1}]
    list [torch::tensor_numel $tensor] [torch::tensor_shape $tensor]
} {1 {1 1}}

# Test 10: Syntax consistency - both syntaxes should work
test randn-10.1 {Positional and named syntax give same result type} {
    set pos_tensor [torch::randn {3 3} cpu float32]
    set named_tensor [torch::randn -shape {3 3} -device cpu -dtype float32]
    
    set pos_shape [torch::tensor_shape $pos_tensor]
    set named_shape [torch::tensor_shape $named_tensor]
    set pos_dtype [torch::tensor_dtype $pos_tensor]
    set named_dtype [torch::tensor_dtype $named_tensor]
    
    list [expr {$pos_shape eq $named_shape}] [expr {$pos_dtype eq $named_dtype}]
} {1 1}

# Test 11: Memory cleanup verification
test randn-11.1 {Multiple tensor creation doesn't crash} {
    for {set i 0} {$i < 10} {incr i} {
        set tensor [torch::randn {10 10}]
    }
    string length $tensor
} {8}

# Test 12: Different shape specifications
test randn-12.1 {Single dimension as list} {
    set tensor [torch::randn {5}]
    torch::tensor_shape $tensor
} {5}

test randn-12.2 {Multiple dimensions} {
    set tensor [torch::randn {2 3 4 5}]
    torch::tensor_shape $tensor
} {2 3 4 5}

# Test 13: Comparison with uniform distribution (randn vs rand)
test randn-13.1 {randn and rand produce different distributions} {
    set normal_tensor [torch::randn {1000}]
    set uniform_tensor [torch::rand {1000}]
    
    # Get some statistics
    set normal_max [torch::tensor_item [torch::tensor_max $normal_tensor]]
    set uniform_max [torch::tensor_item [torch::tensor_max $uniform_tensor]]
    set normal_min [torch::tensor_item [torch::tensor_min $normal_tensor]]
    set uniform_min [torch::tensor_item [torch::tensor_min $uniform_tensor]]
    
    # Normal distribution should have wider range than uniform [0,1)
    # Uniform min should be >= 0, normal min should be < 0 (usually)
    expr {$uniform_min >= 0.0 && $uniform_max < 1.0 && $normal_min < 0.0}
} {1}

cleanupTests 