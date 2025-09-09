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
test rand-1.1 {Basic positional syntax - 2D tensor} {
    set result [torch::rand {2 3}]
    llength [split $result]
} {1}

test rand-1.2 {Basic positional syntax - 1D tensor} {
    set result [torch::rand {5}]
    llength [split $result]
} {1}

test rand-1.3 {Basic positional syntax - scalar tensor} {
    set result [torch::rand {}]
    llength [split $result]
} {1}

# Test 2: Positional syntax with device
test rand-2.1 {Positional syntax with CPU device} {
    set result [torch::rand {2 2} cpu]
    llength [split $result]
} {1}

test rand-2.2 {Positional syntax with device and dtype} {
    set result [torch::rand {3 3} cpu float32]
    llength [split $result]
} {1}

# Test 3: Named parameter syntax
test rand-3.1 {Named parameter syntax - basic} {
    set result [torch::rand -shape {2 3}]
    llength [split $result]
} {1}

test rand-3.2 {Named parameter syntax with device} {
    set result [torch::rand -shape {2 2} -device cpu]
    llength [split $result]
} {1}

test rand-3.3 {Named parameter syntax with dtype} {
    set result [torch::rand -shape {2 2} -dtype float32]
    llength [split $result]
} {1}

test rand-3.4 {Named parameter syntax - all parameters} {
    set result [torch::rand -shape {3 2} -device cpu -dtype float32]
    llength [split $result]
} {1}

test rand-3.5 {Named parameter syntax - parameter order shouldn't matter} {
    set result [torch::rand -dtype float32 -shape {2 2} -device cpu]
    llength [split $result]
} {1}

# Test 4: Different data types
test rand-4.1 {Test with float32 dtype} {
    set result [torch::rand {2 2} cpu float32]
    llength [split $result]
} {1}

test rand-4.2 {Test with float64 dtype} {
    set result [torch::rand {2 2} cpu float64]
    llength [split $result]
} {1}

test rand-4.3 {Named syntax with float64} {
    set result [torch::rand -shape {2 2} -dtype float64]
    llength [split $result]
} {1}

# Test 5: Different tensor shapes
test rand-5.1 {1D tensor of size 10} {
    set result [torch::rand {10}]
    llength [split $result]
} {1}

test rand-5.2 {3D tensor} {
    set result [torch::rand {2 3 4}]
    llength [split $result]
} {1}

test rand-5.3 {4D tensor} {
    set result [torch::rand {2 2 2 2}]
    llength [split $result]
} {1}

test rand-5.4 {Large tensor} {
    set result [torch::rand {100}]
    llength [split $result]
} {1}

# Test 6: Value range validation (random values should be in [0, 1))
test rand-6.1 {Values should be in range [0, 1)} {
    set tensor [torch::rand {100}]
    set max_val [torch::tensor_max $tensor]
    set min_val [torch::tensor_min $tensor]
    
    # Get the actual values
    set max_item [torch::tensor_item $max_val]
    set min_item [torch::tensor_item $min_val]
    
    # Check range: min >= 0 and max < 1
    expr {$min_item >= 0.0 && $max_item < 1.0}
} {1}

# Test 7: Tensor properties validation
test rand-7.1 {Check tensor shape} {
    set tensor [torch::rand {3 4}]
    torch::tensor_shape $tensor
} {3 4}

test rand-7.2 {Check tensor dtype} {
    set tensor [torch::rand {2 2} cpu float32]
    torch::tensor_dtype $tensor
} {Float32}

test rand-7.3 {Check tensor device} {
    set tensor [torch::rand {2 2} cpu]
    set device_info [torch::tensor_device $tensor]
    string match "*device=cpu*" $device_info
} {1}

test rand-7.4 {Check tensor numel} {
    set tensor [torch::rand {3 4}]
    torch::tensor_numel $tensor
} {12}

# Test 8: Error handling - invalid arguments
test rand-8.1 {Error on no arguments} {
    catch {torch::rand} error_msg
    string match "*shape*" $error_msg
} {1}

test rand-8.2 {Invalid device still creates tensor (current behavior)} {
    set result [torch::rand {2 2} invalid_device]
    string match "tensor*" $result
} {1}

test rand-8.3 {Invalid dtype throws error} {
    catch {torch::rand {2 2} cpu invalid_dtype} error_msg
    string match "*Unknown scalar type: invalid_dtype*" $error_msg
} {1}

test rand-8.4 {Error on missing shape parameter in named syntax} {
    catch {torch::rand -device cpu} error_msg
    string match "*shape*" $error_msg
} {1}

test rand-8.5 {Error on unknown parameter} {
    catch {torch::rand -shape {2 2} -invalid_param value} error_msg
    string match "*Unknown parameter*" $error_msg
} {1}

test rand-8.6 {Error on missing value for parameter} {
    catch {torch::rand -shape} error_msg
    string match "*Missing value*" $error_msg
} {1}

# Test 9: Edge cases
test rand-9.1 {Scalar tensor (empty shape)} {
    set result [torch::rand {}]
    torch::tensor_shape $result
} {}

test rand-9.2 {Single element tensor} {
    set tensor [torch::rand {1}]
    torch::tensor_numel $tensor
} {1}

test rand-9.3 {Very small tensor} {
    set tensor [torch::rand {1 1}]
    list [torch::tensor_numel $tensor] [torch::tensor_shape $tensor]
} {1 {1 1}}

# Test 10: Syntax consistency - both syntaxes should work
test rand-10.1 {Positional and named syntax give same result type} {
    set pos_tensor [torch::rand {3 3} cpu float32]
    set named_tensor [torch::rand -shape {3 3} -device cpu -dtype float32]
    
    set pos_shape [torch::tensor_shape $pos_tensor]
    set named_shape [torch::tensor_shape $named_tensor]
    set pos_dtype [torch::tensor_dtype $pos_tensor]
    set named_dtype [torch::tensor_dtype $named_tensor]
    
    list [expr {$pos_shape eq $named_shape}] [expr {$pos_dtype eq $named_dtype}]
} {1 1}

# Test 11: Memory cleanup verification
test rand-11.1 {Multiple tensor creation doesn't crash} {
    for {set i 0} {$i < 10} {incr i} {
        set tensor [torch::rand {10 10}]
    }
    string length $tensor
} {8}

# Test 12: Different shape specifications
test rand-12.1 {Single dimension as list} {
    set tensor [torch::rand {5}]
    torch::tensor_shape $tensor
} {5}

test rand-12.2 {Multiple dimensions} {
    set tensor [torch::rand {2 3 4 5}]
    torch::tensor_shape $tensor
} {2 3 4 5}

cleanupTests 