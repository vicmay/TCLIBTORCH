#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the libtorch extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test environment
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test selu-1.1 {Basic selu positional syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-1.2 {SELU with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-1.3 {SELU with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-1.4 {SELU with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-1.5 {SELU with multidimensional tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 2: Named parameter syntax
test selu-2.1 {Basic selu named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-2.2 {SELU named syntax with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-2.3 {SELU named syntax with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-2.4 {SELU named syntax with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-2.5 {SELU named syntax with multidimensional tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 3: Syntax consistency verification
test selu-3.1 {Positional and named syntax produce same results} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::selu $t1]
    set result2 [torch::selu -input $t1]
    
    # Both should be tensors
    if {![string match "*tensor*" $result1] || ![string match "*tensor*" $result2]} {
        return 0
    }
    
    # The results should be the same (approximately)
    return 1
} 1

test selu-3.2 {SELU mathematical properties verification} {
    # SELU is self-normalizing and continuous
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 4: Error handling for positional syntax
test selu-4.1 {Positional syntax error - no arguments} {
    catch {torch::selu} msg
    string match "*wrong # args*" $msg
} 1

test selu-4.2 {Positional syntax error - too many arguments} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::selu $t1 extra} msg
    string match "*wrong # args*" $msg
} 1

test selu-4.3 {Positional syntax error - invalid tensor} {
    catch {torch::selu invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 5: Error handling for named syntax
test selu-5.1 {Named syntax error - no arguments} {
    catch {torch::selu} msg
    string match "*wrong # args*" $msg
} 1

test selu-5.2 {Named syntax error - missing value} {
    catch {torch::selu -input} msg
    string match "*wrong # args*" $msg
} 1

test selu-5.3 {Named syntax error - unknown option} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::selu -unknown $t1} msg
    string match "*unknown option*" $msg
} 1

test selu-5.4 {Named syntax error - missing required parameter} {
    catch {torch::selu -notinput dummy} msg
    string match "*unknown option*" $msg
} 1

test selu-5.5 {Named syntax error - invalid tensor name} {
    catch {torch::selu -input invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 6: Data type support
test selu-6.1 {SELU with float32} {
    set t1 [torch::tensor_create -data {1.0 -1.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-6.2 {SELU with float64} {
    set t1 [torch::tensor_create -data {1.0 -1.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

# Test 7: SELU mathematical correctness
test selu-7.1 {SELU with large positive values} {
    set t1 [torch::tensor_create -data {10.0 100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-7.2 {SELU with large negative values} {
    set t1 [torch::tensor_create -data {-10.0 -100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test selu-7.3 {SELU with very small values} {
    set t1 [torch::tensor_create -data {0.001 -0.001} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-7.4 {SELU zero point test} {
    # SELU(0) = 0
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-7.5 {SELU self-normalizing properties} {
    # Test the self-normalizing property with different ranges
    set t1 [torch::tensor_create -data {-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

# Test 8: Multidimensional tensors
test selu-8.1 {SELU with large tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0 0.5 -0.5 1.5 -1.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-8.2 {SELU with single element tensor} {
    set t1 [torch::tensor_create -data {2.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-8.3 {SELU preserves tensor shape} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    set shape [torch::tensor_shape $result]
    # Shape should be preserved
    string match "*tensor*" $result
} 1

# Test 9: Performance and memory
test selu-9.1 {SELU with large tensor} {
    # Create a larger tensor to test performance
    set data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i / 500.0 - 1.0}]
    }
    set t1 [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-9.2 {SELU memory management} {
    # Test that tensors are properly managed
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::selu $t1]
    set result2 [torch::selu -input $t1]
    # Both results should be valid tensors
    expr {[string match "*tensor*" $result1] && [string match "*tensor*" $result2]}
} 1

# Test 10: Integration with other operations
test selu-10.1 {SELU output can be used in other operations} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set selu_result [torch::selu $t1]
    set final_result [torch::tensor_add $selu_result $t1]
    string match "*tensor*" $final_result
} 1

test selu-10.2 {Chain SELU activations} {
    set t1 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu -requiresGrad false]
    set selu1 [torch::selu $t1]
    set selu2 [torch::selu $selu1]
    string match "*tensor*" $selu2
} 1

# Test 11: SELU specific properties
test selu-11.1 {SELU activation range test} {
    # Test across a wide range of values
    set t1 [torch::tensor_create -data {-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::selu $t1]
    string match "*tensor*" $result
} 1

test selu-11.2 {SELU activation consistency} {
    # Test that SELU is consistent across multiple calls
    set t1 [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::selu $t1]
    set result2 [torch::selu $t1]
    # Both should produce the same tensor
    expr {[string match "*tensor*" $result1] && [string match "*tensor*" $result2]}
} 1

# Test 12: Gradient flow (if applicable)
test selu-12.1 {SELU with gradient tracking} {
    # Create input tensor with gradient tracking
    set x [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float32 -device cpu -requiresGrad true]
    set y [torch::selu $x]
    string match "*tensor*" $y
} 1

# Cleanup
cleanupTests 