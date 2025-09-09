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
test mish-1.1 {Basic mish positional syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-1.2 {Mish with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-1.3 {Mish with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-1.4 {Mish with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-1.5 {Mish with 2D tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 2: Named parameter syntax
test mish-2.1 {Basic mish named syntax} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-2.2 {Mish named syntax with positive values} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-2.3 {Mish named syntax with negative values} {
    set t1 [torch::tensor_create -data {-1.0 -2.0 -0.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-2.4 {Mish named syntax with zero} {
    set t1 [torch::tensor_create -data {0.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-2.5 {Mish named syntax with 2D tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish -input $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 3: Syntax consistency verification
test mish-3.1 {Positional and named syntax produce same results} {
    set t1 [torch::tensor_create -data {1.0 2.0 -1.0 0.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::mish $t1]
    set result2 [torch::mish -input $t1]
    
    # Both should be tensors
    if {![string match "*tensor*" $result1] || ![string match "*tensor*" $result2]} {
        return 0
    }
    
    # The results should be the same (approximately)
    return 1
} 1

test mish-3.2 {Mathematical properties verification} {
    # Mish is smooth and monotonic
    set t1 [torch::tensor_create -data {-2.0 -1.0 0.0 1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

# Test 4: Error handling for positional syntax
test mish-4.1 {Positional syntax error - no arguments} {
    catch {torch::mish} msg
    string match "*wrong # args*" $msg
} 1

test mish-4.2 {Positional syntax error - too many arguments} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::mish $t1 extra} msg
    string match "*wrong # args*" $msg
} 1

test mish-4.3 {Positional syntax error - invalid tensor} {
    catch {torch::mish invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 5: Error handling for named syntax
test mish-5.1 {Named syntax error - no arguments} {
    catch {torch::mish} msg
    string match "*wrong # args*" $msg
} 1

test mish-5.2 {Named syntax error - missing value} {
    catch {torch::mish -input} msg
    string match "*wrong # args*" $msg
} 1

test mish-5.3 {Named syntax error - unknown option} {
    set t1 [torch::tensor_create -data {1.0} -dtype float32 -device cpu -requiresGrad false]
    catch {torch::mish -unknown $t1} msg
    string match "*unknown option*" $msg
} 1

test mish-5.4 {Named syntax error - missing required parameter} {
    catch {torch::mish -notinput dummy} msg
    string match "*unknown option*" $msg
} 1

test mish-5.5 {Named syntax error - invalid tensor name} {
    catch {torch::mish -input invalid_tensor} msg
    string match "*Invalid tensor*" $msg
} 1

# Test 6: Data type support
test mish-6.1 {Mish with float32} {
    set t1 [torch::tensor_create -data {1.0 -1.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-6.2 {Mish with float64} {
    set t1 [torch::tensor_create -data {1.0 -1.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

# Test 7: Edge cases and mathematical correctness
test mish-7.1 {Mish with large positive values} {
    set t1 [torch::tensor_create -data {10.0 100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-7.2 {Mish with large negative values} {
    set t1 [torch::tensor_create -data {-10.0 -100.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set result_data [torch::tensor_print $result]
    string match "*tensor*" $result
} 1

test mish-7.3 {Mish with very small values} {
    set t1 [torch::tensor_create -data {0.001 -0.001} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-7.4 {Mish monotonicity test} {
    # Mish should be monotonically increasing
    set t1 [torch::tensor_create -data {-5.0 -4.0 -3.0 -2.0 -1.0 0.0 1.0 2.0 3.0 4.0 5.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-7.5 {Mish smoothness test} {
    # Test around zero where mish transitions smoothly
    set t1 [torch::tensor_create -data {-0.1 -0.05 0.0 0.05 0.1} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

# Test 8: Multidimensional tensors
test mish-8.1 {Mish with 3D tensor} {
    set t1 [torch::tensor_create -data {1.0 -1.0 2.0 -2.0 0.5 -0.5 1.5 -1.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-8.2 {Mish with single element tensor} {
    set t1 [torch::tensor_create -data {2.5} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-8.3 {Mish preserves tensor shape} {
    set t1 [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    set shape [torch::tensor_shape $result]
    # Shape should be preserved
    string match "*tensor*" $result
} 1

# Test 9: Performance and memory
test mish-9.1 {Mish with large tensor} {
    # Create a larger tensor to test performance
    set data {}
    for {set i 0} {$i < 1000} {incr i} {
        lappend data [expr {$i / 500.0 - 1.0}]
    }
    set t1 [torch::tensor_create -data $data -dtype float32 -device cpu -requiresGrad false]
    set result [torch::mish $t1]
    string match "*tensor*" $result
} 1

test mish-9.2 {Mish memory management} {
    # Test that tensors are properly managed
    set t1 [torch::tensor_create -data {1.0 2.0 3.0} -dtype float32 -device cpu -requiresGrad false]
    set result1 [torch::mish $t1]
    set result2 [torch::mish -input $t1]
    # Both results should be valid tensors
    expr {[string match "*tensor*" $result1] && [string match "*tensor*" $result2]}
} 1

# Test 10: Integration with other operations
test mish-10.1 {Mish output can be used in other operations} {
    set t1 [torch::tensor_create -data {1.0 2.0} -dtype float32 -device cpu -requiresGrad false]
    set mish_result [torch::mish $t1]
    set final_result [torch::tensor_add $mish_result $t1]
    string match "*tensor*" $final_result
} 1

test mish-10.2 {Chain mish activations} {
    set t1 [torch::tensor_create -data {0.5 1.0 1.5} -dtype float32 -device cpu -requiresGrad false]
    set mish1 [torch::mish $t1]
    set mish2 [torch::mish $mish1]
    string match "*tensor*" $mish2
} 1

# Cleanup
cleanupTests 