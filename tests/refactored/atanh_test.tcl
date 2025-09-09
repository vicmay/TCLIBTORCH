#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the libtorchtcl extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic positional syntax
test atanh-1.1 {Basic positional syntax} {
    set input [torch::tensor_create -data {0.0 0.5 -0.5} -shape {3} -dtype float32 -device cpu]
    set result [torch::atanh $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Named parameter syntax
test atanh-2.1 {Named parameter syntax} {
    set input [torch::tensor_create -data {0.0 0.5 -0.5} -shape {3} -dtype float32 -device cpu]
    set result [torch::atanh -input $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: Mathematical correctness - atanh(0) = 0
test atanh-3.1 {Mathematical correctness - atanh(0) = 0} {
    set input [torch::tensor_create -data {0.0} -shape {1} -dtype float32 -device cpu]
    set result [torch::atanh $input]
    set value [torch::tensor_item $result]
    
    # atanh(0) should be 0
    expr {abs($value) < 1e-6}
} {1}

# Test 4: Mathematical correctness - named syntax
test atanh-4.1 {Mathematical correctness - named syntax} {
    set input [torch::tensor_create -data {0.0} -shape {1} -dtype float32 -device cpu]
    set result [torch::atanh -input $input]
    set value [torch::tensor_item $result]
    
    # atanh(0) should be 0
    expr {abs($value) < 1e-6}
} {1}

# Test 5: Symmetric property - atanh(-x) = -atanh(x)
test atanh-5.1 {Symmetric property - positional syntax} {
    set input1 [torch::tensor_create -data {0.5} -shape {1} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {-0.5} -shape {1} -dtype float32 -device cpu]
    
    set result1 [torch::atanh $input1]
    set result2 [torch::atanh $input2]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    # atanh(-x) should equal -atanh(x)
    expr {abs($value1 + $value2) < 1e-6}
} {1}

test atanh-5.2 {Symmetric property - named syntax} {
    set input1 [torch::tensor_create -data {0.5} -shape {1} -dtype float32 -device cpu]
    set input2 [torch::tensor_create -data {-0.5} -shape {1} -dtype float32 -device cpu]
    
    set result1 [torch::atanh -input $input1]
    set result2 [torch::atanh -input $input2]
    
    set value1 [torch::tensor_item $result1]
    set value2 [torch::tensor_item $result2]
    
    # atanh(-x) should equal -atanh(x)
    expr {abs($value1 + $value2) < 1e-6}
} {1}

# Test 6: Different data types
test atanh-6.1 {Different data types - float32} {
    set input [torch::tensor_create -data {0.5} -shape {1} -dtype float32 -device cpu]
    set result [torch::atanh -input $input]
    
    expr {[string match "tensor*" $result]}
} {1}

test atanh-6.2 {Different data types - float64} {
    set input [torch::tensor_create -data {0.5} -shape {1} -dtype float64 -device cpu]
    set result [torch::atanh -input $input]
    
    expr {[string match "tensor*" $result]}
} {1}

# Test 7: Multidimensional tensors (simplified)
test atanh-7.1 {Simple multi-element tensor - positional syntax} {
    set input [torch::tensor_create -data {0.0 0.5 -0.5} -shape {3} -dtype float32 -device cpu]
    set result [torch::atanh $input]
    
    expr {[string match "tensor*" $result]}
} {1}

test atanh-7.2 {Simple multi-element tensor - named syntax} {
    set input [torch::tensor_create -data {0.0 0.5 -0.5} -shape {3} -dtype float32 -device cpu]
    set result [torch::atanh -input $input]
    
    expr {[string match "tensor*" $result]}
} {1}

# Test 8: Error handling
test atanh-8.1 {Invalid tensor name - positional} {
    set result [catch {torch::atanh invalid_tensor} error]
    
    expr {$result == 1}
} {1}

test atanh-8.2 {Invalid tensor name - named} {
    set result [catch {torch::atanh -input invalid_tensor} error]
    
    expr {$result == 1}
} {1}

test atanh-8.3 {Wrong number of arguments - positional} {
    set result [catch {torch::atanh} error]
    
    expr {$result == 1}
} {1}

test atanh-8.4 {Wrong parameter name} {
    set input [torch::tensor_create {0.5}]
    set result [catch {torch::atanh -wrong $input} error]
    
    expr {$result == 1}
} {1}

test atanh-8.5 {Missing parameter value} {
    set result [catch {torch::atanh -input} error]
    
    expr {$result == 1}
} {1}

# Test 9: Both syntaxes produce same result
test atanh-9.1 {Both syntaxes produce same result} {
    set input [torch::tensor_create {0.5}]
    
    set result_pos [torch::atanh $input]
    set result_named [torch::atanh -input $input]
    
    # Compare the tensor contents (simplified comparison)
    set values_pos [torch::tensor_item $result_pos]
    set values_named [torch::tensor_item $result_named]
    
    expr {abs($values_pos - $values_named) < 1e-6}
} {1}

# Cleanup and summary
cleanupTests 