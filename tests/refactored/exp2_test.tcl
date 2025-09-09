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
test exp2-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::exp2 $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Named parameter syntax
test exp2-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::exp2 -input $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: Mathematical correctness - exp2(1) should be 2
test exp2-3.1 {Mathematical correctness - positional} {
    set input [torch::tensor_create {1.0}]
    set result [torch::exp2 $input]
    set value [torch::tensor_item $result]
    
    # exp2(1) = 2^1 = 2
    expr {abs($value - 2.0) < 1e-6}
} {1}

test exp2-3.2 {Mathematical correctness - named parameters} {
    set input [torch::tensor_create {1.0}]
    set result [torch::exp2 -input $input]
    set value [torch::tensor_item $result]
    
    # exp2(1) = 2^1 = 2
    expr {abs($value - 2.0) < 1e-6}
} {1}

# Test 4: exp2(0) should be 1
test exp2-4.1 {exp2(0) = 1 - positional syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::exp2 $input]
    set value [torch::tensor_item $result]
    
    # exp2(0) = 2^0 = 1
    expr {abs($value - 1.0) < 1e-6}
} {1}

test exp2-4.2 {exp2(0) = 1 - named syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::exp2 -input $input]
    set value [torch::tensor_item $result]
    
    # exp2(0) = 2^0 = 1
    expr {abs($value - 1.0) < 1e-6}
} {1}

# Test 5: Multiple values
test exp2-5.1 {Multiple values - positional syntax} {
    set input [torch::tensor_create {0.0 1.0 2.0}]
    set result [torch::exp2 $input]
    
    # Should process all values correctly
    expr {[string match "tensor*" $result]}
} {1}

test exp2-5.2 {Multiple values - named syntax} {
    set input [torch::tensor_create {0.0 1.0 2.0}]
    set result [torch::exp2 -input $input]
    
    # Should process all values correctly
    expr {[string match "tensor*" $result]}
} {1}

# Test 6: Different data types
test exp2-6.1 {Float32 data type} {
    set input [torch::tensor_create {2.0} float32]
    set result [torch::exp2 -input $input]
    set value [torch::tensor_item $result]
    
    # exp2(2) = 2^2 = 4
    expr {abs($value - 4.0) < 1e-6}
} {1}

test exp2-6.2 {Float64 data type} {
    set input [torch::tensor_create {3.0} float64]
    set result [torch::exp2 $input]
    set value [torch::tensor_item $result]
    
    # exp2(3) = 2^3 = 8
    expr {abs($value - 8.0) < 1e-6}
} {1}

# Test 7: Negative values
test exp2-7.1 {Negative values - positional syntax} {
    set input [torch::tensor_create -data {-1.0} -dtype float32]
    set result [torch::exp2 $input]
    set value [torch::tensor_item $result]
    
    # exp2(-1) = 2^(-1) = 0.5
    expr {abs($value - 0.5) < 1e-6}
} {1}

test exp2-7.2 {Negative values - named syntax} {
    set input [torch::tensor_create -data {-2.0} -dtype float32]
    set result [torch::exp2 -input $input]
    set value [torch::tensor_item $result]
    
    # exp2(-2) = 2^(-2) = 0.25
    expr {abs($value - 0.25) < 1e-6}
} {1}

# Test 8: Error handling
test exp2-8.1 {Error: missing tensor argument} {
    catch {torch::exp2} result
    expr {[string match "*Required parameter*input missing*" $result] || [string match "*arguments*" $result] || [string match "*Usage*" $result]}
} {1}

test exp2-8.2 {Error: invalid tensor name} {
    catch {torch::exp2 invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test exp2-8.3 {Error: missing parameter value in named syntax} {
    catch {torch::exp2 -input} result
    expr {[string match "*Missing value*" $result]}
} {1}

test exp2-8.4 {Error: unknown parameter} {
    set input [torch::tensor_create {1.0}]
    catch {torch::exp2 -unknown $input} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test exp2-8.5 {Error: missing required parameter} {
    catch {torch::exp2 -dtype float32} result
    expr {[string match "*Required parameter*" $result] || [string match "*Unknown parameter*" $result]}
} {1}

# Test 9: Both syntaxes produce same result
test exp2-9.1 {Both syntaxes produce same result} {
    set input [torch::tensor_create {1.5}]
    
    set result_pos [torch::exp2 $input]
    set result_named [torch::exp2 -input $input]
    
    set value_pos [torch::tensor_item $result_pos]
    set value_named [torch::tensor_item $result_named]
    
    expr {abs($value_pos - $value_named) < 1e-6}
} {1}

# Test 10: Fractional exponents
test exp2-10.1 {Fractional exponents} {
    set input [torch::tensor_create {0.5}]
    set result [torch::exp2 -input $input]
    set value [torch::tensor_item $result]
    
    # exp2(0.5) = 2^0.5 = sqrt(2) ≈ 1.414
    expr {abs($value - 1.41421356) < 1e-6}
} {1}

cleanupTests 