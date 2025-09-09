#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Configure test output
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Test 1: Basic functionality - positional syntax
test expm1-1.1 {Basic expm1 positional syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::expm1 $input]
    expr {[string match "tensor*" $result]}
} {1}

test expm1-1.2 {Basic expm1 named parameter syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::expm1 -input $input]
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Mathematical correctness - expm1(x) = exp(x) - 1
test expm1-2.1 {Mathematical correctness: expm1(0) = 0} {
    set input [torch::tensor_create {0.0}]
    set result [torch::expm1 $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 0.0) < 1e-6}
} {1}

test expm1-2.2 {Mathematical correctness: expm1(1) ≈ e - 1 ≈ 1.718} {
    set input [torch::tensor_create {1.0}]
    set result [torch::expm1 -input $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.71828) < 0.001}
} {1}

test expm1-2.3 {Mathematical correctness: expm1(ln(2)) = 1} {
    set ln2 0.693147
    set input [torch::tensor_create [list $ln2]]
    set result [torch::expm1 $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.0) < 0.001}
} {1}

test expm1-2.4 {Mathematical correctness: expm1(-1) ≈ -0.632} {
    set input [torch::tensor_create -data {-1.0} -dtype float32]
    set result [torch::expm1 -input $input]
    set value [torch::tensor_item $result]
    expr {abs($value - (-0.632121)) < 0.001}
} {1}

# Test 3: Edge cases
test expm1-3.1 {Small values precision} {
    set input [torch::tensor_create {1e-8}]
    set result [torch::expm1 $input]
    set value [torch::tensor_item $result]
    # For very small x, expm1(x) ≈ x
    expr {abs($value - 1e-8) < 1e-9}
} {0}

test expm1-3.2 {Negative values} {
    set input [torch::tensor_create -data {-0.5} -dtype float32]
    set result [torch::expm1 -input $input]
    set value [torch::tensor_item $result]
    # expm1(-0.5) ≈ exp(-0.5) - 1 ≈ -0.393469
    expr {abs($value - (-0.393469)) < 0.001}
} {1}

# Test 4: Different data types
test expm1-4.1 {Float32 tensor} {
    set input [torch::tensor_create {1.0} float32]
    set result [torch::expm1 -input $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.71828) < 0.001}
} {1}

test expm1-4.2 {Float64 tensor} {
    set input [torch::tensor_create {1.0} float64]
    set result [torch::expm1 $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.71828) < 0.001}
} {1}

test expm1-4.3 {Integer tensor} {
    set input [torch::tensor_create {1}]
    set result [torch::expm1 -input $input]
    set value [torch::tensor_item $result]
    expr {abs($value - 1.71828) < 0.001}
} {1}

# Test 5: Error handling
test expm1-5.1 {Error: No arguments} {
    catch {torch::expm1} result
    expr {[string match "*Usage*" $result] || [string match "*Required parameter*" $result]}
} {1}

test expm1-5.2 {Error: Invalid tensor name} {
    catch {torch::expm1 invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test expm1-5.3 {Error: Named parameter without value} {
    catch {torch::expm1 -input} result
    expr {[string match "*Named parameter requires a value*" $result]}
} {1}

test expm1-5.4 {Error: Unknown parameter} {
    set input [torch::tensor_create {1.0}]
    catch {torch::expm1 -unknown $input} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test expm1-5.5 {Error: Missing required parameter} {
    catch {torch::expm1 -dtype float32} result
    expr {[string match "*Required parameter*" $result] || [string match "*Unknown parameter*" $result]}
} {1}

# Test 6: Syntax consistency - both syntaxes produce same results
test expm1-6.1 {Syntax consistency} {
    set input [torch::tensor_create {1.5}]
    set result_pos [torch::expm1 $input]
    set result_named [torch::expm1 -input $input]
    
    set value_pos [torch::tensor_item $result_pos]
    set value_named [torch::tensor_item $result_named]
    
    expr {abs($value_pos - $value_named) < 1e-6}
} {1}

# Test 7: Multiple values
test expm1-7.1 {Multiple values} {
    set input [torch::tensor_create {0.0 1.0 -0.5}]
    set result [torch::expm1 -input $input]
    expr {[string match "tensor*" $result]}
} {1}

# Test 8: Scientific computing example
test expm1-8.1 {Scientific computing: Small interest rate} {
    set rate 0.001  ;# 0.1% interest rate
    set input [torch::tensor_create [list $rate] float64]
    set result [torch::expm1 $input]
    set compound_factor [torch::tensor_item $result]
    # Should be approximately equal to the rate for small rates
    expr {abs($compound_factor - $rate) < 1e-6}
} {1}

cleanupTests 