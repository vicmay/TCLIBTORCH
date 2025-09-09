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
test exp10-1.1 {Basic positional syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::exp10 $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 2: Named parameter syntax
test exp10-2.1 {Named parameter syntax} {
    set input [torch::tensor_create {1.0 2.0 3.0}]
    set result [torch::exp10 -input $input]
    
    # Check that result is a valid tensor handle
    expr {[string match "tensor*" $result]}
} {1}

# Test 3: Mathematical correctness - exp10(1) should be 10
test exp10-3.1 {Mathematical correctness - positional} {
    set input [torch::tensor_create {1.0}]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(1) = 10^1 = 10
    expr {abs($value - 10.0) < 1e-6}
} {1}

test exp10-3.2 {Mathematical correctness - named parameters} {
    set input [torch::tensor_create {1.0}]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(1) = 10^1 = 10
    expr {abs($value - 10.0) < 1e-6}
} {1}

# Test 4: exp10(0) should be 1
test exp10-4.1 {exp10(0) = 1 - positional syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(0) = 10^0 = 1
    expr {abs($value - 1.0) < 1e-6}
} {1}

test exp10-4.2 {exp10(0) = 1 - named syntax} {
    set input [torch::tensor_create {0.0}]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(0) = 10^0 = 1
    expr {abs($value - 1.0) < 1e-6}
} {1}

# Test 5: exp10(2) should be 100
test exp10-5.1 {exp10(2) = 100 - positional syntax} {
    set input [torch::tensor_create {2.0}]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(2) = 10^2 = 100
    expr {abs($value - 100.0) < 1e-6}
} {1}

test exp10-5.2 {exp10(2) = 100 - named syntax} {
    set input [torch::tensor_create {2.0}]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(2) = 10^2 = 100
    expr {abs($value - 100.0) < 1e-6}
} {1}

# Test 6: Multiple values
test exp10-6.1 {Multiple values - positional syntax} {
    set input [torch::tensor_create {0.0 1.0 2.0}]
    set result [torch::exp10 $input]
    
    # Should process all values correctly
    expr {[string match "tensor*" $result]}
} {1}

test exp10-6.2 {Multiple values - named syntax} {
    set input [torch::tensor_create {0.0 1.0 2.0}]
    set result [torch::exp10 -input $input]
    
    # Should process all values correctly
    expr {[string match "tensor*" $result]}
} {1}

# Test 7: Different data types
test exp10-7.1 {Float32 data type} {
    set input [torch::tensor_create {1.0} float32]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(1) = 10
    expr {abs($value - 10.0) < 1e-6}
} {1}

test exp10-7.2 {Float64 data type} {
    set input [torch::tensor_create {2.0} float64]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(2) = 100
    expr {abs($value - 100.0) < 1e-6}
} {1}

# Test 8: Negative values
test exp10-8.1 {Negative values - positional syntax} {
    set input [torch::tensor_create -data {-1.0} -dtype float32]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(-1) = 10^(-1) = 0.1
    expr {abs($value - 0.1) < 1e-6}
} {1}

test exp10-8.2 {Negative values - named syntax} {
    set input [torch::tensor_create -data {-2.0} -dtype float32]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(-2) = 10^(-2) = 0.01
    expr {abs($value - 0.01) < 1e-6}
} {1}

# Test 9: Error handling
test exp10-9.1 {Error: missing tensor argument} {
    catch {torch::exp10} result
    expr {[string match "*Required parameter*input missing*" $result] || [string match "*arguments*" $result] || [string match "*Usage*" $result]}
} {1}

test exp10-9.2 {Error: invalid tensor name} {
    catch {torch::exp10 invalid_tensor} result
    expr {[string match "*Invalid tensor*" $result]}
} {1}

test exp10-9.3 {Error: missing parameter value in named syntax} {
    catch {torch::exp10 -input} result
    expr {[string match "*Missing value*" $result]}
} {1}

test exp10-9.4 {Error: unknown parameter} {
    set input [torch::tensor_create {1.0}]
    catch {torch::exp10 -unknown $input} result
    expr {[string match "*Unknown parameter*" $result]}
} {1}

test exp10-9.5 {Error: missing required parameter} {
    catch {torch::exp10 -dtype float32} result
    expr {[string match "*Required parameter*" $result] || [string match "*Unknown parameter*" $result]}
} {1}

# Test 10: Both syntaxes produce same result
test exp10-10.1 {Both syntaxes produce same result} {
    set input [torch::tensor_create {1.5}]
    
    set result_pos [torch::exp10 $input]
    set result_named [torch::exp10 -input $input]
    
    set value_pos [torch::tensor_item $result_pos]
    set value_named [torch::tensor_item $result_named]
    
    expr {abs($value_pos - $value_named) < 1e-6}
} {1}

# Test 11: Fractional exponents
test exp10-11.1 {Fractional exponents} {
    set input [torch::tensor_create {0.5}]
    set result [torch::exp10 -input $input]
    set value [torch::tensor_item $result]
    
    # exp10(0.5) = 10^0.5 = sqrt(10) ≈ 3.162
    expr {abs($value - 3.16227766) < 1e-6}
} {1}

# Test 12: Large exponents
test exp10-12.1 {Large exponents} {
    set input [torch::tensor_create {3.0}]
    set result [torch::exp10 $input]
    set value [torch::tensor_item $result]
    
    # exp10(3) = 10^3 = 1000
    expr {abs($value - 1000.0) < 1e-6}
} {1}

cleanupTests 