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

# Helper function to create test tensors
proc createTestTensor {name values dtype device} {
    set tensor [torch::tensor_create -data $values -dtype $dtype -device $device -requiresGrad false]
    return $tensor
}

# Helper function to check if tensors are approximately equal
proc tensorsApproxEqual {tensor1 tensor2 tolerance} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

#===============================================================================
# POSITIONAL SYNTAX TESTS (Backward Compatibility)
#===============================================================================

test log10-positional-1.1 {Basic log10 with positional syntax - simple values} {
    set input [createTestTensor "input" {10.0 100.0 1000.0} float32 cpu]
    set result [torch::log10 $input]
    
    # log10(10) = 1, log10(100) = 2, log10(1000) = 3
    set expected [createTestTensor "expected" {1.0 2.0 3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log10-positional-1.2 {Log10 with positional syntax - powers of 10} {
    set input [createTestTensor "input" {1.0 10.0 100.0 1000.0 10000.0} float32 cpu]
    set result [torch::log10 $input]
    
    # log10(10^n) = n
    set expected [createTestTensor "expected" {0.0 1.0 2.0 3.0 4.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log10-positional-1.3 {Log10 with positional syntax - fractional values} {
    set input [createTestTensor "input" {0.1 0.01 0.001} float32 cpu]
    set result [torch::log10 $input]
    
    # log10(0.1) = -1, log10(0.01) = -2, log10(0.001) = -3
    set expected [createTestTensor "expected" {-1.0 -2.0 -3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test log10-named-2.1 {Basic log10 with named syntax - simple values} {
    set input [createTestTensor "input" {10.0 100.0 1000.0} float32 cpu]
    set result [torch::log10 -input $input]
    
    # log10(10) = 1, log10(100) = 2, log10(1000) = 3
    set expected [createTestTensor "expected" {1.0 2.0 3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log10-named-2.2 {Log10 with named syntax using -tensor parameter} {
    set input [createTestTensor "input" {1.0 10.0 100.0} float32 cpu]
    set result [torch::log10 -tensor $input]
    
    set expected [createTestTensor "expected" {0.0 1.0 2.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log10-named-2.3 {Log10 with named syntax - large values} {
    set input [createTestTensor "input" {1e6 1e7 1e8} float32 cpu]
    set result [torch::log10 -input $input]
    
    # log10(10^6) = 6, log10(10^7) = 7, log10(10^8) = 8
    set expected [createTestTensor "expected" {6.0 7.0 8.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-5]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test log10-dtype-4.1 {Log10 with float64 tensor} {
    set input [torch::tensor_create -data {10.0 100.0 1000.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log10 -input $input]
    
    set expected [torch::tensor_create -data {1.0 2.0 3.0} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log10-dtype-4.2 {Log10 with double tensor} {
    set input [torch::tensor_create -data {1.0 10.0 100.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log10 -input $input]
    
    set expected [torch::tensor_create -data {0.0 1.0 2.0} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# EDGE CASE TESTS
#===============================================================================

test log10-edge-5.1 {Log10 of 1.0} {
    set input [createTestTensor "input" {1.0} float32 cpu]
    set result [torch::log10 -input $input]
    
    # log10(1) = 0
    set expected [createTestTensor "expected" {0.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log10-edge-5.2 {Log10 of very small positive values} {
    set input [createTestTensor "input" {1e-10 1e-5} float32 cpu]
    set result [torch::log10 -input $input]
    
    # log10(1e-10) = -10, log10(1e-5) = -5
    set expected [createTestTensor "expected" {-10.0 -5.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-5]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test log10-error-6.1 {Error handling - missing arguments} {
    set result [catch {torch::log10} error_msg]
    set result
} {1}

test log10-error-6.2 {Error handling - invalid tensor name} {
    set result [catch {torch::log10 nonexistent_tensor} error_msg]
    set result
} {1}

test log10-error-6.3 {Error handling - missing value for named parameter} {
    set result [catch {torch::log10 -input} error_msg]
    set result
} {1}

test log10-error-6.4 {Error handling - unknown parameter} {
    set input [createTestTensor "input" {10.0} float32 cpu]
    set result [catch {torch::log10 -unknown_param $input} error_msg]
    
    set result
} {1}

#===============================================================================
# MATHEMATICAL ACCURACY TESTS
#===============================================================================

test log10-math-7.1 {Mathematical accuracy - powers of 10} {
    # Test various powers of 10 for exact results
    set powers {1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3}
    set expected_logs {-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0}
    
    set input [torch::tensor_create -data $powers -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log10 -input $input]
    
    set expected [torch::tensor_create -data $expected_logs -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-14]
    
    set isEqual
} {1}

#===============================================================================
# CONSISTENCY TESTS (Both Syntaxes Produce Same Results)
#===============================================================================

test log10-consistency-8.1 {Consistency between positional and named syntax} {
    set input [createTestTensor "input" {10.0 100.0 1000.0 10000.0} float32 cpu]
    
    # Test both syntaxes
    set result_positional [torch::log10 $input]
    set result_named [torch::log10 -input $input]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    
    set isEqual
} {1}

test log10-consistency-8.2 {Consistency with -tensor parameter} {
    set input [createTestTensor "input" {1.0 10.0 100.0} float32 cpu]
    
    # Test both named parameter options
    set result_input [torch::log10 -input $input]
    set result_tensor [torch::log10 -tensor $input]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    
    set isEqual
} {1}

cleanupTests 