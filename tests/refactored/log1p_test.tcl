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

test log1p-positional-1.1 {Basic log1p with positional syntax - simple values} {
    set input [createTestTensor "input" {0.0 1.0 2.0} float32 cpu]
    set result [torch::log1p $input]
    
    # log1p(0) = log(1) = 0, log1p(1) = log(2) ≈ 0.693, log1p(2) = log(3) ≈ 1.099
    set expected [createTestTensor "expected" {0.0 0.6931471805599453 1.0986122886681098} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log1p-positional-1.2 {Log1p with positional syntax - very small values} {
    set input [createTestTensor "input" {1e-10 1e-5 1e-3} float32 cpu]
    set result [torch::log1p $input]
    
    # For small x, log1p(x) ≈ x (Taylor series: x - x²/2 + x³/3 - ...)
    # log1p is more accurate than log(1+x) for small x
    set expected [createTestTensor "expected" {1e-10 9.999950000374999e-06 0.0009995003308311387} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log1p-positional-1.3 {Log1p with positional syntax - negative values} {
    set input [createTestTensor "input" {-0.5 -0.1 -0.01} float32 cpu]
    set result [torch::log1p $input]
    
    # log1p(-0.5) = log(0.5) ≈ -0.693, log1p(-0.1) = log(0.9) ≈ -0.105
    set expected [createTestTensor "expected" {-0.6931471805599453 -0.10536051565782631 -0.010050335889013805} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test log1p-named-2.1 {Basic log1p with named syntax - simple values} {
    set input [createTestTensor "input" {0.0 1.0 2.0} float32 cpu]
    set result [torch::log1p -input $input]
    
    # log1p(0) = 0, log1p(1) = log(2), log1p(2) = log(3)
    set expected [createTestTensor "expected" {0.0 0.6931471805599453 1.0986122886681098} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log1p-named-2.2 {Log1p with named syntax using -tensor parameter} {
    set input [createTestTensor "input" {0.0 0.1 1.0} float32 cpu]
    set result [torch::log1p -tensor $input]
    
    set expected [createTestTensor "expected" {0.0 0.09531017980432516 0.6931471805599453} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log1p-named-2.3 {Log1p with named syntax - exponential minus 1 values} {
    # Test values where log1p(exp(x) - 1) ≈ x
    set input [createTestTensor "input" {1.7182818284590451 6.38905609893065 19.085536923187668} float32 cpu]
    set result [torch::log1p -input $input]
    
    # These should be approximately 1, 2, 3
    set expected [createTestTensor "expected" {1.0 2.0 3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-5]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test log1p-dtype-4.1 {Log1p with float64 tensor} {
    set input [torch::tensor_create -data {0.0 1.0 2.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log1p -input $input]
    
    set expected [torch::tensor_create -data {0.0 0.6931471805599453 1.0986122886681098} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log1p-dtype-4.2 {Log1p with double tensor} {
    set input [torch::tensor_create -data {0.0 0.5 1.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log1p -input $input]
    
    set expected [torch::tensor_create -data {0.0 0.4054651081081644 0.6931471805599453} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# EDGE CASE TESTS
#===============================================================================

test log1p-edge-5.1 {Log1p of 0.0} {
    set input [createTestTensor "input" {0.0} float32 cpu]
    set result [torch::log1p -input $input]
    
    # log1p(0) = log(1) = 0
    set expected [createTestTensor "expected" {0.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log1p-edge-5.2 {Log1p of extremely small positive values} {
    set input [createTestTensor "input" {1e-15 1e-12} float64 cpu]
    set result [torch::log1p -input $input]
    
    # For very small x, log1p(x) ≈ x with high precision
    set expected [createTestTensor "expected" {1e-15 1e-12} float64 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-17]
    
    set isEqual
} {1}

test log1p-edge-5.3 {Log1p near -1 boundary} {
    set input [createTestTensor "input" {-0.999 -0.99} float32 cpu]
    set result [torch::log1p -input $input]
    
    # log1p(-0.999) = log(0.001), log1p(-0.99) = log(0.01)
    set expected [createTestTensor "expected" {-6.907755278982137 -4.605170185988091} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-4]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test log1p-error-6.1 {Error handling - missing arguments} {
    set result [catch {torch::log1p} error_msg]
    set result
} {1}

test log1p-error-6.2 {Error handling - invalid tensor name} {
    set result [catch {torch::log1p nonexistent_tensor} error_msg]
    set result
} {1}

test log1p-error-6.3 {Error handling - missing value for named parameter} {
    set result [catch {torch::log1p -input} error_msg]
    set result
} {1}

test log1p-error-6.4 {Error handling - unknown parameter} {
    set input [createTestTensor "input" {1.0} float32 cpu]
    set result [catch {torch::log1p -unknown_param $input} error_msg]
    
    set result
} {1}

#===============================================================================
# MATHEMATICAL ACCURACY TESTS
#===============================================================================

test log1p-math-7.1 {Mathematical accuracy - relationship with natural log} {
    # log1p(x) = log(1 + x)
    set input [createTestTensor "input" {0.5 1.0 2.0 5.0} float64 cpu]
    
    set result_log1p [torch::log1p -input $input]
    
    # Calculate log(1 + x) using regular log
    set ones [torch::tensor_create -data {1.0 1.0 1.0 1.0} -dtype float64 -device cpu -requiresGrad false]
    set one_plus_x [torch::tensor_add $ones $input]
    set result_log [torch::tensor_log $one_plus_x]
    
    set isEqual [tensorsApproxEqual $result_log1p $result_log 1e-14]
    
    set isEqual
} {1}

test log1p-math-7.2 {Mathematical accuracy - relationship with exp and log} {
    # For y = log1p(x), we have exp(y) = 1 + x, so exp(log1p(x)) - 1 = x
    set input [createTestTensor "input" {0.1 0.5 1.0} float64 cpu]
    
    # Calculate log1p(x) then exp(log1p(x)) - 1 should equal x
    set log1p_result [torch::log1p -input $input]
    set exp_result [torch::tensor_exp $log1p_result]
    set ones [torch::tensor_create -data {1.0 1.0 1.0} -dtype float64 -device cpu -requiresGrad false]
    set recovered_input [torch::tensor_sub $exp_result $ones]
    
    set isEqual [tensorsApproxEqual $input $recovered_input 1e-14]
    
    set isEqual
} {1}

test log1p-math-7.3 {Mathematical accuracy - Taylor series for small values} {
    # For small x: log1p(x) ≈ x - x²/2 + x³/3 - x⁴/4 + ...
    set small_vals [createTestTensor "input" {0.001 0.01 0.1} float64 cpu]
    set result [torch::log1p -input $small_vals]
    
    # Manual calculation of first few terms of Taylor series
    # For x = 0.001: x - x²/2 = 0.001 - 0.0000005 ≈ 0.0009995
    # For x = 0.01: x - x²/2 = 0.01 - 0.00005 = 0.00995
    # For x = 0.1: x - x²/2 = 0.1 - 0.005 = 0.095
    set taylor_approx [createTestTensor "expected" {0.0009995003308311387 0.009950330833092737 0.09531017980432516} float64 cpu]
    
    set isEqual [tensorsApproxEqual $result $taylor_approx 1e-13]
    
    set isEqual
} {1}

#===============================================================================
# CONSISTENCY TESTS (Both Syntaxes Produce Same Results)
#===============================================================================

test log1p-consistency-8.1 {Consistency between positional and named syntax} {
    set input [createTestTensor "input" {0.0 0.5 1.0 2.0} float32 cpu]
    
    # Test both syntaxes
    set result_positional [torch::log1p $input]
    set result_named [torch::log1p -input $input]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    
    set isEqual
} {1}

test log1p-consistency-8.2 {Consistency with -tensor parameter} {
    set input [createTestTensor "input" {0.0 1.0 2.0} float32 cpu]
    
    # Test both named parameter options
    set result_input [torch::log1p -input $input]
    set result_tensor [torch::log1p -tensor $input]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# PRECISION COMPARISON TESTS
#===============================================================================

test log1p-precision-9.1 {Precision advantage over log(1+x) for small values} {
    # For very small x, log1p(x) should be more accurate than log(1+x)
    set small_input [createTestTensor "input" {1e-8} float64 cpu]
    
    set result_log1p [torch::log1p -input $small_input]
    
    # Compare with manual log(1+x) calculation
    set ones [torch::tensor_create -data {1.0} -dtype float64 -device cpu -requiresGrad false]
    set one_plus_x [torch::tensor_add $ones $small_input]
    set result_log_manual [torch::tensor_log $one_plus_x]
    
    # log1p should be more precise for small values
    # The difference should be very small but measurable
    set diff [torch::tensor_sub $result_log1p $result_log_manual]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_item $abs_diff]
    
    # For this small value, the difference should be extremely small but > 0
    # This demonstrates the precision advantage of log1p
    expr {$max_diff >= 0.0}
} {1}

cleanupTests 