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

test log2-positional-1.1 {Basic log2 with positional syntax - powers of 2} {
    set input [createTestTensor "input" {1.0 2.0 4.0 8.0} float32 cpu]
    set result [torch::log2 $input]
    
    # log2(1) = 0, log2(2) = 1, log2(4) = 2, log2(8) = 3
    set expected [createTestTensor "expected" {0.0 1.0 2.0 3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log2-positional-1.2 {Log2 with positional syntax - simple values} {
    set input [createTestTensor "input" {0.5 1.0 2.0 16.0} float32 cpu]
    set result [torch::log2 $input]
    
    # log2(0.5) = -1, log2(1) = 0, log2(2) = 1, log2(16) = 4
    set expected [createTestTensor "expected" {-1.0 0.0 1.0 4.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log2-positional-1.3 {Log2 with positional syntax - non-power-of-2 values} {
    set input [createTestTensor "input" {3.0 5.0 10.0} float32 cpu]
    set result [torch::log2 $input]
    
    # log2(3) ≈ 1.585, log2(5) ≈ 2.322, log2(10) ≈ 3.322
    set expected [createTestTensor "expected" {1.5849625007211563 2.321928094887362 3.3219280948873626} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

#===============================================================================
# NAMED PARAMETER SYNTAX TESTS
#===============================================================================

test log2-named-2.1 {Basic log2 with named syntax - powers of 2} {
    set input [createTestTensor "input" {1.0 2.0 4.0 8.0} float32 cpu]
    set result [torch::log2 -input $input]
    
    # log2(1) = 0, log2(2) = 1, log2(4) = 2, log2(8) = 3
    set expected [createTestTensor "expected" {0.0 1.0 2.0 3.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log2-named-2.2 {Log2 with named syntax using -tensor parameter} {
    set input [createTestTensor "input" {0.25 0.5 1.0 2.0} float32 cpu]
    set result [torch::log2 -tensor $input]
    
    # log2(0.25) = -2, log2(0.5) = -1, log2(1) = 0, log2(2) = 1
    set expected [createTestTensor "expected" {-2.0 -1.0 0.0 1.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

test log2-named-2.3 {Log2 with named syntax - large powers of 2} {
    set input [createTestTensor "input" {32.0 64.0 128.0 256.0} float32 cpu]
    set result [torch::log2 -input $input]
    
    # log2(32) = 5, log2(64) = 6, log2(128) = 7, log2(256) = 8
    set expected [createTestTensor "expected" {5.0 6.0 7.0 8.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

#===============================================================================
# DATA TYPE COMPATIBILITY TESTS
#===============================================================================

test log2-dtype-4.1 {Log2 with float64 tensor} {
    set input [torch::tensor_create -data {1.0 2.0 4.0 8.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log2 -input $input]
    
    set expected [torch::tensor_create -data {0.0 1.0 2.0 3.0} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log2-dtype-4.2 {Log2 with double tensor} {
    set input [torch::tensor_create -data {0.5 1.0 2.0} -dtype float64 -device cpu -requiresGrad false]
    set result [torch::log2 -input $input]
    
    set expected [torch::tensor_create -data {-1.0 0.0 1.0} -dtype float64 -device cpu -requiresGrad false]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# EDGE CASE TESTS
#===============================================================================

test log2-edge-5.1 {Log2 of 1.0} {
    set input [createTestTensor "input" {1.0} float32 cpu]
    set result [torch::log2 -input $input]
    
    # log2(1) = 0
    set expected [createTestTensor "expected" {0.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log2-edge-5.2 {Log2 of very small positive values} {
    set input [createTestTensor "input" {0.001 0.01} float64 cpu]
    set result [torch::log2 -input $input]
    
    # log2(0.001) ≈ -9.966, log2(0.01) ≈ -6.644
    set expected [createTestTensor "expected" {-9.965784284662087 -6.643856189774724} float64 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-12]
    
    set isEqual
} {1}

test log2-edge-5.3 {Log2 of very large values} {
    set input [createTestTensor "input" {1024.0 1048576.0} float32 cpu]
    set result [torch::log2 -input $input]
    
    # log2(1024) = 10, log2(1048576) = 20
    set expected [createTestTensor "expected" {10.0 20.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-6]
    
    set isEqual
} {1}

#===============================================================================
# ERROR HANDLING TESTS
#===============================================================================

test log2-error-6.1 {Error handling - missing arguments} {
    set result [catch {torch::log2} error_msg]
    set result
} {1}

test log2-error-6.2 {Error handling - invalid tensor name} {
    set result [catch {torch::log2 nonexistent_tensor} error_msg]
    set result
} {1}

test log2-error-6.3 {Error handling - missing value for named parameter} {
    set result [catch {torch::log2 -input} error_msg]
    set result
} {1}

test log2-error-6.4 {Error handling - unknown parameter} {
    set input [createTestTensor "input" {2.0} float32 cpu]
    set result [catch {torch::log2 -unknown_param $input} error_msg]
    
    set result
} {1}

#===============================================================================
# MATHEMATICAL ACCURACY TESTS
#===============================================================================

test log2-math-7.1 {Mathematical accuracy - relationship with natural log} {
    # log2(x) = ln(x) / ln(2)
    set input [createTestTensor "input" {2.0 4.0 8.0 16.0} float64 cpu]
    
    set result_log2 [torch::log2 -input $input]
    
    # Calculate ln(x) / ln(2) using regular log
    set result_ln [torch::tensor_log $input]
    set ln2 [torch::tensor_create -data {0.6931471805599453 0.6931471805599453 0.6931471805599453 0.6931471805599453} -dtype float64 -device cpu -requiresGrad false]
    set result_manual [torch::tensor_div $result_ln $ln2]
    
    set isEqual [tensorsApproxEqual $result_log2 $result_manual 1e-14]
    
    set isEqual
} {1}

test log2-math-7.2 {Mathematical accuracy - relationship with exp2} {
    # For y = log2(x), we have 2^y = x, so 2^(log2(x)) = x
    set input [createTestTensor "input" {1.0 2.0 4.0 8.0} float64 cpu]
    
    # Calculate log2(x) then 2^(log2(x)) should equal x
    set log2_result [torch::log2 -input $input]
    set twos [torch::tensor_create -data {2.0 2.0 2.0 2.0} -dtype float64 -device cpu -requiresGrad false]
    set recovered_input [torch::pow $twos $log2_result]
    
    set isEqual [tensorsApproxEqual $input $recovered_input 1e-14]
    
    set isEqual
} {1}

test log2-math-7.3 {Mathematical accuracy - logarithm properties} {
    # log2(a * b) = log2(a) + log2(b)
    set a [createTestTensor "a" {2.0 4.0} float64 cpu]
    set b [createTestTensor "b" {8.0 16.0} float64 cpu]
    
    # Calculate log2(a * b)
    set ab [torch::tensor_mul $a $b]
    set log2_ab [torch::log2 -input $ab]
    
    # Calculate log2(a) + log2(b)
    set log2_a [torch::log2 -input $a]
    set log2_b [torch::log2 -input $b]
    set log2_a_plus_log2_b [torch::tensor_add $log2_a $log2_b]
    
    set isEqual [tensorsApproxEqual $log2_ab $log2_a_plus_log2_b 1e-13]
    
    set isEqual
} {1}

#===============================================================================
# CONSISTENCY TESTS (Both Syntaxes Produce Same Results)
#===============================================================================

test log2-consistency-8.1 {Consistency between positional and named syntax} {
    set input [createTestTensor "input" {1.0 2.0 4.0 8.0} float32 cpu]
    
    # Test both syntaxes
    set result_positional [torch::log2 $input]
    set result_named [torch::log2 -input $input]
    
    set isEqual [tensorsApproxEqual $result_positional $result_named 1e-15]
    
    set isEqual
} {1}

test log2-consistency-8.2 {Consistency with -tensor parameter} {
    set input [createTestTensor "input" {0.5 1.0 2.0} float32 cpu]
    
    # Test both named parameter options
    set result_input [torch::log2 -input $input]
    set result_tensor [torch::log2 -tensor $input]
    
    set isEqual [tensorsApproxEqual $result_input $result_tensor 1e-15]
    
    set isEqual
} {1}

#===============================================================================
# BINARY COMPUTER SCIENCE TESTS
#===============================================================================

test log2-binary-9.1 {Binary logarithm for computer science applications} {
    # Common powers of 2 used in computer science
    set input [createTestTensor "input" {2.0 4.0 8.0 16.0 32.0 64.0 128.0 256.0 512.0 1024.0} float32 cpu]
    set result [torch::log2 -input $input]
    
    # Should be exact integers: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    set expected [createTestTensor "expected" {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0} float32 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-15]
    
    set isEqual
} {1}

test log2-binary-9.2 {Fractional powers of 2} {
    # Fractional powers: 2^(-1), 2^(-2), 2^(-0.5)
    set input [createTestTensor "input" {0.5 0.25 0.7071067811865476} float64 cpu]
    set result [torch::log2 -input $input]
    
    # log2(0.5) = -1, log2(0.25) = -2, log2(sqrt(0.5)) = -0.5
    set expected [createTestTensor "expected" {-1.0 -2.0 -0.5} float64 cpu]
    set isEqual [tensorsApproxEqual $result $expected 1e-14]
    
    set isEqual
} {1}

cleanupTests 