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

# Helper function to check if tensors are close
proc tensors_close {t1 t2 {rtol 1e-5} {atol 1e-8}} {
    set diff [torch::tensor_sub $t1 $t2]
    set abs_diff [torch::tensor_abs $diff]
    set abs_t2 [torch::tensor_abs $t2]
    set max_diff [torch::tensor_max $abs_diff]
    set max_t2 [torch::tensor_max $abs_t2]
    set max_diff_val [torch::tensor_item $max_diff]
    set max_t2_val [torch::tensor_item $max_t2]
    
    puts "t1: [torch::tensor_print $t1]"
    puts "t2: [torch::tensor_print $t2]"
    puts "max_diff_val: $max_diff_val"
    puts "max_t2_val: $max_t2_val"
    puts "threshold: [expr {$rtol * $max_t2_val + $atol}]"
    
    return [expr {$max_diff_val <= ($rtol * $max_t2_val + $atol)}]
}

# Test cases for positional syntax
test rms_norm-1.1 {Basic positional syntax - 1D} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set result [torch::rms_norm $x {4}]
    
    # Expected values calculated manually:
    # RMS = sqrt(mean(x^2)) = sqrt((1^2 + 2^2 + 3^2 + 4^2)/4) ≈ 2.7386
    # result = x/RMS = [0.365148 0.730296 1.09544 1.46059]
    set expected [torch::tensor_create {0.365148 0.730296 1.09544 1.46059} float32]
    tensors_close $result $expected
} 1

test rms_norm-1.2 {Basic positional syntax - 2D} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {2 3}]
    set result [torch::rms_norm $x {3}]
    
    # Normalize along last dimension
    set expected [torch::tensor_create {0.46291 0.925819 1.38873 0.789542 0.986927 1.18431} float32]
    set expected [torch::tensor_reshape $expected {2 3}]
    tensors_close $result $expected
} 1

test rms_norm-1.3 {Positional syntax with custom eps} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set result [torch::rms_norm $x {4} 1e-3]
    
    # Similar to test 1.1 but with different eps
    set expected [torch::tensor_create {0.365124 0.730248 1.09537 1.4605} float32]
    tensors_close $result $expected
} 1

# Test cases for named parameter syntax
test rms_norm-2.1 {Named parameter syntax - 1D} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set result [torch::rmsNorm -input $x -normalizedShape {4}]
    
    set expected [torch::tensor_create {0.365148 0.730296 1.09544 1.46059} float32]
    tensors_close $result $expected
} 1

test rms_norm-2.2 {Named parameter syntax - 2D} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0 5.0 6.0} float32]
    set x [torch::tensor_reshape $x {2 3}]
    set result [torch::rmsNorm -input $x -normalizedShape {3}]
    
    set expected [torch::tensor_create {0.46291 0.925819 1.38873 0.789542 0.986927 1.18431} float32]
    set expected [torch::tensor_reshape $expected {2 3}]
    tensors_close $result $expected
} 1

test rms_norm-2.3 {Named parameter syntax with custom eps} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    set result [torch::rmsNorm -input $x -normalizedShape {4} -eps 1e-3]
    
    set expected [torch::tensor_create {0.365124 0.730248 1.09537 1.4605} float32]
    tensors_close $result $expected
} 1

# Error handling tests
test rms_norm-3.1 {Error - empty tensor} {
    set x [torch::tensor_create {} float32]
    catch {torch::rms_norm $x {1}} err
    set err
} {Input tensor is empty}

test rms_norm-3.2 {Error - invalid normalized_shape} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    catch {torch::rms_norm $x {5}} err
    set err
} {Invalid normalized_shape: dimensions don't match input tensor}

test rms_norm-3.3 {Error - negative eps} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    catch {torch::rmsNorm -input $x -normalizedShape {4} -eps -1.0} err
    set err
} {Invalid eps value: must be positive}

test rms_norm-3.4 {Error - missing required parameters} {
    set x [torch::tensor_create {1.0 2.0 3.0 4.0} float32]
    catch {torch::rmsNorm -input $x} err
    set err
} {Required parameters missing: input tensor and normalized_shape required}

cleanupTests 