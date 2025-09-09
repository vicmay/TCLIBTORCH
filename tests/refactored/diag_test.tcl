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

# Test helper function to verify tensor result
proc verify_diag_result {result} {
    # Verify we got a valid tensor handle back
    if {![string match "tensor*" $result]} {
        return 0
    }
    
    # Basic verification that the tensor exists and can be accessed
    set shape [torch::tensor_shape $result]
    if {$shape == ""} {
        return 0
    }
    
    return 1
}

# Test helper functions to create test tensors
proc create_test_vector {} {
    # Create a simple 1D vector for diagonal matrix creation
    return [torch::tensor_create -data {1.0 2.0 3.0} -shape {3} -dtype float32]
}

proc create_test_matrix {} {
    # Create a simple 3x3 matrix for diagonal extraction
    return [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0} -shape {3 3} -dtype float32]
}

# Test positional syntax
test diag-1.1 {Basic positional syntax - diagonal matrix from vector} {
    set vector [create_test_vector]
    set result [torch::diag $vector]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diag-1.2 {Basic positional syntax - extract diagonal from matrix} {
    set matrix [create_test_matrix]
    set result [torch::diag $matrix]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3"}
} {1}

test diag-1.3 {Positional syntax with diagonal offset} {
    set matrix [create_test_matrix]
    set result [torch::diag $matrix 1]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test diag-1.4 {Positional syntax with negative diagonal offset} {
    set matrix [create_test_matrix]
    set result [torch::diag $matrix -1]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

# Test named parameter syntax
test diag-2.1 {Named parameter syntax - diagonal matrix from vector} {
    set vector [create_test_vector]
    set result [torch::diag -input $vector]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diag-2.2 {Named parameter syntax - extract diagonal from matrix} {
    set matrix [create_test_matrix]
    set result [torch::diag -input $matrix]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3"}
} {1}

test diag-2.3 {Named parameter with diagonal offset} {
    set matrix [create_test_matrix]
    set result [torch::diag -input $matrix -diagonal 1]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test diag-2.4 {Named parameter with negative diagonal offset} {
    set matrix [create_test_matrix]
    set result [torch::diag -input $matrix -diagonal -1]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test diag-2.5 {Named parameters in different order} {
    set matrix [create_test_matrix]
    set result [torch::diag -diagonal 1 -input $matrix]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

# Test camelCase alias (note: diag is already camelCase)
test diag-3.1 {CamelCase command (same as snake_case for diag)} {
    set vector [create_test_vector]
    set result [torch::diag $vector]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

test diag-3.2 {CamelCase with named parameters} {
    set matrix [create_test_matrix]
    set result [torch::diag -input $matrix -diagonal 1]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

# Test mathematical correctness
test diag-4.1 {Mathematical correctness - vector to diagonal matrix} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::diag $vector]
    
    # Result should be 2x2 matrix
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2 2"}
} {1}

test diag-4.2 {Mathematical correctness - matrix diagonal extraction} {
    # Create identity-like matrix
    set matrix [torch::tensor_create -data {1.0 0.0 0.0 1.0} -shape {2 2} -dtype float32]
    set result [torch::diag $matrix]
    
    # Result should be 1D vector with shape {2}
    set shape [torch::tensor_shape $result]
    expr {$shape eq "2"}
} {1}

test diag-4.3 {Diagonal offset behavior} {
    # Create 4x4 matrix
    set data {}
    for {set i 0} {$i < 16} {incr i} {
        lappend data [expr {$i + 1}]
    }
    set matrix [torch::tensor_create -data $data -shape {4 4} -dtype float32]
    
    # Extract upper diagonal (offset +1)
    set result [torch::diag $matrix 1]
    set shape [torch::tensor_shape $result]
    
    # Should give 3 elements (positions [0,1], [1,2], [2,3])
    expr {$shape eq "3"}
} {1}

# Test consistency between syntaxes
test diag-5.1 {Consistency between positional and named syntax} {
    set vector [create_test_vector]
    
    set result1 [torch::diag $vector]
    set result2 [torch::diag -input $vector]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diag_result $result1]
    set valid2 [verify_diag_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

test diag-5.2 {Consistency with diagonal offset} {
    set matrix [create_test_matrix]
    
    set result1 [torch::diag $matrix 2]
    set result2 [torch::diag -input $matrix -diagonal 2]
    
    # Both should be valid tensors with same shape
    set valid1 [verify_diag_result $result1]
    set valid2 [verify_diag_result $result2]
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    
    expr {$valid1 && $valid2 && $shape1 eq $shape2}
} {1}

# Test data type support
test diag-6.1 {Float32 tensor support} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float32]
    set result [torch::diag -input $vector]
    expr {[verify_diag_result $result] && [torch::tensor_dtype $result] eq "Float32"}
} {1}

test diag-6.2 {Float64 tensor support} {
    set vector [torch::tensor_create -data {1.0 2.0} -shape {2} -dtype float64]
    set result [torch::diag -input $vector]
    expr {[verify_diag_result $result] && [torch::tensor_dtype $result] eq "Float64"}
} {1}

test diag-6.3 {Integer tensor support} {
    set vector [torch::tensor_create -data {1 2 3} -shape {3} -dtype int32]
    set result [torch::diag $vector]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "3 3"}
} {1}

# Test edge cases
test diag-7.1 {Single element vector} {
    set vector [torch::tensor_create -data {5.0} -shape {1} -dtype float32]
    set result [torch::diag $vector]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "1 1"}
} {1}

test diag-7.2 {Single element matrix} {
    set matrix [torch::tensor_create -data {7.0} -shape {1 1} -dtype float32]
    set result [torch::diag $matrix]
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "1"}
} {1}

test diag-7.3 {Large diagonal offset} {
    set matrix [create_test_matrix]
    set result [torch::diag $matrix 5]
    # Should work but return empty tensor for offset beyond matrix size
    verify_diag_result $result
} {1}

test diag-7.4 {Zero diagonal offset explicitly} {
    set matrix [create_test_matrix]
    set result1 [torch::diag $matrix]
    set result2 [torch::diag $matrix 0]
    
    # Both should give same result
    set shape1 [torch::tensor_shape $result1]
    set shape2 [torch::tensor_shape $result2]
    expr {$shape1 eq $shape2}
} {1}

test diag-7.5 {Non-square matrix} {
    # Create 2x3 matrix
    set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {2 3} -dtype float32]
    set result [torch::diag $matrix]
    # Should extract diagonal with min(rows, cols) elements
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

test diag-7.6 {3x2 matrix (more rows than columns)} {
    # Create 3x2 matrix
    set matrix [torch::tensor_create -data {1.0 2.0 3.0 4.0 5.0 6.0} -shape {3 2} -dtype float32]
    set result [torch::diag $matrix]
    # Should extract diagonal with min(rows, cols) elements
    expr {[verify_diag_result $result] && [torch::tensor_shape $result] eq "2"}
} {1}

# Test error handling
test diag-8.1 {Invalid tensor name positional} {
    catch {torch::diag invalid_tensor} msg
    expr {[string match "*Invalid input tensor*" $msg]}
} {1}

test diag-8.2 {Invalid tensor name named parameter} {
    catch {torch::diag -input invalid_tensor} msg
    expr {[string match "*Invalid input tensor*" $msg]}
} {1}

test diag-8.3 {Missing required parameters} {
    catch {torch::diag} msg
    expr {[string match "*Usage*" $msg] || [string match "*missing*" $msg]}
} {1}

test diag-8.4 {Invalid parameter name} {
    set vector [create_test_vector]
    catch {torch::diag -invalid $vector} msg
    expr {[string match "*Unknown parameter*" $msg]}
} {1}

test diag-8.5 {Missing value for named parameter} {
    catch {torch::diag -input} msg
    expr {[string match "*Missing value*" $msg] || [string match "*Usage*" $msg]}
} {1}

test diag-8.6 {Invalid diagonal value positional} {
    set matrix [create_test_matrix]
    catch {torch::diag $matrix invalid_int} msg
    expr {[string match "*Invalid diagonal*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diag-8.7 {Invalid diagonal value named} {
    set matrix [create_test_matrix]
    catch {torch::diag -input $matrix -diagonal invalid_int} msg
    expr {[string match "*Invalid diagonal*" $msg] || [string match "*expected integer*" $msg]}
} {1}

test diag-8.8 {Too many positional arguments} {
    set matrix [create_test_matrix]
    catch {torch::diag $matrix 1 extra_arg} msg
    expr {[string match "*Usage*" $msg] || [string match "*wrong # args*" $msg]}
} {1}

test diag-8.9 {Missing required named parameter} {
    catch {torch::diag -diagonal 1} msg
    expr {[string match "*Required parameter missing*" $msg]}
} {1}

cleanupTests 