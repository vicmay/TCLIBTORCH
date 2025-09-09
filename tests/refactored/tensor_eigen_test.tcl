#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper function to create a test matrix
proc create_test_matrix {} {
    set tensor [torch::tensor_create -data {4 1 1 4} -shape {2 2}]
    return $tensor
}

;# Test cases for positional syntax (backward compatibility)
test tensor-eigen-1.1 {Basic positional syntax} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen $tensor]
    
    ;# Parse the result string to get eigenvalues and eigenvectors
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    ;# Check that we got valid tensor handles
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    
    ;# Should have 2 eigenvalues and 2x2 eigenvectors
    return [list [llength $eigenvals_data] [llength $eigenvecs_data]]
} {2 4}

test tensor-eigen-1.2 {Positional syntax with symmetric matrix} {
    ;# Create a symmetric matrix
    set tensor [torch::tensor_create -data {2 1 1 2} -shape {2 2}]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    ;# Check that eigenvalues are real and positive (for symmetric matrix)
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set all_positive [expr {[lindex $eigenvals_data 0] > 0 && [lindex $eigenvals_data 1] > 0}]
    
    return $all_positive
} {1}

;# Test cases for named parameter syntax
test tensor-eigen-2.1 {Named parameter syntax with -input} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen -input $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    set eigenvals_data [torch::tensor_to_list $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list $eigenvectors]
    
    return [list [llength $eigenvals_data] [llength $eigenvecs_data]]
} {2 4}

test tensor-eigen-2.2 {Named parameter syntax with -tensor} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen -tensor $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    
    return [list [llength $eigenvals_data] [llength $eigenvecs_data]]
} {2 4}

test tensor-eigen-2.3 {Named parameter syntax with identity matrix} {
    ;# Create identity matrix
    set tensor [torch::tensor_create -data {1 0 0 1} -shape {2 2}]
    set result [torch::tensor_eigen -input $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    ;# For identity matrix, eigenvalues should be 1
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set all_ones [expr {abs([lindex $eigenvals_data 0] - 1.0) < 1e-6 && abs([lindex $eigenvals_data 1] - 1.0) < 1e-6}]
    
    return $all_ones
} {1}

;# Test cases for camelCase alias
test tensor-eigen-3.1 {CamelCase alias basic functionality} {
    set tensor [create_test_matrix]
    set result [torch::tensorEigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    
    return [list [llength $eigenvals_data] [llength $eigenvecs_data]]
} {2 4}

test tensor-eigen-3.2 {CamelCase alias with named parameters} {
    set tensor [create_test_matrix]
    set result [torch::tensorEigen -input $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    
    return [list [llength $eigenvals_data] [llength $eigenvecs_data]]
} {2 4}

test tensor-eigen-3.3 {CamelCase alias with diagonal matrix} {
    ;# Create diagonal matrix
    set tensor [torch::tensor_create -data {3 0 0 5} -shape {2 2}]
    set result [torch::tensorEigen -input $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    ;# For diagonal matrix, eigenvalues should be diagonal elements
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set correct_eigenvals [expr {abs([lindex $eigenvals_data 0] - 3.0) < 1e-6 && abs([lindex $eigenvals_data 1] - 5.0) < 1e-6}]
    
    return $correct_eigenvals
} {1}

;# Error handling tests
test tensor-eigen-4.1 {Error handling - missing tensor} {
    set result [catch {torch::tensor_eigen} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Required parameter m}}

test tensor-eigen-4.2 {Error handling - invalid tensor name} {
    set result [catch {torch::tensor_eigen invalid_tensor} msg]
    return [list $result $msg]
} {1 {Invalid tensor name}}

test tensor-eigen-4.3 {Error handling - missing named parameter value} {
    set result [catch {torch::tensor_eigen -input} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Missing value for pa}}

test tensor-eigen-4.4 {Error handling - unknown named parameter} {
    set tensor [create_test_matrix]
    set result [catch {torch::tensor_eigen -unknown $tensor} msg]
    return [list $result [string range $msg 0 19]]
} {1 {Unknown parameter: -}}

test tensor-eigen-4.5 {Error handling - non-square matrix} {
    ;# Create a non-square matrix
    set tensor [torch::tensor_create -data {1 2 3 4 5 6} -shape {2 3}]
    set result [catch {torch::tensor_eigen $tensor} msg]
    return [expr {$result && [string match "linalg.eigh: A must*" $msg]}]
} {1}

;# Mathematical correctness tests
test tensor-eigen-5.1 {Mathematical correctness - eigenvalue equation} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvectors [lindex $result 3]
    
    ;# Get the data
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    set tensor_data [torch::tensor_to_list -input $tensor]
    
    ;# Check that A*v = λ*v for first eigenvector
    set lambda1 [lindex $eigenvals_data 0]
    set v1 [list [lindex $eigenvecs_data 0] [lindex $eigenvecs_data 1]]
    
    ;# A*v1
    set av1_0 [expr {[lindex $tensor_data 0] * [lindex $v1 0] + [lindex $tensor_data 1] * [lindex $v1 1]}]
    set av1_1 [expr {[lindex $tensor_data 2] * [lindex $v1 0] + [lindex $tensor_data 3] * [lindex $v1 1]}]
    
    ;# λ*v1
    set lambda_v1_0 [expr {$lambda1 * [lindex $v1 0]}]
    set lambda_v1_1 [expr {$lambda1 * [lindex $v1 1]}]
    
    ;# Check if they're approximately equal
    set diff0 [expr {abs($av1_0 - $lambda_v1_0)}]
    set diff1 [expr {abs($av1_1 - $lambda_v1_1)}]
    
    return [expr {$diff0 < 1e-6 && $diff1 < 1e-6}]
} {1}

test tensor-eigen-5.2 {Mathematical correctness - eigenvectors orthogonality} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvectors [lindex $result 3]
    set eigenvecs_data [torch::tensor_to_list -input $eigenvectors]
    
    ;# Get the two eigenvectors
    set v1 [list [lindex $eigenvecs_data 0] [lindex $eigenvecs_data 1]]
    set v2 [list [lindex $eigenvecs_data 2] [lindex $eigenvecs_data 3]]
    
    ;# Check orthogonality: v1^T * v2 should be close to 0
    set dot_product [expr {[lindex $v1 0] * [lindex $v2 0] + [lindex $v1 1] * [lindex $v2 1]}]
    
    return [expr {abs($dot_product) < 1e-6}]
} {1}

test tensor-eigen-5.3 {Mathematical correctness - trace equals sum of eigenvalues} {
    set tensor [create_test_matrix]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    set tensor_data [torch::tensor_to_list -input $tensor]
    
    ;# Calculate trace
    set trace [expr {[lindex $tensor_data 0] + [lindex $tensor_data 3]}]
    
    ;# Sum of eigenvalues
    set sum_eigenvals [expr {[lindex $eigenvals_data 0] + [lindex $eigenvals_data 1]}]
    
    ;# Check if they're approximately equal
    set diff [expr {abs($trace - $sum_eigenvals)}]
    
    return [expr {$diff < 1e-6}]
} {1}

;# Edge cases
test tensor-eigen-6.1 {Edge case - zero matrix} {
    set tensor [torch::tensor_create -data {0 0 0 0} -shape {2 2}]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    
    ;# All eigenvalues should be zero
    set all_zero [expr {abs([lindex $eigenvals_data 0]) < 1e-6 && abs([lindex $eigenvals_data 1]) < 1e-6}]
    
    return $all_zero
} {1}

test tensor-eigen-6.2 {Edge case - single element matrix} {
    set tensor [torch::tensor_create -data {42} -shape {1 1}]
    set result [torch::tensor_eigen $tensor]
    
    set eigenvalues [lindex $result 1]
    set eigenvals_data [torch::tensor_to_list -input $eigenvalues]
    
    ;# Single eigenvalue should be 42
    return [expr {abs([lindex $eigenvals_data 0] - 42.0) < 1e-6}]
} {1}

;# Syntax consistency tests
test tensor-eigen-7.1 {Syntax consistency - both syntaxes produce same result} {
    set tensor [create_test_matrix]
    
    set result1 [torch::tensor_eigen $tensor]
    set result2 [torch::tensor_eigen -input $tensor]
    
    set eigenvals1 [lindex $result1 1]
    set eigenvals2 [lindex $result2 1]
    
    set data1 [torch::tensor_to_list -input $eigenvals1]
    set data2 [torch::tensor_to_list -input $eigenvals2]
    
    ;# Check if eigenvalues are the same (within tolerance)
    set diff0 [expr {abs([lindex $data1 0] - [lindex $data2 0])}]
    set diff1 [expr {abs([lindex $data1 1] - [lindex $data2 1])}]
    
    return [expr {$diff0 < 1e-6 && $diff1 < 1e-6}]
} {1}

test tensor-eigen-7.2 {Syntax consistency - camelCase produces same result} {
    set tensor [create_test_matrix]
    
    set result1 [torch::tensor_eigen $tensor]
    set result2 [torch::tensorEigen $tensor]
    
    set eigenvals1 [lindex $result1 1]
    set eigenvals2 [lindex $result2 1]
    
    set data1 [torch::tensor_to_list -input $eigenvals1]
    set data2 [torch::tensor_to_list -input $eigenvals2]
    
    ;# Check if eigenvalues are the same (within tolerance)
    set diff0 [expr {abs([lindex $data1 0] - [lindex $data2 0])}]
    set diff1 [expr {abs([lindex $data1 1] - [lindex $data2 1])}]
    
    return [expr {$diff0 < 1e-6 && $diff1 < 1e-6}]
} {1}

cleanupTests 