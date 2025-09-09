#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

;# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

;# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

;# Helper function to create a sparse tensor for testing
proc create_test_sparse_tensor {} {
    ;# Create indices tensor with shape [2,3] where:
    ;# - First dimension (2) is number of dimensions in sparse tensor
    ;# - Second dimension (3) is number of non-zero elements
    ;# Values represent positions: [[0,1,0], [0,1,2]] means elements at (0,0), (1,1), (0,2)
    set indices [torch::tensor_create {{0 1 0} {0 1 2}} -dtype "int64"]
    set values [torch::tensor_create {1.0 2.0 3.0}]
    set size {3 3}
    return [torch::sparse_coo_tensor $indices $values $size]
}

;# Helper function to verify tensor equality with tolerance
proc tensor_equal {tensor1 tensor2} {
    set dense1 [torch::sparse_to_dense $tensor1]
    set dense2 [torch::sparse_to_dense $tensor2]
    return [torch::allclose $dense1 $dense2]
}

;# Test cases for positional syntax with snake_case
test sparse_sum-1.1 {Basic sparse_sum - all dimensions} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_sum $sparse_tensor]
    set dense_result [torch::sparse_to_dense $result]
    expr {abs([torch::tensor_item $dense_result] - 6.0) < 1e-6}
} {1}

test sparse_sum-1.2 {sparse_sum along dimension 0} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_sum $sparse_tensor 0]
    ;# Sum along dimension 0 (rows) should give [1.0 2.0 3.0]
    ;# Because we have:
    ;# [1.0 0.0 3.0]  row 0
    ;# [0.0 2.0 0.0]  row 1
    ;# [0.0 0.0 0.0]  row 2
    set expected [torch::tensor_create {1.0 2.0 3.0}]
    tensor_equal $result $expected
} {1}

test sparse_sum-1.3 {sparse_sum along dimension 1} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_sum $sparse_tensor 1]
    ;# Sum along dimension 1 (columns) should give [4.0 2.0 0.0]
    ;# Because we have:
    ;# [1.0 0.0 3.0]
    ;# [0.0 2.0 0.0]
    ;# [0.0 0.0 0.0]
    set expected [torch::tensor_create {4.0 2.0 0.0}]
    tensor_equal $result $expected
} {1}

;# Test cases for named parameter syntax with snake_case
test sparse_sum-2.1 {Named parameter syntax - all dimensions} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_sum -input $sparse_tensor]
    set dense_result [torch::sparse_to_dense $result]
    expr {abs([torch::tensor_item $dense_result] - 6.0) < 1e-6}
} {1}

test sparse_sum-2.2 {Named parameter syntax - with dimension} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_sum -input $sparse_tensor -dim 0]
    ;# Sum along dimension 0 (rows) should give [1.0 2.0 3.0]
    set expected [torch::tensor_create {1.0 2.0 3.0}]
    tensor_equal $result $expected
} {1}

;# Test cases for camelCase alias
test sparse_sum-3.1 {camelCase - all dimensions} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparseSum -input $sparse_tensor]
    set dense_result [torch::sparse_to_dense $result]
    expr {abs([torch::tensor_item $dense_result] - 6.0) < 1e-6}
} {1}

test sparse_sum-3.2 {camelCase - with dimension} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparseSum -input $sparse_tensor -dim 1]
    ;# Sum along dimension 1 (columns) should give [4.0 2.0 0.0]
    set expected [torch::tensor_create {4.0 2.0 0.0}]
    tensor_equal $result $expected
} {1}

;# Error handling tests
test sparse_sum-4.1 {Invalid tensor} {
    catch {torch::sparse_sum "invalid_tensor"} err
    set err
} {Invalid sparse tensor}

test sparse_sum-4.2 {Invalid dimension type} {
    set sparse_tensor [create_test_sparse_tensor]
    catch {torch::sparse_sum $sparse_tensor "not_a_number"} err
    set err
} {expected integer but got "not_a_number"}

test sparse_sum-4.3 {Too many arguments} {
    set sparse_tensor [create_test_sparse_tensor]
    catch {torch::sparse_sum $sparse_tensor 0 extra_arg} err
    set err
} {wrong # args: should be "torch::sparse_sum sparse_tensor ?dim?"}

cleanupTests 