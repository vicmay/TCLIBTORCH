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

# Helper function to create a test sparse tensor
proc create_test_sparse_tensor {} {
    set indices [torch::tensor_create -data {0 1 2 0 1 2} -dtype int64]
    set indices [torch::tensor_reshape -input $indices -shape {2 3}]
    set values [torch::tensor_create -data {1.0 2.0 3.0}]
    set size {3 3}
    return [torch::sparse_coo_tensor $indices $values $size]
}

# Helper function to check if tensors are approximately equal
proc tensors_approx_equal {tensor1 tensor2} {
    # For numeric tensors, use isclose to find differences
    set result [torch::isclose -input $tensor1 -other $tensor2]
    # Convert to float for summation
    set result_float [torch::tensor_to -input $result -device cpu -dtype float32]
    # Sum all elements (true = 1.0, false = 0.0)
    set sum_result [torch::tensor_sum $result_float]
    set total_sum [torch::tensor_item $sum_result]
    # Get total number of elements
    set numel [torch::tensor_numel $result]
    # All elements are true if sum equals total number of elements
    return [expr {$total_sum == $numel}]
}

# Test cases for positional syntax with snake_case
test sparse_to_dense-1.1 {Basic sparse_to_dense} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparse_to_dense $sparse_tensor]
    set expected [torch::tensor_create -data {1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0}]
    set expected [torch::tensor_reshape -input $expected -shape {3 3}]
    tensors_approx_equal $dense $expected
} {1}

# Test cases for named parameter syntax with snake_case
test sparse_to_dense-2.1 {Named parameter syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparse_to_dense -input $sparse_tensor]
    set expected [torch::tensor_create -data {1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0}]
    set expected [torch::tensor_reshape -input $expected -shape {3 3}]
    tensors_approx_equal $dense $expected
} {1}

# Test cases for camelCase alias
test sparse_to_dense-3.1 {CamelCase alias} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparseToDense -input $sparse_tensor]
    set expected [torch::tensor_create -data {1.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 3.0}]
    set expected [torch::tensor_reshape -input $expected -shape {3 3}]
    tensors_approx_equal $dense $expected
} {1}

# Error handling tests
test sparse_to_dense-4.1 {Error handling - invalid tensor} {
    catch {torch::sparse_to_dense "invalid_tensor"} err
    set err
} {Invalid sparse tensor}

test sparse_to_dense-4.2 {Error handling - missing input} {
    catch {torch::sparse_to_dense} err
    set err
} {Usage: torch::sparse_to_dense sparse_tensor
   or: torch::sparse_to_dense -input TENSOR}

cleanupTests 