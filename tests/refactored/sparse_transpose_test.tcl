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
    set indices [torch::tensor_create -data {{0 1 2} {0 1 2}} -dtype "int64"]
    set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]
    set size {3 3}
    return [torch::sparse_coo_tensor $indices $values $size]
}

# Helper function to check if tensors are approximately equal
proc tensors_approx_equal {tensor1 tensor2} {
    # For numeric tensors, use isclose to find differences
    set result [torch::isclose -input $tensor1 -other $tensor2]
    # Convert to float for summation
    set result_float [torch::tensor_to -input $result -dtype float32 -device cpu]
    # Sum all elements (true = 1.0, false = 0.0)
    set sum_result [torch::tensor_sum $result_float]
    set total_sum [torch::tensor_item $sum_result]
    # Get total number of elements
    set numel [torch::tensor_numel $result]
    # All elements are true if sum equals total number of elements
    return [expr {$total_sum == $numel}]
}

# Test cases for positional syntax
test sparse-transpose-1.1 {Basic positional syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_transpose $sparse_tensor 0 1]
    set dense_result [torch::sparse_to_dense $result]
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 2.0 0.0} {0.0 0.0 3.0}}]
    tensors_approx_equal $dense_result $expected
} {1}

test sparse-transpose-1.2 {Error handling - invalid dimensions} {
    set sparse_tensor [create_test_sparse_tensor]
    catch {torch::sparse_transpose $sparse_tensor 0 2} err
    set err
} {Invalid dimension}

test sparse-transpose-1.3 {Error handling - invalid tensor} {
    catch {torch::sparse_transpose "invalid_tensor" 0 1} err
    set err
} {Invalid sparse tensor}

# Test cases for named parameter syntax
test sparse-transpose-2.1 {Named parameter syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparse_transpose -tensor $sparse_tensor -dim0 0 -dim1 1]
    set dense_result [torch::sparse_to_dense $result]
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 2.0 0.0} {0.0 0.0 3.0}}]
    tensors_approx_equal $dense_result $expected
} {1}

test sparse-transpose-2.2 {Named parameter syntax - error handling} {
    set sparse_tensor [create_test_sparse_tensor]
    catch {torch::sparse_transpose -tensor $sparse_tensor -dim0 0} err
    set err
} {Missing required parameter: dim1}

# Test cases for camelCase alias
test sparse-transpose-3.1 {CamelCase alias} {
    set sparse_tensor [create_test_sparse_tensor]
    set result [torch::sparseTranspose -tensor $sparse_tensor -dim0 0 -dim1 1]
    set dense_result [torch::sparse_to_dense $result]
    set expected [torch::tensor_create -data {{1.0 0.0 0.0} {0.0 2.0 0.0} {0.0 0.0 3.0}}]
    tensors_approx_equal $dense_result $expected
} {1}

cleanupTests 