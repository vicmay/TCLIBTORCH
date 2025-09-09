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
    ;# Create indices for a 3D sparse tensor (3 non-zero elements)
    ;# Indices must be in shape (ndim, nnz) = (3, 3) for a 3D tensor with 3 non-zero elements
    set indices [torch::tensor_create {{0 1 2} {0 1 2} {0 1 2}} -dtype "int64"]
    set values [torch::tensor_create {1.0 2.0 3.0}]
    set size {3 3 3}
    if {[catch {
        set sparse_tensor [torch::sparse_coo_tensor $indices $values $size]
    } err]} {
        error "Failed to create sparse tensor: $err"
    }
    return $sparse_tensor
}

# Helper function to create expected dense tensor
proc create_expected_dense_tensor {} {
    # Create a 3x3x3 tensor with all values
    set expected [torch::tensor_create {1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0} -shape {3 3 3}]
    return $expected
}

# Test cases for positional syntax with snake_case
test sparse_tensor_dense-1.1 {Basic sparse_to_dense} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparse_to_dense $sparse_tensor]
    set expected [create_expected_dense_tensor]
    expr {[torch::allclose $dense $expected]}
} {1}

# Test cases for named parameter syntax with snake_case
test sparse_tensor_dense-2.1 {Named parameter syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparse_to_dense -input $sparse_tensor]
    set expected [create_expected_dense_tensor]
    expr {[torch::allclose $dense $expected]}
} {1}

# Test cases for camelCase alias
test sparse_tensor_dense-3.1 {CamelCase alias - positional syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparseToDense $sparse_tensor]
    set expected [create_expected_dense_tensor]
    expr {[torch::allclose $dense $expected]}
} {1}

test sparse_tensor_dense-3.2 {CamelCase alias - named parameter syntax} {
    set sparse_tensor [create_test_sparse_tensor]
    set dense [torch::sparseToDense -input $sparse_tensor]
    set expected [create_expected_dense_tensor]
    expr {[torch::allclose $dense $expected]}
} {1}

# Error handling tests
test sparse_tensor_dense-4.1 {Invalid tensor handle} {
    catch {torch::sparse_to_dense invalid_tensor} err
    set err
} {Invalid sparse tensor}

test sparse_tensor_dense-4.2 {Missing input parameter} {
    catch {torch::sparse_to_dense} err
    set err
} {Usage: torch::sparse_to_dense sparse_tensor
   or: torch::sparse_to_dense -input TENSOR}

test sparse_tensor_dense-4.3 {Invalid named parameter} {
    set sparse_tensor [create_test_sparse_tensor]
    catch {torch::sparse_to_dense -invalid $sparse_tensor} err
    set err
} {Unknown parameter: -invalid}

cleanupTests 