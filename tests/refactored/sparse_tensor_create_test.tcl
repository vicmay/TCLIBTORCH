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
proc create_test_tensors {} {
    set indices [torch::tensor_create {{0 0} {1 1} {2 2}} -dtype "int64"]
    set values [torch::tensor_create {1.0 2.0 3.0}]
    return [list $indices $values]
}

# Test cases for positional syntax with snake_case
test sparse_tensor_create-1.1 {Basic sparse_tensor_create} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_tensor_create $indices $values {3 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_tensor_create-1.2 {Verify sparse tensor properties} {
    lassign [create_test_tensors] indices values
    set sparse_tensor [torch::sparse_tensor_create $indices $values {3 3}]
    set dense_tensor [torch::sparse_to_dense $sparse_tensor]
    set expected [torch::tensor_create {{1.0 0.0 0.0} {0.0 2.0 0.0} {0.0 0.0 3.0}}]
    expr {[torch::allclose $dense_tensor $expected]}
} {1}

# Test cases for named parameter syntax with snake_case
test sparse_tensor_create-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_tensor_create -indices $indices -values $values -size {3 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_tensor_create-2.2 {Verify named parameter result} {
    lassign [create_test_tensors] indices values
    set sparse_tensor [torch::sparse_tensor_create -indices $indices -values $values -size {3 3}]
    set dense_tensor [torch::sparse_to_dense $sparse_tensor]
    set expected [torch::tensor_create {{1.0 0.0 0.0} {0.0 2.0 0.0} {0.0 0.0 3.0}}]
    expr {[torch::allclose $dense_tensor $expected]}
} {1}

# Test cases for camelCase alias
test sparse_tensor_create-3.1 {camelCase - positional syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparseTensorCreate $indices $values {3 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_tensor_create-3.2 {camelCase - named parameter syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparseTensorCreate -indices $indices -values $values -size {3 3}]
    expr {[string match "tensor*" $result]}
} {1}

# Error handling tests
test sparse_tensor_create-4.1 {Invalid indices tensor} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_tensor_create "invalid_tensor" $values {3 3}} err
    set err
} {Invalid tensor handle}

test sparse_tensor_create-4.2 {Invalid values tensor} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_tensor_create $indices "invalid_tensor" {3 3}} err
    set err
} {Invalid tensor handle}

test sparse_tensor_create-4.3 {Invalid size format} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_tensor_create $indices $values "not_a_list"} err
    set err
} {expected list but got "not_a_list"}

test sparse_tensor_create-4.4 {Too few arguments} {
    catch {torch::sparse_tensor_create} err
    set err
} {Usage: torch::sparse_tensor_create indices values size
   or: torch::sparse_tensor_create -indices TENSOR -values TENSOR -size LIST}

test sparse_tensor_create-4.5 {Missing named parameter value} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_tensor_create -indices $indices -values} err
    set err
} {Missing value for parameter}

test sparse_tensor_create-4.6 {Unknown named parameter} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_tensor_create -indices $indices -values $values -unknown value} err
    set err
} {Unknown parameter: -unknown}

cleanupTests 