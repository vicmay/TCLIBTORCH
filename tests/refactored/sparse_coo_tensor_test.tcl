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
    ;# Create indices tensor (2x3 matrix)
    set indices [torch::tensor_create -data {0 1 1 1 0 2} -dtype int64 -shape {2 3}]
    
    ;# Create values tensor (3 values)
    set values [torch::tensor_create {1.0 2.0 3.0} float32]
    
    return [list $indices $values]
}

# Test cases for positional syntax
test sparse_coo_tensor-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor $indices $values {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-1.2 {Positional syntax with dtype} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor $indices $values {2 3} float64]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-1.3 {Positional syntax with dtype and device} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor $indices $values {2 3} float32 cpu]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-1.4 {Positional syntax with all parameters} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor $indices $values {2 3} float32 cpu 1]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for named parameter syntax
test sparse_coo_tensor-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor -indices $indices -values $values -size {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-2.2 {Named parameter syntax with dtype} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor -indices $indices -values $values -size {2 3} -dtype float64]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-2.3 {Named parameter syntax with dtype and device} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor -indices $indices -values $values -size {2 3} -dtype float32 -device cpu]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-2.4 {Named parameter syntax with all parameters} {
    lassign [create_test_tensors] indices values
    set result [torch::sparse_coo_tensor -indices $indices -values $values -size {2 3} -dtype float32 -device cpu -requires_grad 1]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for camelCase alias
test sparse_coo_tensor-3.1 {CamelCase alias with positional syntax} {
    lassign [create_test_tensors] indices values
    set result [torch::sparseCooTensor $indices $values {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_coo_tensor-3.2 {CamelCase alias with named parameters} {
    lassign [create_test_tensors] indices values
    set result [torch::sparseCooTensor -indices $indices -values $values -size {2 3} -dtype float32]
    expr {[string match "tensor*" $result]}
} {1}

# Error handling tests
test sparse_coo_tensor-4.1 {Error: Invalid indices tensor} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_coo_tensor invalid_tensor $values {2 3}} result
    set result
} {Invalid indices tensor}

test sparse_coo_tensor-4.2 {Error: Invalid values tensor} {
    lassign [create_test_tensors] indices values
    catch {torch::sparse_coo_tensor $indices invalid_tensor {2 3}} result
    set result
} {Invalid values tensor}

test sparse_coo_tensor-4.3 {Error: Missing required parameters} {
    catch {torch::sparse_coo_tensor} result
    set result
} {wrong # args: should be "torch::sparse_coo_tensor indices values size ?dtype? ?device? ?requires_grad?"}

cleanupTests 