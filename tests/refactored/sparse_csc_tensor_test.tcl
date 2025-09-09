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
    ;# Create ccol_indices tensor (2 x 3 matrix)
    set ccol_indices [torch::tensor_create -data {0 1 2} -dtype "int64"]
    
    ;# Create row_indices tensor (2 x 3 matrix)
    set row_indices [torch::tensor_create -data {0 1 0} -dtype "int64"]
    
    ;# Create values tensor
    set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]
    
    return [list $ccol_indices $row_indices $values]
}

# Test cases for positional syntax
test sparse_csc_tensor-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor $ccol_indices $row_indices $values {2 2}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-1.2 {Positional syntax with dtype} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor $ccol_indices $row_indices $values {2 2} "float64"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-1.3 {Positional syntax with dtype and device} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor $ccol_indices $row_indices $values {2 2} "float32" "cpu"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-1.4 {Positional syntax with all parameters} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor $ccol_indices $row_indices $values {2 2} "float32" "cpu" 0]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for named parameter syntax
test sparse_csc_tensor-2.1 {Named parameter syntax - minimal} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor -ccol_indices $ccol_indices -row_indices $row_indices -values $values -size {2 2}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-2.2 {Named parameter syntax - with dtype} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor -ccol_indices $ccol_indices -row_indices $row_indices -values $values -size {2 2} -dtype "float64"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-2.3 {Named parameter syntax - with device} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor -ccol_indices $ccol_indices -row_indices $row_indices -values $values -size {2 2} -device "cpu"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-2.4 {Named parameter syntax - all parameters} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparse_csc_tensor -ccol_indices $ccol_indices -row_indices $row_indices -values $values -size {2 2} -dtype "float32" -device "cpu" -requires_grad 0]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for camelCase alias
test sparse_csc_tensor-3.1 {CamelCase alias - basic usage} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparseCscTensor $ccol_indices $row_indices $values {2 2}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csc_tensor-3.2 {CamelCase alias - named parameters} {
    lassign [create_test_tensors] ccol_indices row_indices values
    set result [torch::sparseCscTensor -ccol_indices $ccol_indices -row_indices $row_indices -values $values -size {2 2}]
    expr {[string match "tensor*" $result]}
} {1}

# Error handling tests
test sparse_csc_tensor-4.1 {Error - missing required parameters} {
    catch {torch::sparse_csc_tensor} result
    set result
} {}

test sparse_csc_tensor-4.2 {Error - invalid ccol_indices tensor} {
    lassign [create_test_tensors] _ row_indices values
    catch {torch::sparse_csc_tensor invalid_tensor $row_indices $values {2 2}} result
    set result
} {Invalid ccol_indices tensor}

test sparse_csc_tensor-4.3 {Error - invalid row_indices tensor} {
    lassign [create_test_tensors] ccol_indices _ values
    catch {torch::sparse_csc_tensor $ccol_indices invalid_tensor $values {2 2}} result
    set result
} {Invalid row_indices tensor}

test sparse_csc_tensor-4.4 {Error - invalid values tensor} {
    lassign [create_test_tensors] ccol_indices row_indices _
    catch {torch::sparse_csc_tensor $ccol_indices $row_indices invalid_tensor {2 2}} result
    set result
} {Invalid values tensor}

test sparse_csc_tensor-4.5 {Error - invalid size format} {
    lassign [create_test_tensors] ccol_indices row_indices values
    catch {torch::sparse_csc_tensor $ccol_indices $row_indices $values invalid_size} result
    expr {[string match "*list*" $result]}
} {1}

cleanupTests 