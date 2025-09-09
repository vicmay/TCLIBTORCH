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
    ;# Create crow_indices tensor (2 x 3 matrix)
    set crow_indices [torch::tensor_create -data {0 2 3} -dtype "int64"]
    
    ;# Create col_indices tensor (2 x 3 matrix)
    set col_indices [torch::tensor_create -data {1 2 0} -dtype "int64"]
    
    ;# Create values tensor
    set values [torch::tensor_create -data {1.0 2.0 3.0} -dtype "float32"]
    
    return [list $crow_indices $col_indices $values]
}

# Test cases for positional syntax
test sparse_csr_tensor-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor $crow_indices $col_indices $values {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-1.2 {Positional syntax with dtype} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor $crow_indices $col_indices $values {2 3} "float64"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-1.3 {Positional syntax with dtype and device} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor $crow_indices $col_indices $values {2 3} "float32" "cpu"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-1.4 {Positional syntax with all parameters} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor $crow_indices $col_indices $values {2 3} "float32" "cpu" 1]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for named parameter syntax
test sparse_csr_tensor-2.1 {Named parameter syntax - minimal} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-2.2 {Named parameter syntax - with dtype} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3} -dtype "float64"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-2.3 {Named parameter syntax - with device} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3} -device "cpu"]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-2.4 {Named parameter syntax - with requires_grad} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3} -requires_grad 1]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-2.5 {Named parameter syntax - all parameters} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3} -dtype "float64" -device "cpu" -requires_grad 1]
    expr {[string match "tensor*" $result]}
} {1}

# Test cases for camelCase alias
test sparse_csr_tensor-3.1 {CamelCase alias - positional syntax} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparseCsrTensor $crow_indices $col_indices $values {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

test sparse_csr_tensor-3.2 {CamelCase alias - named parameter syntax} {
    lassign [create_test_tensors] crow_indices col_indices values
    set result [torch::sparseCsrTensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3}]
    expr {[string match "tensor*" $result]}
} {1}

# Error handling tests
test sparse_csr_tensor-4.1 {Error - missing required parameters} {
    catch {torch::sparse_csr_tensor} result
    set result
} {}

test sparse_csr_tensor-4.2 {Error - invalid crow_indices tensor} {
    lassign [create_test_tensors] crow_indices col_indices values
    catch {torch::sparse_csr_tensor "invalid_tensor" $col_indices $values {2 3}} result
    set result
} {Invalid crow_indices tensor}

test sparse_csr_tensor-4.3 {Error - invalid col_indices tensor} {
    lassign [create_test_tensors] crow_indices col_indices values
    catch {torch::sparse_csr_tensor $crow_indices "invalid_tensor" $values {2 3}} result
    set result
} {Invalid col_indices tensor}

test sparse_csr_tensor-4.4 {Error - invalid values tensor} {
    lassign [create_test_tensors] crow_indices col_indices values
    catch {torch::sparse_csr_tensor $crow_indices $col_indices "invalid_tensor" {2 3}} result
    set result
} {Invalid values tensor}

test sparse_csr_tensor-4.5 {Error - invalid dtype} {
    lassign [create_test_tensors] crow_indices col_indices values
    catch {torch::sparse_csr_tensor -crow_indices $crow_indices -col_indices $col_indices -values $values -size {2 3} -dtype "invalid"} result
    set result
} {Unknown scalar type: invalid}

cleanupTests 