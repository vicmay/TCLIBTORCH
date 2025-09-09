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

# Helper function to check tensor shape
proc check_tensor_shape {tensor expected_shape} {
    set shape [torch::tensor_shape $tensor]
    if {$shape != $expected_shape} {
        error "Expected shape $expected_shape but got $shape"
    }
}

# Test cases for positional syntax
test sparse_mm-1.1 {Basic positional syntax} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    
    # Create a dense tensor
    set dense_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu]
    set dense_tensor [torch::tensor_reshape -input $dense_tensor -shape {2 2}]
    
    # Perform matrix multiplication
    set result [torch::sparse_mm $sparse_tensor $dense_tensor]
    check_tensor_shape $result {2 2}
    expr {1}
} {1}

# Test cases for named parameter syntax
test sparse_mm-2.1 {Named parameter syntax} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    
    # Create a dense tensor
    set dense_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu]
    set dense_tensor [torch::tensor_reshape -input $dense_tensor -shape {2 2}]
    
    # Perform matrix multiplication
    set result [torch::sparse_mm -sparse_tensor $sparse_tensor -dense_tensor $dense_tensor]
    check_tensor_shape $result {2 2}
    expr {1}
} {1}

# Test cases for camelCase alias
test sparse_mm-3.1 {CamelCase alias} {
    # Create a sparse tensor
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    
    # Create a dense tensor
    set dense_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu]
    set dense_tensor [torch::tensor_reshape -input $dense_tensor -shape {2 2}]
    
    # Perform matrix multiplication
    set result [torch::sparseMm -sparse_tensor $sparse_tensor -dense_tensor $dense_tensor]
    check_tensor_shape $result {2 2}
    expr {1}
} {1}

# Error handling tests
test sparse_mm-4.1 {Error: Invalid sparse tensor} {
    set dense_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu]
    set dense_tensor [torch::tensor_reshape -input $dense_tensor -shape {2 2}]
    catch {torch::sparse_mm invalid_tensor $dense_tensor} err
    set err
} {Invalid sparse tensor}

test sparse_mm-4.2 {Error: Invalid dense tensor} {
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    catch {torch::sparse_mm $sparse_tensor invalid_tensor} err
    set err
} {Invalid dense tensor}

test sparse_mm-4.3 {Error: Missing required parameters} {
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    catch {torch::sparse_mm -sparse_tensor $sparse_tensor} err
    set err
} {Required parameters missing: sparse_tensor, dense_tensor}

test sparse_mm-4.4 {Error: Unknown parameter} {
    set indices [torch::tensor_create {0 1 0 1} int64 cpu]
    set indices [torch::tensor_reshape -input $indices -shape {2 2}]
    set values [torch::tensor_create {1.0 2.0} float32 cpu]
    set sparse_tensor [torch::sparse_coo_tensor $indices $values {2 2}]
    
    set dense_tensor [torch::tensor_create {1.0 2.0 3.0 4.0} float32 cpu]
    set dense_tensor [torch::tensor_reshape -input $dense_tensor -shape {2 2}]
    
    catch {torch::sparse_mm -sparse_tensor $sparse_tensor -dense_tensor $dense_tensor -invalid_param 0} err
    set err
} {Unknown parameter: -invalid_param}

cleanupTests 