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

# Helper function to create a sparse tensor
proc create_test_sparse_tensor {} {
    ;# Create indices tensor (2x3 matrix)
    set indices [torch::tensor_create -data {0 1 1 0 0 1} -dtype int64 -shape {2 3}]
    
    ;# Create values tensor (3 elements)
    set values [torch::tensor_create {1.0 2.0 3.0} float32]
    
    ;# Create sparse tensor with size [2, 2]
    set sparse [torch::sparse_coo_tensor $indices $values {2 2}]
    
    return $sparse
}

# Test cases for positional syntax
test sparse-reshape-1.1 {Basic positional syntax} {
    set sparse [create_test_sparse_tensor]
    set reshaped [torch::sparse_reshape $sparse {1 4}]
    set shape [torch::tensor_size $reshaped]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {2 1 4}

test sparse-reshape-1.2 {Positional syntax - same size} {
    set sparse [create_test_sparse_tensor]
    set reshaped [torch::sparse_reshape $sparse {2 2}]
    set shape [torch::tensor_size $reshaped]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {2 2 2}

test sparse-reshape-1.3 {Positional syntax - error on invalid shape} {
    set sparse [create_test_sparse_tensor]
    catch {torch::sparse_reshape $sparse {1 1}} err
    set err
} {Invalid integer in shape list}

# Test cases for named parameter syntax
test sparse-reshape-2.1 {Named parameter syntax} {
    set sparse [create_test_sparse_tensor]
    set reshaped [torch::sparse_reshape -input $sparse -shape {1 4}]
    set shape [torch::tensor_size $reshaped]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {2 1 4}

test sparse-reshape-2.2 {Named parameter syntax - same size} {
    set sparse [create_test_sparse_tensor]
    set reshaped [torch::sparse_reshape -input $sparse -shape {2 2}]
    set shape [torch::tensor_size $reshaped]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {2 2 2}

test sparse-reshape-2.3 {Named parameter syntax - error on invalid shape} {
    set sparse [create_test_sparse_tensor]
    catch {torch::sparse_reshape -input $sparse -shape {1 1}} err
    set err
} {Invalid integer in shape list}

# Test cases for camelCase alias
test sparse-reshape-3.1 {CamelCase alias} {
    set sparse [create_test_sparse_tensor]
    set reshaped [torch::sparseReshape -input $sparse -shape {1 4}]
    set shape [torch::tensor_size $reshaped]
    list [llength $shape] [lindex $shape 0] [lindex $shape 1]
} {2 1 4}

# Error cases
test sparse-reshape-4.1 {Error on missing tensor} {
    catch {torch::sparse_reshape invalid_tensor {1 4}} err
    set err
} {Invalid sparse tensor}

test sparse-reshape-4.2 {Error on missing shape} {
    set sparse [create_test_sparse_tensor]
    catch {torch::sparse_reshape -input $sparse} err
    set err
} {Missing value for parameter}

test sparse-reshape-4.3 {Error on invalid parameter} {
    set sparse [create_test_sparse_tensor]
    catch {torch::sparse_reshape -input $sparse -invalid value} err
    set err
} {Unknown parameter: -invalid}

cleanupTests 