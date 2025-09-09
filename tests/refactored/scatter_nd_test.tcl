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
    set input [torch::tensor_create {1 2 3 4 5 6 7 8} float32]
    set indices [torch::tensor_create {0 4 2 1} int64]
    set updates [torch::tensor_create {10 40 30 20} float32]
    return [list $input $indices $updates]
}

# Test cases for positional syntax
test scatter_nd-1.1 {Basic positional syntax} {
    lassign [create_test_tensors] input indices updates
    set result [torch::scatter_nd $input $indices $updates]
    set values [torch::tensor_to_list $result]
    expr {[lindex $values 0] == 10 && [lindex $values 1] == 20 && \
          [lindex $values 2] == 30 && [lindex $values 4] == 40}
} {1}

# Test cases for named parameter syntax
test scatter_nd-2.1 {Named parameter syntax} {
    lassign [create_test_tensors] input indices updates
    set result [torch::scatter_nd -input $input -indices $indices -updates $updates]
    set values [torch::tensor_to_list $result]
    expr {[lindex $values 0] == 10 && [lindex $values 1] == 20 && \
          [lindex $values 2] == 30 && [lindex $values 4] == 40}
} {1}

# Test cases for camelCase alias
test scatter_nd-3.1 {CamelCase alias} {
    lassign [create_test_tensors] input indices updates
    set result [torch::scatterNd -input $input -indices $indices -updates $updates]
    set values [torch::tensor_to_list $result]
    expr {[lindex $values 0] == 10 && [lindex $values 1] == 20 && \
          [lindex $values 2] == 30 && [lindex $values 4] == 40}
} {1}

# Error handling tests
test scatter_nd-4.1 {Missing required parameters} {
    lassign [create_test_tensors] input indices updates
    catch {torch::scatter_nd -input $input -indices $indices} err
    set err
} {Required parameters missing}

test scatter_nd-4.2 {Invalid tensor} {
    lassign [create_test_tensors] input indices updates
    catch {torch::scatter_nd -input "invalid" -indices $indices -updates $updates} err
    set err
} {Invalid input tensor}

test scatter_nd-4.3 {Unknown parameter} {
    lassign [create_test_tensors] input indices updates
    catch {torch::scatter_nd -input $input -invalid value -indices $indices -updates $updates} err
    set err
} {Unknown parameter: -invalid}

cleanupTests 