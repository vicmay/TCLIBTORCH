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

# Helper to create tensors
proc create_tensor {data {dtype float32}} {
    return [torch::tensor_create -data $data -dtype $dtype]
}

# Test cases for positional syntax
test tensor_where-1.1 {Basic positional syntax} {
    set cond [create_tensor {0 1 0 1} bool]
    set x [create_tensor {10 10 10 10}]
    set y [create_tensor {1 2 3 4}]
    set out [torch::tensor_where $cond $x $y]
    set result [torch::tensor_to_list $out]
    set result
} {1.0 10.0 3.0 10.0}

test tensor_where-1.2 {Positional, all true condition} {
    set cond [create_tensor {1 1 1 1} bool]
    set x [create_tensor {5 6 7 8}]
    set y [create_tensor {0 0 0 0}]
    set out [torch::tensor_where $cond $x $y]
    set result [torch::tensor_to_list $out]
    set result
} {5.0 6.0 7.0 8.0}

test tensor_where-1.3 {Positional, all false condition} {
    set cond [create_tensor {0 0 0 0} bool]
    set x [create_tensor {5 6 7 8}]
    set y [create_tensor {1 2 3 4}]
    set out [torch::tensor_where $cond $x $y]
    set result [torch::tensor_to_list $out]
    set result
} {1.0 2.0 3.0 4.0}

# Test cases for named parameter syntax
test tensor_where-2.1 {Named parameter syntax} {
    set cond [create_tensor {1 0 1 0} bool]
    set x [create_tensor {100 200 300 400}]
    set y [create_tensor {10 20 30 40}]
    set out [torch::tensor_where -condition $cond -x $x -y $y]
    set result [torch::tensor_to_list $out]
    set result
} {100.0 20.0 300.0 40.0}

# Test cases for camelCase alias
test tensor_where-3.1 {CamelCase alias} {
    set cond [create_tensor {0 1 1 0} bool]
    set x [create_tensor {1 2 3 4}]
    set y [create_tensor {9 8 7 6}]
    set out [torch::tensorWhere -condition $cond -x $x -y $y]
    set result [torch::tensor_to_list $out]
    set result
} {9.0 2.0 3.0 6.0}

# Error handling tests
test tensor_where-4.1 {Missing argument error} -returnCodes error -body {
    torch::tensor_where
} -match glob -result {*Required parameters missing: condition, x, y*}

test tensor_where-4.2 {Invalid tensor handle} -returnCodes error -body {
    set x [create_tensor {1 2 3}]
    set y [create_tensor {4 5 6}]
    torch::tensor_where invalid $x $y
} -match glob -result {*Invalid condition tensor name*}

test tensor_where-4.3 {Unknown named parameter} -returnCodes error -body {
    set cond [create_tensor {1 0 1} bool]
    set x [create_tensor {1 2 3}]
    set y [create_tensor {4 5 6}]
    torch::tensor_where -foo $cond -x $x -y $y
} -match glob -result {*Unknown parameter: -foo*}

# Edge case: broadcasting
test tensor_where-5.1 {Broadcasting support} {
    set cond [create_tensor {{1 0} {0 1}} bool]
    set x [create_tensor {10 20}]
    set y [create_tensor {1 2}]
    set out [torch::tensor_where $cond $x $y]
    set result [torch::tensor_to_list $out]
    set result
} {10.0 2.0 1.0 20.0}

cleanupTests 