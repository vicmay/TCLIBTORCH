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

# Helper function to check if tensors are approximately equal
proc tensorsApproxEqual {tensor1 tensor2 tolerance} {
    set diff [torch::tensor_sub $tensor1 $tensor2]
    set abs_diff [torch::tensor_abs $diff]
    set max_diff [torch::tensor_max $abs_diff]
    set max_val [torch::tensor_item $max_diff]
    return [expr {$max_val < $tolerance}]
}

# Create a test tensor
set t [torch::ones -shape {2 3} -dtype float32]

# Test cases for positional syntax

# mean along dim 1 (columns), keepdim false
set result1 [torch::mean_dim $t 1]
set expected1 [torch::tensor_mean $t 1]
test mean_dim-1.1 {Positional syntax, keepdim default} {
    tensorsApproxEqual $result1 $expected1 1e-6
} {1}

# mean along dim 0 (rows), keepdim true
set result2 [torch::mean_dim $t 0 1]
# torch::tensor_mean does not support keepdim, so we cannot compare shapes for keepdim=1

# Test cases for named parameter syntax
set result3 [torch::mean_dim -input $t -dim 1]
set expected3 [torch::tensor_mean -input $t -dim 1]
test mean_dim-2.1 {Named parameter syntax, keepdim default} {
    tensorsApproxEqual $result3 $expected3 1e-6
} {1}

set result4 [torch::mean_dim -input $t -dim 0 -keepdim 1]
# torch::tensor_mean does not support keepdim, so we cannot compare shapes for keepdim=1

# Test cases for camelCase alias
set result5 [torch::meanDim -input $t -dim 1]
set expected5 [torch::tensor_mean -input $t -dim 1]
test mean_dim-3.1 {CamelCase alias, named syntax} {
    tensorsApproxEqual $result5 $expected5 1e-6
} {1}

set result6 [torch::meanDim $t 1]
set expected6 [torch::tensor_mean $t 1]
test mean_dim-3.2 {CamelCase alias, positional syntax} {
    tensorsApproxEqual $result6 $expected6 1e-6
} {1}

# Error handling tests
# Invalid tensor name
catch {torch::mean_dim invalid 1} err1

# Missing required parameter
catch {torch::mean_dim} err2

# Invalid dim type
catch {torch::mean_dim -input $t -dim foo} err3

test mean_dim-4.1 {Error: invalid tensor name} {
    string match *Invalid* $err1
} {1}

test mean_dim-4.2 {Error: missing required parameter} {
    string match *number* $err2
} {1}

test mean_dim-4.3 {Error: invalid dim type} {
    string match *Invalid* $err3
} {1}

cleanupTests 