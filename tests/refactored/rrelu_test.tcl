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

# Test cases for positional syntax
test rrelu-1.1 {Basic positional syntax} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-1.2 {Positional syntax with lower bound} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu $input 0.1]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-1.3 {Positional syntax with lower and upper bounds} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu $input 0.1 0.3]
    expr {[string match "tensor*" $result]}
} -result {1}

# Note: training mode parameter is not supported in current implementation

# Test cases for named parameter syntax
test rrelu-2.1 {Named parameter syntax basic} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu -input $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-2.2 {Named parameter syntax with lower bound} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu -input $input -lower 0.1]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-2.3 {Named parameter syntax with bounds} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu -input $input -lower 0.1 -upper 0.3]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-2.4 {Named parameter syntax different order} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu -upper 0.3 -input $input -lower 0.1]
    expr {[string match "tensor*" $result]}
} -result {1}

# Test cases for camelCase alias
test rrelu-3.1 {CamelCase alias positional} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rRelu $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rrelu-3.2 {CamelCase alias named parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rRelu -input $input -lower 0.1 -upper 0.3]
    expr {[string match "tensor*" $result]}
} -result {1}

# Error handling tests
test rrelu-4.1 {Error on missing parameters} -body {
    torch::rrelu
} -returnCodes error -result {Usage: torch::rrelu tensor ?lower? ?upper? | torch::rrelu -input tensor ?-lower value? ?-upper value?}

test rrelu-4.2 {Error on invalid parameter name} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::rrelu -invalid $input
} -returnCodes error -result {Unknown parameter: -invalid}

test rrelu-4.3 {Error on missing parameter value} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::rrelu -input $input -lower
} -returnCodes error -result {Missing value for parameter}

test rrelu-4.4 {Error on invalid tensor handle} -body {
    torch::rrelu "invalid_tensor"
} -returnCodes error -result {Invalid tensor name}

# Functional tests
test rrelu-5.1 {RReLU produces correct output shape} -body {
    set input [torch::randn [list 2 3 4] -dtype float32]
    set result [torch::rrelu $input]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} -result {1}

test rrelu-5.2 {RReLU with bounds} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set result [torch::rrelu -input $input -lower 0.1 -upper 0.3]
    expr {[string match "tensor*" $result]}
} -result {1}

# Test that the functions exist and can be called
test function_existence-1.1 {rrelu function exists} -body {
    info commands torch::rrelu
} -result "::torch::rrelu"

test function_existence-1.2 {rRelu camelCase alias exists} -body {
    info commands torch::rRelu
} -result "::torch::rRelu"

cleanupTests 