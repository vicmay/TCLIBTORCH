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
test rsqrt-1.1 {Basic positional syntax} -body {
    set input [torch::ones [list 2 3] float32]
    set result [torch::rsqrt $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rsqrt-1.2 {Positional syntax with different shapes} -body {
    set input [torch::ones [list 4 5 6] float32]
    set result [torch::rsqrt $input]
    expr {[string match "tensor*" $result]}
} -result {1}

# Test cases for named parameter syntax
test rsqrt-2.1 {Named parameter syntax} -body {
    set input [torch::ones [list 2 3] float32]
    set result [torch::rsqrt -input $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rsqrt-2.2 {Named parameter syntax with tensor alias} -body {
    set input [torch::ones [list 2 3] float32]
    set result [torch::rsqrt -tensor $input]
    expr {[string match "tensor*" $result]}
} -result {1}

# Test cases for camelCase alias
test rsqrt-3.1 {CamelCase alias positional} -body {
    set input [torch::ones [list 2 3] float32]
    set result [torch::rSqrt $input]
    expr {[string match "tensor*" $result]}
} -result {1}

test rsqrt-3.2 {CamelCase alias named parameters} -body {
    set input [torch::ones [list 2 3] float32]
    set result [torch::rSqrt -input $input]
    expr {[string match "tensor*" $result]}
} -result {1}

# Error handling tests
test rsqrt-4.1 {Error on missing parameters} -body {
    torch::rsqrt
} -returnCodes error -result {Usage: torch::rsqrt tensor | torch::rsqrt -input tensor}

test rsqrt-4.2 {Error on invalid parameter name} -body {
    set input [torch::ones [list 2 3] float32]
    torch::rsqrt -invalid $input
} -returnCodes error -result {Unknown parameter: -invalid}

test rsqrt-4.3 {Error on missing parameter value} -body {
    set input [torch::ones [list 2 3] float32]
    torch::rsqrt -input
} -returnCodes error -result {Missing value for parameter}

test rsqrt-4.4 {Error on invalid tensor handle} -body {
    torch::rsqrt "invalid_tensor"
} -returnCodes error -result {Invalid tensor name}

# Functional tests
test rsqrt-5.1 {RSqrt produces correct output shape} -body {
    set input [torch::ones [list 2 3 4] float32]
    set result [torch::rsqrt $input]
    set input_shape [torch::tensor_shape $input]
    set result_shape [torch::tensor_shape $result]
    expr {$input_shape eq $result_shape}
} -result {1}

test rsqrt-5.2 {RSqrt mathematical correctness} -body {
    # Create a single element tensor with value 4.0, rsqrt should give 0.5
    set input [torch::full [list 1] 4.0 float32]
    set result [torch::rsqrt $input]
    set first_value [torch::tensor_item $result]
    # Check if result is approximately 0.5 (1/sqrt(4) = 1/2 = 0.5)
    expr {abs($first_value - 0.5) < 1e-6}
} -result {1}

test rsqrt-5.3 {RSqrt with named parameters} -body {
    set input [torch::ones [list 3 3] float32]
    set result [torch::rsqrt -input $input]
    expr {[string match "tensor*" $result]}
} -result {1}

# Test that the functions exist and can be called
test function_existence-1.1 {rsqrt function exists} -body {
    info commands torch::rsqrt
} -result "::torch::rsqrt"

test function_existence-1.2 {rSqrt camelCase alias exists} -body {
    info commands torch::rSqrt
} -result "::torch::rSqrt"

cleanupTests 