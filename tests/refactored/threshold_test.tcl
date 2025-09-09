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
test threshold-1.1 {Basic positional syntax} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 1.0 2.0}

test threshold-1.2 {Positional syntax with negative threshold} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold $input -1.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.0 1.0 2.0}

test threshold-1.3 {Positional syntax with high threshold} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold $input 1.5 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 0.5 2.0}

# Test cases for named syntax
test threshold-2.1 {Named parameter syntax} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold -input $input -threshold 0.0 -value 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 1.0 2.0}

test threshold-2.2 {Named syntax with negative threshold} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold -input $input -threshold -1.0 -value 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.0 1.0 2.0}

test threshold-2.3 {Named syntax with high threshold} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::threshold -input $input -threshold 1.5 -value 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 0.5 2.0}

# Test cases for camelCase alias
test threshold-3.1 {CamelCase alias} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::Threshold -input $input -threshold 0.0 -value 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 1.0 2.0}

test threshold-3.2 {CamelCase with negative threshold} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    set out [torch::Threshold -input $input -threshold -1.0 -value 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.0 1.0 2.0}

# Error handling tests
test threshold-4.1 {Missing argument error} -returnCodes error -body {
    torch::threshold
} -match glob -result {*Required parameters missing: input tensor required*}

test threshold-4.2 {Invalid tensor} -returnCodes error -body {
    torch::threshold invalid_tensor 0.0 0.5
} -match glob -result {*Invalid tensor name*}

test threshold-4.3 {Invalid threshold value} -returnCodes error -body {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    torch::threshold $input invalid 0.5
} -match glob -result {*Invalid threshold value*}

test threshold-4.4 {Invalid value} -returnCodes error -body {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    torch::threshold $input 0.0 invalid
} -match glob -result {*Invalid value*}

test threshold-4.5 {Unknown parameter} -returnCodes error -body {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    torch::threshold -input $input -invalid 0.0 -value 0.5
} -match glob -result {*Unknown parameter: -invalid*}

test threshold-4.6 {Missing parameter value} -returnCodes error -body {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0}]
    torch::threshold -input $input -threshold
} -match glob -result {*Missing value for parameter*}

# Edge cases and different data types
test threshold-5.1 {Zero tensor} {
    set input [create_tensor {0.0 0.0 0.0 0.0 0.0}]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 0.5 0.5}

test threshold-5.2 {All values above threshold} {
    set input [create_tensor {1.0 2.0 3.0 4.0 5.0}]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {1.0 2.0 3.0 4.0 5.0}

test threshold-5.3 {All values below threshold} {
    set input [create_tensor {-5.0 -4.0 -3.0 -2.0 -1.0}]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 0.5 0.5}

test threshold-5.4 {Large values} {
    set input [create_tensor {1000.0 2000.0 3000.0 4000.0 5000.0}]
    set out [torch::threshold $input 2500.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 3000.0 4000.0 5000.0}

test threshold-5.5 {Negative values} {
    set input [create_tensor {-1000.0 -2000.0 -3000.0 -4000.0 -5000.0}]
    set out [torch::threshold $input -2500.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {-1000.0 -2000.0 0.5 0.5 0.5}

test threshold-5.6 {Different data types} {
    set input [create_tensor {-2.0 -1.0 0.0 1.0 2.0} float32]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 1.0 2.0}

# 2D tensor tests
test threshold-6.1 {2D tensor threshold} {
    set input [create_tensor {{-2.0 -1.0} {0.0 1.0} {2.0 3.0}}]
    set out [torch::threshold $input 0.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.5 1.0 2.0 3.0}

test threshold-6.2 {2D tensor with negative threshold} {
    set input [create_tensor {{-2.0 -1.0} {0.0 1.0} {2.0 3.0}}]
    set out [torch::threshold $input -1.0 0.5]
    set result [torch::tensor_to_list $out]
    set result
} {0.5 0.5 0.0 1.0 2.0 3.0}

cleanupTests 