#!/usr/bin/env tclsh
package require tcltest
namespace import tcltest::*

# Load the extension
if {[catch {load ../../build/libtorchtcl.so}]} {
    puts "Failed to load libtorchtcl.so"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper function to get tensor values as list
proc tensor_to_list {tensor} {
    # Convert tensor to float32 first
    set float_tensor [torch::tensor_to -input $tensor -dtype float32 -device cpu]
    set values [torch::tensor_to_list $float_tensor]
    return $values
}

# Basic functionality tests - Positional syntax
test arange-1.1 {Basic arange with end only} {
    set t [torch::arange 5]
    tensor_to_list $t
} {0.0 1.0 2.0 3.0 4.0}

test arange-1.2 {Arange with start and end} {
    set t [torch::arange 2 8]
    tensor_to_list $t
} {2.0 3.0 4.0 5.0 6.0 7.0}

test arange-1.3 {Arange with start, end, and step} {
    set t [torch::arange 1 6 2]
    tensor_to_list $t
} {1.0 3.0 5.0}

test arange-1.4 {Arange with dtype} {
    set t [torch::arange 3 int32]
    tensor_to_list $t
} {0.0 1.0 2.0}

test arange-1.5 {Arange with all parameters} {
    set t [torch::arange 1 6 1 float64 cpu]
    tensor_to_list $t
} {1.0 2.0 3.0 4.0 5.0}

# Named parameter syntax tests
test arange-2.1 {Basic arange with end - named} {
    set t [torch::arange -end 5]
    tensor_to_list $t
} {0.0 1.0 2.0 3.0 4.0}

test arange-2.2 {Arange with start and end - named} {
    set t [torch::arange -start 2 -end 8]
    tensor_to_list $t
} {2.0 3.0 4.0 5.0 6.0 7.0}

test arange-2.3 {Arange with start, end, and step - named} {
    set t [torch::arange -start 1 -end 6 -step 2]
    tensor_to_list $t
} {1.0 3.0 5.0}

test arange-2.4 {Arange with dtype - named} {
    set t [torch::arange -end 3 -dtype int32]
    tensor_to_list $t
} {0.0 1.0 2.0}

test arange-2.5 {Arange with all parameters - named} {
    set t [torch::arange -start 1 -end 6 -step 1 -dtype float64 -device cpu]
    tensor_to_list $t
} {1.0 2.0 3.0 4.0 5.0}

test arange-2.6 {Parameters in different order - named} {
    set t [torch::arange -dtype float32 -end 3 -start 0 -step 1]
    tensor_to_list $t
} {0.0 1.0 2.0}

# Error handling tests
test arange-3.1 {Missing end parameter - named} -body {
    torch::arange -start 1
} -returnCodes error -match glob -result {upper bound and larger bound inconsistent with step sign*}

test arange-3.2 {Unknown parameter - named} -body {
    torch::arange -invalid 5
} -returnCodes error -result {Unknown parameter}

test arange-3.3 {Missing value for parameter - named} -body {
    torch::arange -start
} -returnCodes error -result {Missing value for parameter}

test arange-3.4 {Invalid dtype - named} -body {
    torch::arange -end 3 -dtype invalid
} -returnCodes error -result {Unknown scalar type: invalid}

cleanupTests 