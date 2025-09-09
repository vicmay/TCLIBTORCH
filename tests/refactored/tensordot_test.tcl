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
test tensordot-1.1 {Basic positional syntax} {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    set out [torch::tensordot $a $b {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {70.0}

test tensordot-1.2 {Positional syntax with different dimensions} {
    set a [create_tensor {{1 2 3} {4 5 6}}]
    set b [create_tensor {{7 8 9} {10 11 12}}]
    set out [torch::tensordot $a $b {1 0}]
    set result [torch::tensor_to_list $out]
    set result
} {217.0}

test tensordot-1.3 {Positional syntax with scalar result} {
    set a [create_tensor {1 2 3}]
    set b [create_tensor {4 5 6}]
    set out [torch::tensordot $a $b {0}]
    set result [torch::tensor_to_list $out]
    set result
} {32.0}

# Test cases for named syntax
test tensordot-2.1 {Named parameter syntax} {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    set out [torch::tensordot -a $a -b $b -dims {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {70.0}

test tensordot-2.2 {Named syntax with different dimensions} {
    set a [create_tensor {{1 2 3} {4 5 6}}]
    set b [create_tensor {{7 8 9} {10 11 12}}]
    set out [torch::tensordot -a $a -b $b -dims {1 0}]
    set result [torch::tensor_to_list $out]
    set result
} {217.0}

test tensordot-2.3 {Named syntax with scalar result} {
    set a [create_tensor {1 2 3}]
    set b [create_tensor {4 5 6}]
    set out [torch::tensordot -a $a -b $b -dims {0}]
    set result [torch::tensor_to_list $out]
    set result
} {32.0}

# Test cases for camelCase alias
test tensordot-3.1 {CamelCase alias} {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    set out [torch::tensorDot -a $a -b $b -dims {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {70.0}

test tensordot-3.2 {CamelCase with different dimensions} {
    set a [create_tensor {{1 2 3} {4 5 6}}]
    set b [create_tensor {{7 8 9} {10 11 12}}]
    set out [torch::tensorDot -a $a -b $b -dims {1 0}]
    set result [torch::tensor_to_list $out]
    set result
} {217.0}

# Error handling tests
test tensordot-4.1 {Missing argument error} -returnCodes error -body {
    torch::tensordot
} -match glob -result {*Required parameters missing: a, b, and dims required*}

test tensordot-4.2 {Invalid tensor a} -returnCodes error -body {
    set b [create_tensor {{1 2} {3 4}}]
    torch::tensordot invalid_tensor $b {0 1}
} -match glob -result {*Invalid tensor a*}

test tensordot-4.3 {Invalid tensor b} -returnCodes error -body {
    set a [create_tensor {{1 2} {3 4}}]
    torch::tensordot $a invalid_tensor {0 1}
} -match glob -result {*Invalid tensor b*}

test tensordot-4.4 {Invalid dims format} -returnCodes error -body {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    torch::tensordot $a $b invalid_dims
} -match glob -result {*Invalid dimension value in dims list*}

test tensordot-4.5 {Invalid dimension value} -returnCodes error -body {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    torch::tensordot $a $b {0 invalid}
} -match glob -result {*Invalid dimension value in dims list*}

test tensordot-4.6 {Unknown parameter} -returnCodes error -body {
    set a [create_tensor {{1 2} {3 4}}]
    set b [create_tensor {{5 6} {7 8}}]
    torch::tensordot -a $a -b $b -invalid {0 1}
} -match glob -result {*Unknown parameter: -invalid*}

test tensordot-4.7 {Missing parameter value} -returnCodes error -body {
    set a [create_tensor {{1 2} {3 4}}]
    torch::tensordot -a $a -b
} -match glob -result {*Missing value for parameter*}

# Edge cases and different data types
test tensordot-5.1 {Zero tensor} {
    set a [create_tensor {{0 0} {0 0}}]
    set b [create_tensor {{1 2} {3 4}}]
    set out [torch::tensordot $a $b {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {0.0}

test tensordot-5.2 {Identity matrix tensordot} {
    set a [create_tensor {{1 0} {0 1}}]
    set b [create_tensor {{1 0} {0 1}}]
    set out [torch::tensordot $a $b {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {2.0}

test tensordot-5.3 {Large values} {
    set a [create_tensor {{1000 2000} {3000 4000}}]
    set b [create_tensor {{5000 6000} {7000 8000}}]
    set out [torch::tensordot $a $b {0 1}]
    set result [torch::tensor_to_list $out]
    set result
} {70000000.0}

cleanupTests 