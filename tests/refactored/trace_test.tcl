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
test trace-1.1 {Basic positional syntax} {
    set matrix [create_tensor {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {15.0}

test trace-1.2 {Positional syntax with 2x2 matrix} {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {5.0}

test trace-1.3 {Positional syntax with identity matrix} {
    set matrix [create_tensor {{1.0 0.0 0.0} {0.0 1.0 0.0} {0.0 0.0 1.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {3.0}

# Test cases for named syntax
test trace-2.1 {Named parameter syntax} {
    set matrix [create_tensor {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}}]
    set out [torch::trace -input $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {15.0}

test trace-2.2 {Named syntax with 2x2 matrix} {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0}}]
    set out [torch::trace -input $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {5.0}

test trace-2.3 {Named syntax with identity matrix} {
    set matrix [create_tensor {{1.0 0.0 0.0} {0.0 1.0 0.0} {0.0 0.0 1.0}}]
    set out [torch::trace -input $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {3.0}

# Test cases for camelCase alias
test trace-3.1 {CamelCase alias} {
    set matrix [create_tensor {{1.0 2.0 3.0} {4.0 5.0 6.0} {7.0 8.0 9.0}}]
    set out [torch::Trace -input $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {15.0}

test trace-3.2 {CamelCase with 2x2 matrix} {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0}}]
    set out [torch::Trace -input $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {5.0}

# Error handling tests
test trace-4.1 {Missing argument error} -returnCodes error -body {
    torch::trace
} -match glob -result {*Required parameters missing: input tensor required*}

test trace-4.2 {Invalid tensor} -returnCodes error -body {
    torch::trace invalid_tensor
} -match glob -result {*Invalid input tensor*}

test trace-4.3 {Unknown parameter} -returnCodes error -body {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0}}]
    torch::trace -input $matrix -invalid param
} -match glob -result {*Unknown parameter: -invalid*}

test trace-4.4 {Missing parameter value} -returnCodes error -body {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0}}]
    torch::trace -input
} -match glob -result {*Missing value for parameter*}

# Edge cases and different matrix types
test trace-5.1 {Zero matrix} {
    set matrix [create_tensor {{0.0 0.0 0.0} {0.0 0.0 0.0} {0.0 0.0 0.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {0.0}

test trace-5.2 {Large matrix} {
    set matrix [create_tensor {{100.0 200.0 300.0} {400.0 500.0 600.0} {700.0 800.0 900.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {1500.0}

test trace-5.3 {Negative values} {
    set matrix [create_tensor {{-1.0 -2.0} {-3.0 -4.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {-5.0}

test trace-5.4 {Mixed positive and negative} {
    set matrix [create_tensor {{1.0 -2.0 3.0} {-4.0 5.0 -6.0} {7.0 -8.0 9.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {15.0}

test trace-5.5 {Decimal values} {
    set matrix [create_tensor {{1.5 2.5} {3.5 4.5}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {6.0}

# Test with different data types
test trace-6.1 {Int32 input} {
    set matrix [create_tensor {{1 2} {3 4}} int32]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {5}

# Test mathematical properties
test trace-7.1 {Trace of identity matrix equals dimension} {
    set matrix [create_tensor {{1.0 0.0 0.0 0.0} {0.0 1.0 0.0 0.0} {0.0 0.0 1.0 0.0} {0.0 0.0 0.0 1.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {4.0}

test trace-7.2 {Trace of zero matrix is zero} {
    set matrix [create_tensor {{0.0 0.0 0.0} {0.0 0.0 0.0} {0.0 0.0 0.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {0.0}

test trace-7.3 {Trace of diagonal matrix equals sum of diagonal} {
    set matrix [create_tensor {{2.0 0.0 0.0} {0.0 3.0 0.0} {0.0 0.0 4.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {9.0}

# Test with non-square matrices (should work for 2D tensors)
test trace-8.1 {Non-square matrix 2x3} {
    set matrix [create_tensor {{1.0 2.0 3.0} {4.0 5.0 6.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {6.0}

test trace-8.2 {Non-square matrix 3x2} {
    set matrix [create_tensor {{1.0 2.0} {3.0 4.0} {5.0 6.0}}]
    set out [torch::trace $matrix]
    set result [torch::tensor_to_list $out]
    set result
} {5.0}

cleanupTests 