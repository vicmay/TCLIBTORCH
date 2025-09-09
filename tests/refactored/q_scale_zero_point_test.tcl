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

# These quantization accessor functions require truly quantized tensors
# Since PyTorch quantization has complex requirements, we'll focus on testing
# the error conditions and the camelCase aliases exist

# Test cases for q_scale
test q_scale-1.1 {Error on non-quantized tensor} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::q_scale $tensor
} -returnCodes error -result "Invalid quantized tensor"

test q_scale-2.1 {CamelCase alias exists} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::qScale $tensor
} -returnCodes error -result "Invalid quantized tensor"

# Test cases for q_zero_point
test q_zero_point-1.1 {Error on non-quantized tensor} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::q_zero_point $tensor
} -returnCodes error -result "Invalid quantized tensor"

test q_zero_point-2.1 {CamelCase alias exists} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::qZeroPoint $tensor
} -returnCodes error -result "Invalid quantized tensor"

# Test that the functions exist and can be called
test function_existence-1.1 {q_scale function exists} -body {
    info commands torch::q_scale
} -result "::torch::q_scale"

test function_existence-1.2 {qScale camelCase alias exists} -body {
    info commands torch::qScale
} -result "::torch::qScale"

test function_existence-2.1 {q_zero_point function exists} -body {
    info commands torch::q_zero_point
} -result "::torch::q_zero_point"

test function_existence-2.2 {qZeroPoint camelCase alias exists} -body {
    info commands torch::qZeroPoint
} -result "::torch::qZeroPoint"

cleanupTests 