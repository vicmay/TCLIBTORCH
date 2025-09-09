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

# Error handling tests (these commands currently only support positional syntax)
test quantize_per_tensor-1.1 {Error on missing parameters} -body {
    torch::quantize_per_tensor
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_tensor input scale zero_point dtype"}

test quantize_per_tensor-1.2 {Error on insufficient parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::quantize_per_tensor $input 0.1
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_tensor input scale zero_point dtype"}

test quantize_per_tensor-1.3 {Error on too many parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::quantize_per_tensor $input 0.1 128 qint8 extra
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_tensor input scale zero_point dtype"}

# CamelCase alias error tests
test quantize_per_tensor-2.1 {CamelCase alias error on missing parameters} -body {
    torch::quantizePerTensor
} -returnCodes error -result {wrong # args: should be "torch::quantizePerTensor input scale zero_point dtype"}

test quantize_per_tensor-2.2 {CamelCase alias error on insufficient parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::quantizePerTensor $input 0.1
} -returnCodes error -result {wrong # args: should be "torch::quantizePerTensor input scale zero_point dtype"}

# Invalid tensor handle tests
test quantize_per_tensor-3.1 {Error on invalid tensor handle} -body {
    torch::quantize_per_tensor "invalid_tensor" 0.1 128 qint8
} -returnCodes error -result {Invalid input tensor}

test quantize_per_tensor-3.2 {CamelCase alias error on invalid tensor handle} -body {
    torch::quantizePerTensor "invalid_tensor" 0.1 128 qint8
} -returnCodes error -result {Invalid input tensor}

# Test that the functions exist and can be called
test function_existence-1.1 {quantize_per_tensor function exists} -body {
    info commands torch::quantize_per_tensor
} -result "::torch::quantize_per_tensor"

test function_existence-1.2 {quantizePerTensor camelCase alias exists} -body {
    info commands torch::quantizePerTensor
} -result "::torch::quantizePerTensor"

# Note: Functional tests for quantization are complex due to PyTorch's strict requirements
# for quantization backends and tensor types. We focus on testing camelCase aliases
# and error handling, which are the main goals of this refactoring.

cleanupTests 