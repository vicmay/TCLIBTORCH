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
test quantize_per_channel-1.1 {Error on missing parameters} -body {
    torch::quantize_per_channel
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_channel input scales zero_points axis dtype"}

test quantize_per_channel-1.2 {Error on insufficient parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::quantize_per_channel $input
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_channel input scales zero_points axis dtype"}

test quantize_per_channel-1.3 {Error on too many parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set scales [torch::ones [list 3] -dtype float32]
    set zero_points [torch::zeros [list 3] -dtype int32]
    torch::quantize_per_channel $input $scales $zero_points 1 qint8 extra
} -returnCodes error -result {wrong # args: should be "torch::quantize_per_channel input scales zero_points axis dtype"}

# CamelCase alias error tests
test quantize_per_channel-2.1 {CamelCase alias error on missing parameters} -body {
    torch::quantizePerChannel
} -returnCodes error -result {wrong # args: should be "torch::quantizePerChannel input scales zero_points axis dtype"}

test quantize_per_channel-2.2 {CamelCase alias error on insufficient parameters} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    torch::quantizePerChannel $input
} -returnCodes error -result {wrong # args: should be "torch::quantizePerChannel input scales zero_points axis dtype"}

# Invalid tensor handle tests
test quantize_per_channel-3.1 {Error on invalid input tensor handle} -body {
    set scales [torch::ones [list 3] -dtype float32]
    set zero_points [torch::zeros [list 3] -dtype int32]
    torch::quantize_per_channel "invalid_tensor" $scales $zero_points 1 qint8
} -returnCodes error -result {Invalid input tensor}

test quantize_per_channel-3.2 {Error on invalid scales tensor handle} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set zero_points [torch::zeros [list 3] -dtype int32]
    torch::quantize_per_channel $input "invalid_scales" $zero_points 1 qint8
} -returnCodes error -result {Invalid scales tensor}

test quantize_per_channel-3.3 {Error on invalid zero_points tensor handle} -body {
    set input [torch::randn [list 2 3] -dtype float32]
    set scales [torch::ones [list 3] -dtype float32]
    torch::quantize_per_channel $input $scales "invalid_zero_points" 1 qint8
} -returnCodes error -result {Invalid zero_points tensor}

test quantize_per_channel-3.4 {CamelCase alias error on invalid tensor handle} -body {
    set scales [torch::ones [list 3] -dtype float32]
    set zero_points [torch::zeros [list 3] -dtype int32]
    torch::quantizePerChannel "invalid_tensor" $scales $zero_points 1 qint8
} -returnCodes error -result {Invalid input tensor}

# Test that the functions exist and can be called
test function_existence-1.1 {quantize_per_channel function exists} -body {
    info commands torch::quantize_per_channel
} -result "::torch::quantize_per_channel"

test function_existence-1.2 {quantizePerChannel camelCase alias exists} -body {
    info commands torch::quantizePerChannel
} -result "::torch::quantizePerChannel"

# Note: Functional tests for quantization are complex due to PyTorch's strict requirements
# for quantization backends and tensor types. We focus on testing camelCase aliases
# and error handling, which are the main goals of this refactoring.

cleanupTests 