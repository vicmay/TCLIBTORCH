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

# These per-channel accessor functions require truly quantized tensors
# Since PyTorch quantization has complex requirements, we'll focus on testing
# the error conditions and the camelCase aliases exist

# Test cases for q_per_channel_axis
test q_per_channel_axis-1.1 {Error on non-quantized tensor} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::q_per_channel_axis $tensor
} -returnCodes error -result "Invalid quantized tensor"

test q_per_channel_axis-2.1 {CamelCase alias exists} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::qPerChannelAxis $tensor
} -returnCodes error -result "Invalid quantized tensor"

# Test cases for q_per_channel_scales
test q_per_channel_scales-1.1 {Error on non-quantized tensor} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::q_per_channel_scales $tensor
} -returnCodes error -result "Invalid quantized tensor"

test q_per_channel_scales-2.1 {CamelCase alias exists} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::qPerChannelScales $tensor
} -returnCodes error -result "Invalid quantized tensor"

# Test cases for q_per_channel_zero_points
test q_per_channel_zero_points-1.1 {Error on non-quantized tensor} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::q_per_channel_zero_points $tensor
} -returnCodes error -result "Invalid quantized tensor"

test q_per_channel_zero_points-2.1 {CamelCase alias exists} -body {
    set tensor [torch::ones [list 2 3 4]]
    torch::qPerChannelZeroPoints $tensor
} -returnCodes error -result "Invalid quantized tensor"

# Test that the functions exist and can be called
test function_existence-1.1 {q_per_channel_axis function exists} -body {
    info commands torch::q_per_channel_axis
} -result "::torch::q_per_channel_axis"

test function_existence-1.2 {qPerChannelAxis camelCase alias exists} -body {
    info commands torch::qPerChannelAxis
} -result "::torch::qPerChannelAxis"

test function_existence-2.1 {q_per_channel_scales function exists} -body {
    info commands torch::q_per_channel_scales
} -result "::torch::q_per_channel_scales"

test function_existence-2.2 {qPerChannelScales camelCase alias exists} -body {
    info commands torch::qPerChannelScales
} -result "::torch::qPerChannelScales"

test function_existence-3.1 {q_per_channel_zero_points function exists} -body {
    info commands torch::q_per_channel_zero_points
} -result "::torch::q_per_channel_zero_points"

test function_existence-3.2 {qPerChannelZeroPoints camelCase alias exists} -body {
    info commands torch::qPerChannelZeroPoints
} -result "::torch::qPerChannelZeroPoints"

cleanupTests 