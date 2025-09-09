#!/usr/bin/env tclsh
# Test file for torch::batch_norm3d dual-syntax refactor
# Verifies positional syntax, named-parameter syntax, camelCase alias, and error handling.

package require tcltest
namespace import tcltest::*

# Load the LibTorch TCL extension built in ../build
if {[catch {load ../../build/libtorchtcl.so} err]} {
    puts "Failed to load libtorchtcl.so: $err"
    exit 1
}

# Test configuration
configure -testdir [file dirname [info script]]
configure -verbose {pass fail skip error}

# Helper to create random input tensor
proc create_input_tensor {} {
    return [torch::randn_like -input [torch::zeros -shape {2 3 4 4 4} -dtype float32 -device cpu]]
}

# -----------------------------
# 1. Positional Syntax
# -----------------------------
set input [create_input_tensor]

test batchnorm3d-1.1 {Positional syntax basic} {
    set result [torch::batch_norm3d $input]
    expr {[string length $result] > 0}
} 1

# -----------------------------
# 2. Named Parameter Syntax
# -----------------------------
set input2 [create_input_tensor]

test batchnorm3d-2.1 {Named parameter syntax} {
    set result [torch::batch_norm3d -input $input2 -eps 1e-4 -momentum 0.07]
    expr {[string length $result] > 0}
} 1

# -----------------------------
# 3. CamelCase Alias
# -----------------------------
set input3 [create_input_tensor]

test batchnorm3d-3.1 {CamelCase alias} {
    set result [torch::batchNorm3d -input $input3]
    expr {[string length $result] > 0}
} 1

# -----------------------------
# 4. Error Handling
# -----------------------------

test batchnorm3d-4.1 {Error missing input} -body {
    torch::batch_norm3d
} -returnCodes error -match glob -result *

test batchnorm3d-4.2 {Error invalid tensor handle} -body {
    torch::batch_norm3d invalid_handle
} -returnCodes error -match glob -result *

cleanupTests 